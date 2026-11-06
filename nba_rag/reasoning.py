"""Structured retrieval and reasoning: from a question to labeled rows with evidence.

Similarity search alone can't answer "season opener", "most total points" or "largest
margin". Every question goes through the same general steps:

1. link_entities / parse_*: teams, players, dates, scores and stats the question names.
2. planner: filters only wording can decide, checked against the question text.
3. retrieval: candidate games, box scores, recaps and notes by SQL.
4. derive_games / filter_games / aggregate_players: order, winners, margins, totals,
   maximums and threshold counts computed in Python.
5. Rows: short labeled tables (G1.., P1.., R1.., N1..), each carrying the values it states
   and the source rows that support it, so the answer step can cite and verify.
"""
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy.engine import Connection

from nba_rag import parsing, planner, retrieval
from nba_rag.catalog import Catalog, Entities, fold, link_entities

MAX_TABLE_ROWS = 8
# Recaps and notes are long; show them only once filters leave a handful of games.
MAX_TEXT_GAMES = 3
GAME_FIELDS = ("score", "winner", "opponent", "margin", "deficit", "date")
# Integer fields Python computes from the tables. Any other int field (e.g. "deficit") is
# a number only the recap text states.
COMPUTED_INTS = {"margin", "games", *parsing.STATS}


def text_only_int_fields(fields: dict) -> list[str]:
    return [k for k, t in fields.items() if t == "int" and k not in COMPUTED_INTS]


@dataclass
class ContextRow:
    """One labeled row shown to the LLM, with the values it states and its evidence."""
    label: str
    line: str
    values: dict
    evidence: list[dict]
    player_id: int | None = None
    game_ids: list[int] = field(default_factory=list)
    rank_value: int | None = None   # what the table is sorted by, to spot ties for first


@dataclass
class Context:
    question: str
    entities: Entities
    plan: planner.Plan
    window: tuple[date, date] | None = None
    games: list[dict] = field(default_factory=list)
    box_rows: list[dict] = field(default_factory=list)
    rows: list[ContextRow] = field(default_factory=list)
    abstain_reason: str | None = None
    # Recaps of every matching game, when there are too many to show at once.
    recaps_by_game: dict[int, dict] = field(default_factory=dict)
    retrieval_notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        e = self.entities
        parts = []
        if e.teams:
            parts.append("Teams named: " + ", ".join(t["name"] for t in e.teams))
        if e.players:
            parts.append("Players named: " + ", ".join(p["full_name"] for p in e.players))
        parts.append(f"Games matching the question: {len(self.games)}")
        p = self.plan
        order = "smallest first" if p.rank == "least" else "largest first"
        what = (f"games with {p.min_value}+ {p.stat}" if p.min_value is not None
                else f"best single game {p.stat}" if p.per_game else f"total {p.stat}")
        for prefix, title in (("G", f"Games (by margin, {order})"), ("P", f"Players over those games (by {what}, {order})"),
                              ("R", "Recaps"), ("N", "Availability notes")):
            block = [f"[{r.label}] {r.line}" for r in self.rows if r.label.startswith(prefix)]
            if block:
                parts.append(f"{title}:\n" + "\n".join(block))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Derived facts
# ---------------------------------------------------------------------------
def derive_games(rows: list[dict], focus_team_id: int | None) -> list[dict]:
    """Add winner, loser, score, margin, the focus team's result, and schedule position.

    game_number counts the focus team's games of the same season phase in date order
    (1 = regular-season opener), so it is only meaningful over a full schedule.
    """
    games, counters = [], {}
    for r in sorted(rows, key=lambda r: (r["game_date"], r["game_id"])):
        home, away = int(r["home_team_id"]), int(r["away_team_id"])
        hp, ap = int(r["home_points"]), int(r["away_points"])
        winner = int(r["winner_team_id"])
        counters[r["season_type"]] = counters.get(r["season_type"], 0) + 1
        g = {
            "game_id": int(r["game_id"]), "date": r["game_date"], "season_type": r["season_type"],
            "label": r.get("label"), "home_team_id": home, "away_team_id": away,
            "winner_id": winner, "loser_id": away if winner == home else home,
            "winner_points": max(hp, ap), "loser_points": min(hp, ap), "margin": abs(hp - ap),
            "game_number": counters[r["season_type"]] if focus_team_id is not None else None,
        }
        g["score"] = f"{g['winner_points']}-{g['loser_points']}"
        if focus_team_id in (home, away):
            g["focus_result"] = "win" if winner == focus_team_id else "loss"
            g["opponent_id"] = away if focus_team_id == home else home
            g["focus_venue"] = "home" if focus_team_id == home else "away"
        games.append(g)
    return games


def filter_games(games: list[dict], plan: planner.Plan, window, scores) -> list[dict]:
    out = []
    for g in games:
        if window and not (window[0] <= g["date"] <= window[1]):
            continue
        if scores and (g["winner_points"], g["loser_points"]) not in scores:
            continue
        if plan.season_phase and g["season_type"] != plan.season_phase:
            continue
        if plan.result != "any" and g.get("focus_result") != plan.result:
            continue
        if plan.venue and g.get("focus_venue") != plan.venue:
            continue
        if plan.min_margin is not None and g["margin"] < plan.min_margin:
            continue
        out.append(g)
    if plan.game_number is not None:
        out = out[plan.game_number - 1:plan.game_number] if plan.game_number >= 1 else []
    elif plan.game_pick == "first":
        out = out[:1]
    elif plan.game_pick == "last":
        out = out[-1:]
    return out


def aggregate_players(box_rows: list[dict], games: list[dict], plan: planner.Plan,
                      player_ids: list[int]) -> list[dict]:
    """Per-player totals of plan.stat over the selected games, best first.

    Each entry has games played, the total, the best single game, and (with a
    threshold) the games that reached it, plus the box rows behind each number. Ties
    break on fewer games, then player_id, so the order is stable.
    """
    stat = plan.stat
    game_ids = {g["game_id"] for g in games}
    players: dict[int, dict] = {}
    for r in box_rows:
        gid, pid, tid = int(r["game_id"]), int(r["player_id"]), int(r["team_id"])
        if gid not in game_ids or (player_ids and pid not in player_ids):
            continue
        if plan.starters_only and not r["starter"]:
            continue
        if plan.bench_only and r["starter"]:
            continue
        # A player the question names is kept whichever side the plan picked.
        if not player_ids and plan.players_from == "focus" and tid != plan.focus_team_id:
            continue
        if not player_ids and plan.players_from == "opponents" and tid == plan.focus_team_id:
            continue
        value = int(r[stat])
        a = players.setdefault(pid, {"player_id": pid, "team_ids": set(), "games": 0, "total": 0,
                                     "best": None, "best_game_id": None, "rows": [], "hit_rows": [], "hit_total": 0})
        a["team_ids"].add(tid)
        a["games"] += 1
        a["total"] += value
        a["rows"].append((gid, pid))
        if a["best"] is None or (value < a["best"] if plan.rank == "least" else value > a["best"]):
            a["best"], a["best_game_id"] = value, gid
        if plan.min_value is not None and value >= plan.min_value:
            a["hit_rows"].append((gid, pid))
            a["hit_total"] += value
    out = list(players.values())
    sign = 1 if plan.rank == "least" else -1
    if plan.min_value is not None:
        out = [a for a in out if a["hit_rows"]]
        out.sort(key=lambda a: (sign * len(a["hit_rows"]), sign * a["hit_total"], a["player_id"]))
    elif plan.per_game:
        out.sort(key=lambda a: (sign * a["best"], a["player_id"]))
    else:
        out.sort(key=lambda a: (sign * a["total"], a["games"], a["player_id"]))
    return out


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def _game_ev(gid: int) -> dict:
    return {"table": "games", "game_id": gid}


def _box_ev(gid: int, pid: int) -> dict:
    return {"table": "box_scores", "game_id": gid, "player_id": pid}


def game_rows(games: list[dict], catalog: Catalog, plan: planner.Plan) -> list[ContextRow]:
    """One row per game, largest margin first (smallest first for "least")."""
    sign = 1 if plan.rank == "least" else -1
    shown = sorted(games, key=lambda g: (sign * g["margin"], g["date"], g["game_id"]))[:MAX_TABLE_ROWS]
    name = lambda tid: catalog.team_by_id[tid]["name"]  # noqa: E731
    rows = []
    for i, g in enumerate(shown, start=1):
        line = (f"{g['date'].isoformat()} {name(g['away_team_id'])} at {name(g['home_team_id'])}: "
                f"{name(g['winner_id'])} beat {name(g['loser_id'])} {g['score']} (margin {g['margin']})")
        if g["season_type"] != "regular":
            line += f" [{g.get('label') or g['season_type']}]"
        values = {"winner": name(g["winner_id"]), "loser": name(g["loser_id"]), "score": g["score"],
                  "margin": g["margin"], "date": g["date"].isoformat()}
        if g.get("focus_result"):
            focus = name(plan.focus_team_id)
            line += (f"; {focus} {'won' if g['focus_result'] == 'win' else 'lost'}, opponent {name(g['opponent_id'])}, "
                     f"{focus} {g['season_type']} game #{g['game_number']}")
            values["opponent"] = name(g["opponent_id"])
        rows.append(ContextRow(label=f"G{i}", line=line, values=values, evidence=[_game_ev(g["game_id"])],
                               game_ids=[g["game_id"]], rank_value=g["margin"]))
    return rows


def player_rows(aggs: list[dict], catalog: Catalog, plan: planner.Plan, games: list[dict]) -> list[ContextRow]:
    stat = plan.stat
    dates = {g["game_id"]: g["date"].isoformat() for g in games}
    rows = []
    for i, a in enumerate(aggs[:MAX_TABLE_ROWS], start=1):
        name = catalog.player_by_id[a["player_id"]]["full_name"]
        teams = "/".join(sorted(catalog.team_by_id[t]["name"] for t in a["team_ids"]))
        line = f"{name} ({teams}): {a['games']} games, {a['total']} total {stat}, best game {a['best']}"
        values = {"player_name": name, stat: a["total"], "games": a["games"]}
        cited, rank_value = a["rows"], a["total"]
        if plan.min_value is not None:
            line += f", {len(a['hit_rows'])} games with {plan.min_value}+ {stat}"
            # Counts and totals then cover just the qualifying games, matching the citations.
            values.update({"games": len(a["hit_rows"]), stat: a["hit_total"]})
            cited, rank_value = a["hit_rows"], len(a["hit_rows"])
        elif plan.per_game:
            line += f" on {dates.get(a['best_game_id'], '?')}"
            values.update({stat: a["best"], "games": 1, "date": dates.get(a["best_game_id"])})
            cited, rank_value = [(a["best_game_id"], a["player_id"])], a["best"]
        evidence = []
        for gid, pid in cited:
            evidence += [_game_ev(gid), _box_ev(gid, pid)]
        rows.append(ContextRow(label=f"P{i}", line=line, values=values, evidence=evidence,
                               player_id=a["player_id"], game_ids=sorted({g for g, _ in cited}),
                               rank_value=rank_value))
    return rows


def text_rows(recaps: list[dict], notes: list[dict], catalog: Catalog) -> list[ContextRow]:
    """Recap and availability-note rows, each cited together with its game."""
    rows = []
    for i, r in enumerate(recaps, start=1):
        gid = int(r["game_id"])
        rows.append(ContextRow(label=f"R{i}", line=f"recap of game {gid}: {r['text']}", values={},
                               evidence=[_game_ev(gid), {"table": "recaps", "recap_id": int(r["recap_id"])}],
                               game_ids=[gid]))
    for i, n in enumerate(notes, start=1):
        gid, pid = int(n["game_id"]), int(n["player_id"])
        who = catalog.player_by_id[pid]["full_name"]
        rows.append(ContextRow(label=f"N{i}",
                               line=f"availability note on {who} for game {gid} (status: {n['status']}): {n['text']}",
                               values={"status": n["status"], "reason": n["reason"]}, player_id=pid,
                               evidence=[_game_ev(gid), {"table": "availability_notes", "note_id": int(n["note_id"])}],
                               game_ids=[gid]))
    return rows


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------
def unexplained_numbers(question: str, entities: Entities, plan: planner.Plan) -> list[int]:
    """Numbers in the question that no date, score, name or filter accounts for.

    A leftover number means the question asked for a condition the plan missed, and
    answering would silently drop it.
    """
    used = {plan.min_margin, plan.min_value, plan.game_number}
    return [n for n in parsing.standalone_numbers(parsing.mask_spans(question, entities.spans)) if n not in used]


def resolve_surname(surname: str, catalog: Catalog, box_rows: list[dict],
                    notes: list[dict]) -> tuple[int | None, str | None]:
    """The one player with this surname in the matching games."""
    involved = {int(r["player_id"]) for r in box_rows} | {int(n["player_id"]) for n in notes}
    matches = sorted(pid for pid in involved if fold(catalog.player_by_id[pid]["surname"]) == surname)
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, f"no player named {surname.title()} in the matching games"
    names = ", ".join(catalog.player_by_id[p]["full_name"] for p in matches)
    return None, f"{surname.title()} is ambiguous in the matching games ({names})"


def build_context(cx: Connection, question: str, answer_spec: dict, catalog: Catalog, llm) -> Context:
    """Run steps 1-5 for one question.

    :param answer_spec: {"fields": {name: type}, "evidence": [table, ...]}
    :param llm: callable(prompt) -> text, used for the plan
    """
    fields = answer_spec["fields"]
    wanted_tables = set(answer_spec.get("evidence", []))
    entities = link_entities(question, catalog)
    plan = planner.validate_plan(llm(planner.plan_prompt(question, entities)), question, entities)
    plan.game_number = parsing.parse_game_ordinal(question)
    plan.stat = parsing.detect_stat(question, fields)
    plan.venue = parsing.detect_venue(question)
    window = parsing.parse_date_window(question, catalog)
    scores = parsing.parse_scores(question)
    # A date or final score already pins the game; otherwise "this season" means the
    # regular season unless the question says playoffs, play-in or Cup final.
    plan.season_phase = parsing.detect_season_phase(question)
    if plan.season_phase is None and not scores and not (window and window[0] == window[1]):
        plan.season_phase = "regular"
    ctx = Context(question=question, entities=entities, plan=plan, window=window)

    if entities.unresolved_names:
        ctx.abstain_reason = f"no player named {', '.join(entities.unresolved_names)} in this season's data"
        return ctx
    leftover = unexplained_numbers(question, entities, plan)
    if leftover:
        ctx.abstain_reason = f"the question mentions {leftover} but no filter used it"
        return ctx

    if entities.teams:
        others = [t["team_id"] for t in entities.teams if t["team_id"] != plan.focus_team_id]
        schedule = derive_games(retrieval.team_games(cx, [plan.focus_team_id]), plan.focus_team_id)
        if others:
            # Head-to-head games keep their position in the focus team's full schedule.
            number = {g["game_id"]: g["game_number"] for g in schedule}
            all_games = derive_games(retrieval.team_games(cx, [plan.focus_team_id, others[0]]), plan.focus_team_id)
            for g in all_games:
                g["game_number"] = number[g["game_id"]]
        else:
            all_games = schedule
    elif entities.players:
        all_games = derive_games(retrieval.player_games(cx, [p["player_id"] for p in entities.players]), None)
    else:
        all_games = derive_games(retrieval.all_games(cx), None)

    if scores or (window and window[0] == window[1]):
        # A single date or a final score identifies the game; "their loss" there only
        # describes it, and applying it could drop the game on a misread.
        plan.result, plan.game_pick = "any", "all"
    games = filter_games(all_games, plan, window, scores)
    ctx.games = games
    if not games:
        ctx.abstain_reason = "no games match the question's teams, dates and filters"
        return ctx

    game_ids = [g["game_id"] for g in games]
    ctx.box_rows = retrieval.box_scores(cx, game_ids)
    if entities.surnames:
        game_notes = retrieval.notes(cx, game_ids)
        for surname in entities.surnames:
            pid, reason = resolve_surname(surname, catalog, ctx.box_rows, game_notes)
            if pid is None:
                ctx.abstain_reason = reason
                return ctx
            if pid not in [p["player_id"] for p in entities.players]:
                entities.players.append(catalog.player_by_id[pid])
    player_ids = [p["player_id"] for p in entities.players]

    wants_players = "player_name" in fields or "box_scores" in wanted_tables
    wants_game_fields = any(k in fields for k in GAME_FIELDS)
    if wants_game_fields or not wants_players or len(games) <= MAX_TEXT_GAMES:
        ctx.rows += game_rows(games, catalog, plan)
    if wants_players:
        ctx.rows += player_rows(aggregate_players(ctx.box_rows, games, plan, player_ids), catalog, plan, games)

    text_only_ints = text_only_int_fields(fields)
    if len(games) <= MAX_TEXT_GAMES:
        recs = retrieval.recaps(cx, game_ids) if "recaps" in wanted_tables else []
        nts = retrieval.notes(cx, game_ids, player_ids) if "availability_notes" in wanted_tables else []
        ctx.rows += text_rows(recs, nts, catalog)
    elif "recaps" in wanted_tables and text_only_ints:
        # A number only recaps state, over many games: answer.py reads them one by one.
        ctx.recaps_by_game = {int(r["game_id"]): r for r in retrieval.recaps(cx, game_ids)}
    elif wanted_tables & {"recaps", "availability_notes"}:
        # Too many games to read: SQL picked the candidates, embeddings pick the closest few.
        recs, nts = [], []
        if "recaps" in wanted_tables:
            top = retrieval.rank_documents(cx, question, "recaps", game_ids, MAX_TEXT_GAMES)
            by_id = {int(r["recap_id"]): r for r in retrieval.recaps(cx, [t["game_id"] for t in top])}
            recs = [by_id[int(t["source_id"])] for t in top]
            ctx.retrieval_notes.append(f"vector-ranked {len(game_ids)} games' recaps, kept {len(recs)}")
        if "availability_notes" in wanted_tables:
            candidates = retrieval.notes(cx, game_ids, player_ids)
            top = retrieval.rank_documents(cx, question, "availability_notes", game_ids, MAX_TEXT_GAMES,
                                           [n["note_id"] for n in candidates])
            by_id = {int(n["note_id"]): n for n in candidates}
            nts = [by_id[int(t["source_id"])] for t in top]
            ctx.retrieval_notes.append(f"vector-ranked {len(candidates)} notes, kept {len(nts)}")
        ctx.rows += text_rows(recs, nts, catalog)
    return ctx
