"""Answer one question in the shape its answer spec asks for, with cited, verified evidence.

An answer spec is {"fields": {name: "int" | "str" | "bool"}, "evidence": [table, ...]}.
Which path answers comes from the spec alone, never from the question's id or wording:

- Computed: no text table is cited. The answer is the top-ranked computed row (players if
  a player is asked for, else games), and every field is a value Python computed.
- Text: the spec cites recaps or availability_notes. The LLM reads the few rows the
  filters left and names the row that states the answer; every field it writes must be
  in that row, and fields the row already holds (status, reason) come from the row.
- Text over many games: a number only the recaps state ("the largest deficit they
  overcame"). Each recap is read for that one number, and Python ranks the results.

Whenever a step can't be verified, the answer abstains with a reason instead of guessing.
"""
import json
import logging
import re

from sqlalchemy.engine import Connection

from nba_rag import llm as llm_mod
from nba_rag.build.text import possessive
from nba_rag.catalog import Catalog, fold
from nba_rag.parsing import standalone_numbers
from nba_rag.reasoning import Context, ContextRow, build_context, game_rows, text_only_int_fields, text_rows

logger = logging.getLogger(__name__)

TEXT_TABLES = ("recaps", "availability_notes")
STOPWORDS = {"the", "and", "for", "with", "was", "his", "her", "due", "from", "because", "game"}
MAX_TEXT_FIELD_WORDS = 8


class Answerer:
    """Counts LLM calls and keeps the prompts, so traces can show what the model saw."""

    def __init__(self, generate=llm_mod.generate):
        self._generate = generate
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return self._generate(prompt)


def coerce(value, type_name: str):
    if type_name == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value)
    if type_name == "int":
        return int(round(float(value)))
    if type_name == "str":
        return str(value)
    raise ValueError(f"unknown field type {type_name!r}")


def abstain_result(spec: dict) -> dict:
    zeros = {"bool": False, "int": 0, "str": ""}
    out = {"answerable": False}
    out.update({k: zeros[t] for k, t in spec["fields"].items()})
    out["evidence"] = []
    return out


def value_supported(key: str, value, row: ContextRow) -> bool:
    """Is a field the model wrote actually stated by the row it cited?

    Computed values must match exactly. Text must be copied from the row, or be a short
    phrase whose content words all appear in it; numbers must appear in it.
    """
    if key in row.values:
        return fold(str(value)).strip() == fold(str(row.values[key])).strip()
    line = fold(row.line)
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return re.search(rf"(?<!\d){value}(?!\d)", line) is not None
    v = fold(str(value)).strip().rstrip(".")
    if v and v in line:
        return True
    if len(v.split()) > MAX_TEXT_FIELD_WORDS:
        return False
    words = [w for w in re.findall(r"[a-z]+", v) if len(w) >= 3 and w not in STOPWORDS]
    return bool(words) and all(re.search(rf"\b{re.escape(w)}", line) for w in words)


def resolve_player(name: str, catalog: Catalog, box_rows: list[dict], game_ids: list[int]) -> int | None:
    """Map a name the model wrote to a player who played in those games (full name, or a unique surname)."""
    name = fold(name).strip()
    in_games = {int(r["player_id"]) for r in box_rows if int(r["game_id"]) in game_ids}
    full = [p for p in in_games if fold(catalog.player_by_id[p]["full_name"]) == name]
    if len(full) == 1:
        return full[0]
    last = name.split()[-1] if name else ""
    by_last = [p for p in in_games if fold(catalog.player_by_id[p]["surname"]).split()[-1] == last]
    return by_last[0] if len(by_last) == 1 else None


def ground_quote(quote: str, row: ContextRow) -> str | None:
    """The row sentence the model's quote came from, allowing small copying slips.

    The sentence sharing the most content words with the quote wins, if it covers at
    least 60% of them. A quote that matches no sentence returns None.
    """
    words = set(re.findall(r"[a-z0-9]{3,}", fold(quote)))
    if not words:
        return None
    best, best_overlap = None, 0.0
    for sentence in split_sentences(row.line.split(": ", 1)[-1]):
        overlap = len(words & set(re.findall(r"[a-z0-9]{3,}", fold(sentence)))) / len(words)
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    return best if best_overlap >= 0.6 else None


def names_player(sentence: str, player: dict) -> bool:
    """Does the sentence name this player, by full name or surname?"""
    return re.search(rf"\b{re.escape(fold(player['surname']))}\b", fold(sentence)) is not None


def players_named_in(sentence: str, catalog: Catalog, box_rows: list[dict], game_ids: list[int]) -> list[int]:
    in_games = sorted({int(r["player_id"]) for r in box_rows if int(r["game_id"]) in game_ids})
    return [p for p in in_games if names_player(sentence, catalog.player_by_id[p])]


def text_prompt(question: str, fields: dict, context: str) -> str:
    # "quote" comes before the answer fields so the model commits to a sentence first.
    shape = {"row": "str", "quote": "str", **fields, "answerable": "bool"}
    return (
        "You answer questions about NBA games from the records below. Game and player tables come from "
        "official box scores; recaps (R rows) and availability notes (N rows) are written reports.\n"
        f"Reply with one JSON object with exactly these fields (types shown): {json.dumps(shape)}\n"
        "Rules:\n"
        "- \"row\" is the label of the recap or note (an R or N row) that states the answer.\n"
        "- \"quote\" is the one sentence from that row that states the answer, copied exactly.\n"
        "- Copy names exactly as that row writes them. For a text field such as a reason, give only the few "
        "words that state it.\n"
        "- Set \"answerable\": false and \"row\": \"\" if no recap or note states the answer.\n"
        "- Teams by nickname only (\"Celtics\"). Scores as \"<winner points>-<loser points>\".\n\n"
        f"Records:\n{context}\n\nQuestion: {question}\nJSON:"
    )


# Made-up examples that teach which direction a team stat runs. Without them a 2B model
# reads any "gap" as both a lead and a deficit.
NUMBER_EXAMPLES = (
    "Read the sentences and give the number asked for, or none if they don't state it. "
    "A team's deficit is how far it was behind at its worst, however that is worded: trailing by N, "
    "down N, or the other team leading by N all mean a deficit of N. Its lead is how far it was ahead. "
    "Points, rebounds and assists belong to players, not to the team. Reply with only a number or none.\n\n"
    "Sentences: The Comets trailed by 15 in the third quarter before winning. The Comets' biggest lead was 6.\n"
    "What was the Comets' deficit?\nAnswer: 15\n\n"
    "Sentences: The Comets trailed by 15 in the third quarter before winning. The Comets' biggest lead was 6.\n"
    "What was the Comets' lead?\nAnswer: 6\n\n"
    "Sentences: The Lancers were up by as much as 9 before the Comets pulled ahead.\n"
    "What was the Comets' deficit?\nAnswer: 9\n\n"
    "Sentences: Moss led the Comets with 30 points. The Comets erased a 12-point gap.\n"
    "What was the Comets' deficit?\nAnswer: 12\n\n"
    "Sentences: The Comets never trailed. Moss scored 24 points for the Comets.\n"
    "What was the Comets' deficit?\nAnswer: none\n\n"
)
# A team's deficit is its opponent's lead, and vice versa. Recaps often say it that way round
# ("the Nets led by 9 before the Raptors took over"), which a 2B model doesn't invert itself.
COUNTERPART = {"deficit": "lead", "lead": "deficit"}


def split_sentences(text_: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text_) if s.strip()]


def team_sentences(recap: str, team: dict) -> list[str]:
    names = [(team["name"], re.IGNORECASE), (team["city"], re.IGNORECASE), (team["abbreviation"], 0)]
    return [s for s in split_sentences(recap)
            if any(re.search(rf"\b{re.escape(n)}\b", s, f) for n, f in names)]


def read_number(ask, sentences: list[str], team_name: str, field_name: str) -> int | None:
    reply = ask(NUMBER_EXAMPLES + f"Sentences: {' '.join(sentences)}\nWhat was the {possessive(team_name)} "
                f"{field_name}?\nAnswer:").strip().lower()
    m = re.match(r"(\d{1,3})\b", reply)
    return int(m.group(1)) if m else None


def rank_recap_numbers(ctx: Context, catalog: Catalog, field_name: str, ask, trace: dict):
    """Read one number per recap for the focus team, verify it, and pick the extreme game.

    Only sentences that name the team and hold a candidate number (not a date, a score, or
    either team's final points) are read, one sentence per call; given several sentences at
    once a 2B model misses most of them. When the model finds no value for the team, a
    sentence naming both teams is asked for the opponent's counterpart stat (its lead, for
    a deficit). A value counts only if it appears in its sentence. A game's value is the
    largest one read from its recap.
    """
    team = catalog.team_by_id.get(ctx.plan.focus_team_id)
    if team is None:
        return None
    counterpart = COUNTERPART.get(field_name)
    found = []
    for game in ctx.games:
        recap = ctx.recaps_by_game.get(game["game_id"])
        if recap is None:
            continue
        final = {game["winner_points"], game["loser_points"]}
        opponent = catalog.team_by_id.get(game.get("opponent_id"))
        values = []
        for sentence in team_sentences(recap["text"], team):
            candidates = set(standalone_numbers(sentence)) - final
            if not candidates:
                continue
            value = read_number(ask, [sentence], team["name"], field_name)
            if value is None and counterpart and opponent and team_sentences(sentence, opponent):
                value = read_number(ask, [sentence], opponent["name"], counterpart)
            if value is not None and value > 0 and value in candidates:
                values.append(value)
        if values:
            found.append((max(values), game, recap))
    trace["per_game_reads"] = [(v, g["game_id"]) for v, g, _ in found]
    if not found:
        return None
    sign = 1 if ctx.plan.rank == "least" else -1
    found.sort(key=lambda x: (sign * x[0], x[1]["date"], x[1]["game_id"]))
    value, game, recap = found[0]
    tied = [g["game_id"] for v, g, _ in found[1:] if v == value]
    if tied:
        trace["ties"] = [game["game_id"]] + tied
    return value, game_rows([game], catalog, ctx.plan)[0], text_rows([recap], [], catalog)[0]


def answer_question(cx: Connection, question: str, spec: dict, catalog: Catalog,
                    generate=llm_mod.generate) -> tuple[dict, dict]:
    """Answer one question. Returns (result, trace); result follows spec's fields."""
    ask = Answerer(generate)
    fields = spec["fields"]
    declared = set(spec.get("evidence", []))
    trace: dict = {}

    def abstain(reason: str):
        trace["abstain_reason"] = reason
        trace["llm_calls"] = ask.calls
        return abstain_result(spec), trace

    ctx = build_context(cx, question, spec, catalog, ask)
    trace.update({"plan": ctx.plan.as_dict(),
                  "teams": [t["name"] for t in ctx.entities.teams],
                  "players": [p["full_name"] for p in ctx.entities.players],
                  "window": [d.isoformat() for d in ctx.window] if ctx.window else None,
                  "games_matched": len(ctx.games),
                  "retrieval": ctx.retrieval_notes})
    if ctx.abstain_reason:
        return abstain(ctx.abstain_reason)
    wants_box = "box_scores" in declared
    if ctx.entities.players and wants_box and not any(r.label.startswith("P") for r in ctx.rows):
        return abstain("the named player has no box score in the matching games")
    rendered = ctx.render()
    trace["context"] = rendered

    result: dict = {}
    support_row = None  # the quoted sentence, on the text path
    text_tables = declared & set(TEXT_TABLES)
    text_only = text_only_int_fields(fields)
    if ctx.recaps_by_game and len(text_only) == 1:
        best = rank_recap_numbers(ctx, catalog, text_only[0], ask, trace)
        if best is None:
            return abstain(f"no recap of the matching games states a {text_only[0]}")
        value, row, recap_row = best
        for key, type_name in fields.items():
            computed = value if key == text_only[0] else row.values.get(key)
            if computed is None:
                return abstain(f"row {row.label} has no {key!r}")
            result[key] = coerce(computed, type_name)
        ctx.rows += [row, recap_row]
        trace["context"] += f"\n[answer read from recap {recap_row.evidence[-1]['recap_id']}]"
    elif text_tables:
        candidates = {r.label: r for r in ctx.rows if r.label[0] in "RN"
                      and any(e["table"] in text_tables for e in r.evidence)}
        if not candidates:
            return abstain("no recap or availability note covers the matching games")
        raw = ask(text_prompt(question, fields, rendered))
        trace["model_reply"] = raw
        parsed = llm_mod.parse_json_object(raw)
        if parsed is None or not coerce(parsed.get("answerable", False), "bool"):
            return abstain("the recaps and notes for the matching games don't state it")
        row = candidates.get(str(parsed.get("row", "")).strip().strip("[]"))
        if row is None:
            return abstain(f"the model cited no recap or note ({parsed.get('row')!r})")
        sentence = ground_quote(str(parsed.get("quote", "")), row)
        if sentence is None:
            return abstain(f"the model's quote is not a sentence of row {row.label}")
        trace["quote"] = sentence
        # Fields read from text must be stated in the quoted sentence, not just somewhere in the row.
        support_row = ContextRow(label=row.label, line=sentence, values=row.values, evidence=row.evidence)
        for key, type_name in fields.items():
            computed = row.values.get(key)
            try:
                result[key] = coerce(computed if computed is not None else parsed[key], type_name)
            except (KeyError, TypeError, ValueError):
                return abstain(f"the model gave no usable {key!r}")
            if key == "player_name":
                continue  # checked against the quoted sentence and the box scores below
            if computed is None and not value_supported(key, result[key], support_row):
                return abstain(f"{key}={result[key]!r} is not stated in the quoted sentence of row {row.label}")
        if "player_name" in fields and row.player_id is None:
            # The player must be the one the quoted sentence names. When the model wrote someone
            # else, a sentence naming exactly one player from the game settles it.
            pid = resolve_player(result["player_name"], catalog, ctx.box_rows, row.game_ids)
            if pid is None or not names_player(sentence, catalog.player_by_id[pid]):
                named = players_named_in(sentence, catalog, ctx.box_rows, row.game_ids)
                if len(named) != 1:
                    return abstain(f"{result['player_name']!r} is not the player the quoted sentence names")
                result["player_name"] = catalog.player_by_id[named[0]]["full_name"]
                trace["name_taken_from_quote"] = True
    else:
        prefix = "P" if "player_name" in fields or (ctx.entities.players and wants_box) else "G"
        candidates = [r for r in ctx.rows if r.label.startswith(prefix)]
        if not candidates:
            return abstain(f"no {'player' if prefix == 'P' else 'game'} rows match")
        row = candidates[0]
        tied = [r for r in candidates[1:] if r.rank_value is not None and r.rank_value == row.rank_value]
        if tied and not ctx.entities.players:
            trace["ties"] = [r.values.get("player_name") or r.game_ids[0] for r in [row] + tied]
        for key, type_name in fields.items():
            computed = row.values.get(key)
            if computed is None:
                return abstain(f"row {row.label} has no computed {key!r}")
            result[key] = coerce(computed, type_name)
    trace["answer_row"] = row.label

    evidence = list(row.evidence)
    relevant = {p["player_id"] for p in ctx.entities.players}
    if "player_name" in result:
        pid = row.player_id or resolve_player(result["player_name"], catalog, ctx.box_rows, row.game_ids)
        if pid is None:
            return abstain(f"{result['player_name']!r} did not play in the cited game")
        if row.player_id is None and not names_player((support_row or row).line, catalog.player_by_id[pid]):
            return abstain(f"{result['player_name']!r} is not named in row {row.label}")
        result["player_name"] = catalog.player_by_id[pid]["full_name"]
        relevant.add(pid)
    if wants_box and row.player_id is None:
        evidence += [{"table": "box_scores", "game_id": int(b["game_id"]), "player_id": int(b["player_id"])}
                     for b in ctx.box_rows if int(b["player_id"]) in relevant and int(b["game_id"]) in row.game_ids]
    # Cite a recap or note shown for the same game when the answer row didn't cover that table.
    covered = {e["table"] for e in evidence}
    for other in ctx.rows:
        if other is row or not set(other.game_ids) & set(row.game_ids):
            continue
        if other.player_id is not None and other.player_id not in relevant:
            continue
        evidence += [e for e in other.evidence if e["table"] in declared - covered]
    grounded = []
    for item in evidence:
        if item["table"] in declared and item not in grounded:
            grounded.append(item)
    if not declared or not declared <= {e["table"] for e in grounded}:
        return abstain("no supporting row for every evidence table the answer must cite")
    trace["llm_calls"] = ask.calls
    return {"answerable": True, **{k: result[k] for k in fields}, "evidence": grounded}, trace
