"""Generate evaluation question sets with gold answers computed straight from the data.

Gold answers are computed here with pandas over data/processed/*.csv, independently of the
pipeline's SQL and reasoning code. Recap questions get their gold from the play-by-play
facts (game_flow.csv), which the pipeline never reads; it has to find them in the text.

Three sets come out, each with its own entities AND its own phrasings:
  dev      - developed against
  heldout  - first run blind; its misses then exposed bugs that were fixed, so it is no
             longer a clean measurement
  test     - written after all fixes and run once; the honest generalization number

usage:
    python -m eval.generate
"""
import json
import random
from datetime import date
from pathlib import Path

import pandas as pd

from nba_rag.config import PROCESSED_DIR

OUT_DIR = Path(__file__).resolve().parent / "sets"
STAT_LABEL = {"points": "points", "rebounds": "rebounds", "assists": "assists"}

# Templates per kind. dev and heldout never share a phrasing.
TEMPLATES = {
    "player_game_stat": {
        "dev": ["How many {stat} did {player} have against the {opp} on {date}?"],
        "heldout": ["In the {date} game against the {opp}, how many {stat} were credited to {player}?"],
        "test": ['Give me {player_pos} {stat} total from the {opp} game on {date}.'],
    },
    "game_result": {
        "dev": ["What was the final score of the {team_pos} game against the {opp} on {date}, and who won?"],
        "heldout": ["Who won when the {team} played the {opp} on {date}, and what was the final score?"],
        "test": ['{opp} vs. {team} on {date}: what was the final score, and which team won?'],
    },
    "team_leader": {
        "dev": ["Who led the {team} in {stat} in their {result} against the {opp} on {date}, and how many did they have?"],
        "heldout": ["Which {team} player had the most {stat} in the {date} {result} to the {opp}, and how many?"],
        "test": ['Who paced the {team} in {stat} when they {result_verb} the {opp} on {date}, and with how many?'],
    },
    "opener_points": {
        "dev": ["How many points did {player} score in the {team_pos} season opener?"],
        "heldout": ["What was {player_pos} point total in the {team_pos} first game of the season?"],
        "test": ['In the {team_pos} opening game, how many points did {player} put up?'],
    },
    "month_total": {
        "dev": ["How many {stat} did {player} have in {month}, and in how many games?"],
        "heldout": ["Across all games in {month}, what were {player_pos} total {stat} and games played?"],
        "test": ['{player}: total {stat} and games played during {month}?'],
    },
    "extreme_margin": {
        "dev": ["What was the {team_pos} largest margin of {outcome} this season, who {verb}, and what was the final score?"],
        "heldout": ["Name the opponent, margin and final score of the {team_pos} biggest {outcome2} of the season."],
        "test": ['{margin_q}'],
    },
    "starter_in_losses": {
        "dev": ["Which {team} starter scored the most total points in games the {team} lost by {n} or more points?"],
        "heldout": ["In {team} losses by at least {n} points, which starter put up the most total points?"],
        "test": ['Among {team} starters, who scored the most combined points in defeats of {n}+ points?'],
    },
    "threshold_games_against": {
        "dev": ["Which player had the most games with {n} or more points against the {team} this season?"],
        "heldout": ["Against the {team}, who reached {n}+ points the most times this season?"],
        "test": ['Who scored at least {n} points against the {team} in the most games this season?'],
    },
    "after_break": {
        "dev": ["How many points did {player} score after the All-Star break, and in how many games?"],
        "heldout": ["Since the All-Star break, what is {player_pos} point total and how many games did it take?"],
        "test": ['{player} after the All-Star break: how many points, over how many games?'],
    },
    "single_game_high": {
        "dev": ["Who had the most {stat} in a single game against the {team} this season, and how many?"],
        "heldout": ["What was the season-high for {stat} by an opponent of the {team}, and who had it?"],
        "test": ['Which opponent had the biggest single-game {stat} total against the {team}, and what was it?'],
    },
    "go_ahead": {
        "dev": ["Who hit the go-ahead basket in the {team_pos} {score} win over the {opp} on {date}?"],
        "heldout": ["In the {team_pos} {score} victory against the {opp} on {date}, who made the go-ahead shot?"],
        "test": ['Who put the {team} ahead for good in their {score} win over the {opp} on {date}?'],
    },
    "comeback": {
        "dev": ["What was the largest deficit the {team} overcame to win a game this season, and who were they playing?"],
        "heldout": ["What was the biggest comeback win for the {team} this season: how many points did they trail by, against whom, and what was the score?"],
        "test": ['Biggest hole the {team} climbed out of to win this season: how many points, which opponent, and what was the final?'],
    },
    "why_missed": {
        "dev": ["Why did {player} miss the {team_pos} game against the {opp} on {date}?"],
        "heldout": ["What kept {player} out of the {date} game between the {team} and the {opp}?"],
        "test": ["{player} didn't play for the {team} against the {opp} on {date}. Why?"],
    },
}
SPECS = {
    "player_game_stat": ("core", None, ["games", "box_scores"]),
    "game_result": ("core", {"winner": "str", "score": "str"}, ["games"]),
    "team_leader": ("core", None, ["games", "box_scores"]),
    "opener_points": ("core", {"points": "int"}, ["games", "box_scores"]),
    "month_total": ("extended", None, ["games", "box_scores"]),
    "extreme_margin": ("extended", {"opponent": "str", "margin": "int", "score": "str"}, ["games"]),
    "starter_in_losses": ("extended", {"player_name": "str", "points": "int"}, ["games", "box_scores"]),
    "threshold_games_against": ("extended", {"player_name": "str", "games": "int"}, ["games", "box_scores"]),
    "after_break": ("extended", {"points": "int", "games": "int"}, ["games", "box_scores"]),
    "single_game_high": ("extended", None, ["games", "box_scores"]),
    "go_ahead": ("stretch", {"player_name": "str"}, ["games", "recaps", "box_scores"]),
    "comeback": ("stretch", {"opponent": "str", "deficit": "int", "score": "str"}, ["games", "recaps"]),
    "why_missed": ("stretch", {"status": "str", "reason": "str"}, ["availability_notes", "games"]),
}


def possessive(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


class Data:
    def __init__(self):
        read = lambda n: pd.read_csv(PROCESSED_DIR / f"{n}.csv")  # noqa: E731
        self.teams = read("teams").set_index("team_id")
        self.players = read("players").set_index("player_id")
        self.games = read("games")
        self.games["game_date"] = pd.to_datetime(self.games.game_date).dt.date
        self.box = read("box_scores")
        self.flow = read("game_flow").set_index("game_id")
        self.notes = read("availability_notes")
        self.recaps = read("recaps").set_index("game_id")
        self.box = self.box.merge(self.games[["game_id", "game_date", "season_type", "home_team_id", "away_team_id",
                                              "home_points", "away_points", "winner_team_id"]], on="game_id")
        self.regular = self.games[self.games.season_type == "regular"]
        days = sorted(set(self.regular.game_date))
        gaps = [(b - a, a, b) for a, b in zip(days, days[1:]) if a.month == 2 or b.month == 2]
        _, before, after = max(gaps)
        self.break_resume = after

    def team(self, tid) -> str:
        return self.teams.loc[int(tid), "name"]

    def player(self, pid) -> str:
        p = self.players.loc[int(pid)]
        return f"{p.first_name} {p.last_name}"

    def score(self, g) -> str:
        return f"{max(g.home_points, g.away_points)}-{min(g.home_points, g.away_points)}"


def fmt_date(d: date, style: str) -> str:
    if style == "dev":
        return f"{d:%B} {d.day}, {d.year}"
    return d.isoformat() if style == "heldout" else f"{d:%b} {d.day}"


def ev_game(gid):
    return {"table": "games", "game_id": int(gid)}


def ev_box(gid, pid):
    return {"table": "box_scores", "game_id": int(gid), "player_id": int(pid)}


def unanswerable():
    return {"answerable": False, "evidence": []}


class Generator:
    def __init__(self, data: Data, split: str, seed: int):
        self.d = data
        self.split = split
        self.rng = random.Random(seed)
        self.items = []

    def add(self, kind: str, fields: dict, gold: dict, **slots) -> None:
        tier, fixed_fields, evidence = SPECS[kind]
        template = self.rng.choice(TEMPLATES[kind][self.split])
        if "player" in slots:
            slots.setdefault("player_pos", possessive(slots["player"]))
        if "team" in slots:
            slots.setdefault("team_pos", possessive(slots["team"]))
        self.items.append({
            "id": f"{self.split}-{len(self.items) + 1:03d}", "tier": tier, "kind": kind,
            "question": template.format(**slots),
            "answer": {"fields": fixed_fields or fields, "evidence": evidence},
            "gold": gold,
        })

    def opponent_slots(self, g, tid):
        opp = g.away_team_id if g.home_team_id == tid else g.home_team_id
        team = self.d.team(tid)
        return {"team": team, "team_pos": possessive(team), "opp": self.d.team(opp)}

    # ---- core -----------------------------------------------------------------------
    def player_game_stat(self, negative=False):
        d = self.d
        if negative:
            # A real player who did not play that game: there's no box score to answer from.
            n = d.notes[d.notes.reason == "injury/illness"].sample(1, random_state=self.rng.randrange(10**6)).iloc[0]
            g = d.games.set_index("game_id").loc[n.game_id]
            stat = "points"
            slots = self.opponent_slots(g, n.team_id)
            self.add("player_game_stat", {stat: "int"}, unanswerable(), stat=stat, player=d.player(n.player_id),
                     opp=slots["opp"], date=fmt_date(g.game_date, self.split))
            return
        rows = d.box[(d.box.season_type == "regular") & (d.box.minutes >= 20)]
        r = rows.sample(1, random_state=self.rng.randrange(10**6)).iloc[0]
        stat = self.rng.choice(list(STAT_LABEL))
        self.add("player_game_stat", {stat: "int"},
                 {"answerable": True, stat: int(r[stat]), "evidence": [ev_game(r.game_id), ev_box(r.game_id, r.player_id)]},
                 stat=stat, player=d.player(r.player_id), opp=d.team(r.opponent_team_id),
                 date=fmt_date(r.game_date, self.split))

    def game_result(self, negative=False):
        d = self.d
        g = d.regular.sample(1, random_state=self.rng.randrange(10**6)).iloc[0]
        tid = self.rng.choice([g.home_team_id, g.away_team_id])
        slots = self.opponent_slots(g, tid)
        when = g.game_date
        gold = {"answerable": True, "winner": d.team(g.winner_team_id), "score": d.score(g),
                "evidence": [ev_game(g.game_id)]}
        if negative:
            # Shift to a day the two teams didn't meet.
            met = set(d.games[((d.games.home_team_id == g.home_team_id) & (d.games.away_team_id == g.away_team_id)) |
                              ((d.games.home_team_id == g.away_team_id) & (d.games.away_team_id == g.home_team_id))].game_date)
            when = when.replace(day=min(28, when.day + 1)) if when.day < 28 else when.replace(day=1)
            while when in met:
                when = when.replace(day=when.day % 28 + 1)
            gold = unanswerable()
        self.add("game_result", {}, gold, date=fmt_date(when, self.split), **slots)

    def team_leader(self):
        d = self.d
        while True:
            g = d.regular.sample(1, random_state=self.rng.randrange(10**6)).iloc[0]
            tid = self.rng.choice([g.home_team_id, g.away_team_id])
            stat = self.rng.choice(list(STAT_LABEL))
            rows = d.box[(d.box.game_id == g.game_id) & (d.box.team_id == tid)].sort_values(stat, ascending=False)
            if len(rows) > 1 and rows.iloc[0][stat] != rows.iloc[1][stat]:
                break
        top = rows.iloc[0]
        result = "win" if g.winner_team_id == tid else "loss"
        slots = self.opponent_slots(g, tid)
        if self.split == "heldout":
            result = "win over" if result == "win" else "loss"
        self.add("team_leader", {"player_name": "str", stat: "int"},
                 {"answerable": True, "player_name": d.player(top.player_id), stat: int(top[stat]),
                  "evidence": [ev_game(g.game_id), ev_box(g.game_id, top.player_id)]},
                 stat=stat, result=result, result_verb="beat" if result.startswith("win") else "lost to",
                 date=fmt_date(g.game_date, self.split), **slots)

    def opener_points(self, negative=False):
        d = self.d
        tid = int(self.rng.choice(list(d.teams.index)))
        sched = d.regular[(d.regular.home_team_id == tid) | (d.regular.away_team_id == tid)].sort_values(["game_date", "game_id"])
        opener = sched.iloc[0]
        in_opener = d.box[(d.box.game_id == opener.game_id) & (d.box.team_id == tid)]
        team_players = set(d.box[(d.box.team_id == tid) & (d.box.season_type == "regular")].player_id)
        slots = {"team": d.team(tid), "team_pos": possessive(d.team(tid))}
        if negative:
            missing = sorted(team_players - set(in_opener.player_id))
            if not missing:
                return self.opener_points(negative)
            pid = self.rng.choice(missing)
            gold = unanswerable()
        else:
            r = in_opener.sample(1, random_state=self.rng.randrange(10**6)).iloc[0]
            pid = r.player_id
            gold = {"answerable": True, "points": int(r.points),
                    "evidence": [ev_game(opener.game_id), ev_box(opener.game_id, pid)]}
        name = d.player(pid)
        self.add("opener_points", {}, gold, player=name, player_pos=possessive(name), **slots)

    # ---- extended -------------------------------------------------------------------
    def month_total(self):
        d = self.d
        reg = d.box[d.box.season_type == "regular"]
        r = reg.sample(1, random_state=self.rng.randrange(10**6)).iloc[0]
        month_start = r.game_date.replace(day=1)
        rows = reg[(reg.player_id == r.player_id) & reg.game_date.map(lambda x: x.replace(day=1) == month_start)]
        stat = self.rng.choice(list(STAT_LABEL))
        name = d.player(r.player_id)
        ev = []
        for x in rows.itertuples():
            ev += [ev_game(x.game_id), ev_box(x.game_id, x.player_id)]
        self.add("month_total", {stat: "int", "games": "int"},
                 {"answerable": True, stat: int(rows[stat].sum()), "games": len(rows), "evidence": ev},
                 stat=stat, player=name, player_pos=possessive(name), month=f"{month_start:%B %Y}")

    def extreme_margin(self):
        d = self.d
        tid = int(self.rng.choice(list(d.teams.index)))
        want_win = self.rng.random() < 0.5
        sched = d.regular[(d.regular.home_team_id == tid) | (d.regular.away_team_id == tid)].copy()
        sched = sched[(sched.winner_team_id == tid) == want_win]
        sched["margin"] = (sched.home_points - sched.away_points).abs()
        best = sched.margin.max()
        top = sched[sched.margin == best].sort_values(["game_date", "game_id"])
        opp = lambda g: d.team(g.away_team_id if g.home_team_id == tid else g.home_team_id)  # noqa: E731
        first = top.iloc[0]
        gold = {"answerable": True, "opponent": opp(first), "margin": int(best), "score": d.score(first),
                "evidence": [ev_game(first.game_id)],
                "accept": [{"opponent": opp(g), "score": d.score(g), "evidence": [ev_game(g.game_id)]}
                           for g in top.iloc[1:].itertuples()]}
        team = d.team(tid)
        self.add("extreme_margin", {}, gold, team=team, team_pos=possessive(team),
                 outcome="victory" if want_win else "defeat", outcome2="win" if want_win else "loss",
                 verb="did they beat" if want_win else "beat them",
                 margin_q=(f"Who did the {team} beat by the most points this season, by how much, and what was the score?"
                           if want_win else
                           f"Who beat the {team} by the most points this season, by how much, and what was the score?"))

    def starter_in_losses(self):
        d = self.d
        while True:
            tid = int(self.rng.choice(list(d.teams.index)))
            n = self.rng.choice([10, 15, 20])
            reg = d.box[(d.box.season_type == "regular") & (d.box.team_id == tid) & (d.box.winner_team_id != tid)]
            reg = reg[(reg.home_points - reg.away_points).abs() >= n]
            starters = reg[reg.starter]
            if starters.empty:
                continue
            totals = starters.groupby("player_id").points.sum().sort_values(ascending=False)
            if len(totals) > 1 and totals.iloc[0] == totals.iloc[1]:
                continue
            break
        pid = totals.index[0]
        ev = []
        for x in starters[starters.player_id == pid].itertuples():
            ev += [ev_game(x.game_id), ev_box(x.game_id, pid)]
        self.add("starter_in_losses", {}, {"answerable": True, "player_name": d.player(pid),
                                          "points": int(totals.iloc[0]), "evidence": ev},
                 team=d.team(tid), n=n)

    def threshold_games_against(self):
        d = self.d
        while True:
            tid = int(self.rng.choice(list(d.teams.index)))
            n = self.rng.choice([25, 30, 35])
            rows = d.box[(d.box.season_type == "regular") & (d.box.opponent_team_id == tid) & (d.box.points >= n)]
            counts = rows.groupby("player_id").size().sort_values(ascending=False)
            if counts.empty or (len(counts) > 1 and counts.iloc[0] == counts.iloc[1]):
                continue
            break
        pid = counts.index[0]
        ev = []
        for x in rows[rows.player_id == pid].itertuples():
            ev += [ev_game(x.game_id), ev_box(x.game_id, pid)]
        self.add("threshold_games_against", {}, {"answerable": True, "player_name": d.player(pid),
                                                "games": int(counts.iloc[0]), "evidence": ev},
                 team=d.team(tid), n=n)

    def after_break(self):
        d = self.d
        rows = d.box[(d.box.season_type == "regular") & (d.box.game_date >= d.break_resume)]
        pid = rows.groupby("player_id").points.sum().sort_values(ascending=False).index[self.rng.randrange(150)]
        mine = rows[rows.player_id == pid]
        ev = []
        for x in mine.itertuples():
            ev += [ev_game(x.game_id), ev_box(x.game_id, pid)]
        name = d.player(pid)
        self.add("after_break", {}, {"answerable": True, "points": int(mine.points.sum()), "games": len(mine),
                                     "evidence": ev}, player=name, player_pos=possessive(name))

    def single_game_high(self):
        d = self.d
        while True:
            tid = int(self.rng.choice(list(d.teams.index)))
            stat = self.rng.choice(list(STAT_LABEL))
            rows = d.box[(d.box.season_type == "regular") & (d.box.opponent_team_id == tid)].sort_values(stat, ascending=False)
            if rows.iloc[0][stat] != rows.iloc[1][stat]:
                break
        top = rows.iloc[0]
        self.add("single_game_high", {"player_name": "str", stat: "int"},
                 {"answerable": True, "player_name": d.player(top.player_id), stat: int(top[stat]),
                  "evidence": [ev_game(top.game_id), ev_box(top.game_id, top.player_id)]},
                 stat=stat, team=d.team(tid))

    # ---- stretch --------------------------------------------------------------------
    def go_ahead(self):
        d = self.d
        f = d.flow[(d.flow.go_ahead_period >= 4) & d.flow.go_ahead_action.isin(["2pt", "3pt"])]
        gid = int(self.rng.choice(sorted(f.index)))
        g = d.games.set_index("game_id").loc[gid]
        pid = int(f.loc[gid, "go_ahead_player_id"])
        slots = self.opponent_slots(g, g.winner_team_id)
        self.add("go_ahead", {}, {"answerable": True, "player_name": d.player(pid),
                                  "evidence": [ev_game(gid), {"table": "recaps", "recap_id": int(d.recaps.loc[gid, "recap_id"])},
                                               ev_box(gid, pid)]},
                 score=d.score(g), date=fmt_date(g.game_date, self.split), **slots)

    def comeback(self):
        d = self.d
        tid = int(self.rng.choice(list(d.teams.index)))
        wins = d.regular[d.regular.winner_team_id == tid]
        wins = wins[wins.game_id.isin(d.flow.index)].copy()
        wins["deficit"] = wins.game_id.map(d.flow.winner_max_deficit)
        best = wins.deficit.max()
        top = wins[wins.deficit == best].sort_values(["game_date", "game_id"])
        opp = lambda g: d.team(g.away_team_id if g.home_team_id == tid else g.home_team_id)  # noqa: E731
        ev = lambda g: [ev_game(g.game_id), {"table": "recaps", "recap_id": int(d.recaps.loc[g.game_id, "recap_id"])}]  # noqa: E731
        first = top.iloc[0]
        self.add("comeback", {}, {"answerable": True, "opponent": opp(first), "deficit": int(best),
                                  "score": d.score(first), "evidence": ev(first),
                                  "accept": [{"opponent": opp(g), "score": d.score(g), "evidence": ev(g)}
                                             for g in top.iloc[1:].itertuples()]},
                 team=d.team(tid))

    def why_missed(self, negative=False):
        d = self.d
        reasons = d.notes if self.rng.random() < 0.3 else d.notes[d.notes.reason != "coach's decision"]
        n = reasons.sample(1, random_state=self.rng.randrange(10**6)).iloc[0]
        g = d.games.set_index("game_id").loc[n.game_id]
        pid, tid = n.player_id, n.team_id
        gold = {"answerable": True, "status": n.status, "reason": n.reason,
                "evidence": [{"table": "availability_notes", "note_id": int(n.note_id)}, ev_game(n.game_id)]}
        if negative:
            # Ask about a teammate who actually played.
            played = d.box[(d.box.game_id == n.game_id) & (d.box.team_id == tid)]
            pid = played.sample(1, random_state=self.rng.randrange(10**6)).iloc[0].player_id
            gold = unanswerable()
        slots = self.opponent_slots(g, tid)
        self.add("why_missed", {}, gold, player=d.player(pid), date=fmt_date(g.game_date, self.split), **slots)

    def unknown_player(self):
        """A plausible name that isn't a player this season."""
        d = self.d
        firsts, lasts = list(d.players.first_name), list(d.players.last_name)
        full = {f"{p.first_name} {p.last_name}" for p in d.players.itertuples()}
        while True:
            name = f"{self.rng.choice(firsts)} {self.rng.choice(lasts)}"
            if name not in full and " " not in name.split(" ", 1)[1]:
                break
        g = d.regular.sample(1, random_state=self.rng.randrange(10**6)).iloc[0]
        self.add("player_game_stat", {"points": "int"}, unanswerable(), stat="points", player=name,
                 opp=d.team(g.home_team_id), date=fmt_date(g.game_date, self.split))


def build(split: str, seed: int, per_kind: int) -> list[dict]:
    gen = Generator(Data(), split, seed)
    for _ in range(per_kind):
        for kind in ("player_game_stat", "game_result", "team_leader", "opener_points", "month_total",
                     "extreme_margin", "starter_in_losses", "threshold_games_against", "after_break",
                     "single_game_high", "go_ahead", "comeback", "why_missed"):
            getattr(gen, kind)()
    # Questions the data can't answer, mixed into the same kinds.
    gen.player_game_stat(negative=True)
    gen.game_result(negative=True)
    gen.opener_points(negative=True)
    gen.why_missed(negative=True)
    gen.unknown_player()
    return gen.items


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, seed in (("dev", 7), ("heldout", 2026), ("test", 99)):
        items = build(split, seed, per_kind=3)
        with open(OUT_DIR / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1, default=int)
        print(f"{split}: {len(items)} questions")


if __name__ == "__main__":
    main()
