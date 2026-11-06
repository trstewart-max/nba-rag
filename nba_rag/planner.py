"""Turn a question's wording into filters: the LLM proposes, the question text decides.

The model is asked only for the judgments wording needs (whose point of view, won or
lost, thresholds, which game, most or least). Every value it returns is then checked
against the question: a filter survives only if a word in the question expresses it and
any threshold is a number the question actually wrote. A 2B model can't add a filter the
question never asked for.
"""
import re
from dataclasses import asdict, dataclass

from nba_rag.catalog import Entities
from nba_rag.llm import parse_json_object

RESULTS = ("any", "win", "loss")
GAME_PICKS = ("all", "first", "last")

# A filter value the model proposes is kept only if the question has a word for it.
PLAN_CUES = {
    ("result", "win"): r"\b(win|wins|won|victor(y|ies)|beat|over|overcame|came back|comeback|rall(y|ied))\b",
    ("result", "loss"): r"\b(loss|losses|lost|lose|losing|defeat(s|ed)?|beat|beaten|fell|fall)\b",
    ("game_pick", "first"): r"\b(first game|opener|opening|1st game|debut|first matchup|first meeting)\b",
    ("game_pick", "last"): r"\b(last|finale|closing|final game|final regular|most recent)\b",
    ("rank", "least"): r"\b(least|fewest|lowest|smallest|minimum|narrowest|closest|worst)\b",
    ("per_game", True): r"\b(single game|in a game|one game|season[- ]high|single-game|in one night|best game)\b",
    ("starters_only", True): r"\bstart(er|ers|ing|ed)?\b",
    ("bench_only", True): r"\b(bench|reserves?|off the bench|non-starters?)\b",
}
STRICT_WIN = r"\b(win|wins|won|victor(y|ies)|overcame|came back|comeback)\b"
STRICT_LOSS = r"\b(loss|losses|lost|lose|losing|defeat(s|ed)?|fell)\b"
# Ways of writing a threshold, read directly when the model leaves one out.
THRESHOLD_FALLBACKS = {
    "min_margin": (r"\bby (?:at least )?(\d{1,3})(?: or more)?(?: points)?\b"
                   r"|\b(?:wins?|loss(?:es)?|defeats?|victor(?:y|ies)) of (?:at least )?(\d{1,3})"),
    "min_value": (r"\b(\d{1,3})-(?:point|rebound|assist) (?:games?|nights?|performances?|outings?)"
                  r"|\bgames? with (?:at least )?(\d{1,3})(?:\+| or more)"
                  r"|\b(?:at least|or more than) (\d{1,3}) (?:points|rebounds|assists|steals|blocks|threes)"
                  r"|\b(\d{1,3})\+ (?:points|rebounds|assists|steals|blocks|threes)"
                  r"|\b(\d{1,3}) or more (?:points|rebounds|assists|steals|blocks|threes)"),
}

INSTRUCTIONS = """You turn a basketball question into search filters. Reply with one JSON object only:
{"focus_team": "<one of the listed teams, or empty>", "result": "win|loss|any", "min_margin": <int or null>,
 "min_value": <int or null>, "starters_only": <true|false>, "bench_only": <true|false>,
 "game_pick": "all|first|last", "rank": "most|least", "per_game": <true|false>}
- focus_team: the team whose games and point of view the question is about.
- result: whether the focus team won or lost the games asked about.
- min_margin: only games decided by at least this many points.
- min_value: count only games where a player reached at least this many of the stat asked about.
- game_pick: "first" for an opener or first game, "last" for a finale or last game, else "all".
- rank: "least" when the question asks for the fewest, lowest or smallest; otherwise "most".
- per_game: true when the question is about a single game's best (a season high), not a total.

Example: "Which Celtics reserve had the fewest rebounds in wins over the Knicks?" teams: Celtics, Knicks
{"focus_team": "Celtics", "result": "win", "min_margin": null, "min_value": null, "starters_only": false, "bench_only": true, "game_pick": "all", "rank": "least", "per_game": false}
Example: "Who scored 25 or more points most often against the Heat in games the Heat won by at least 8?" teams: Heat
{"focus_team": "Heat", "result": "win", "min_margin": 8, "min_value": 25, "starters_only": false, "bench_only": false, "game_pick": "all", "rank": "most", "per_game": false}
Example: "Who had the most assists in a single game for the Suns?" teams: Suns
{"focus_team": "Suns", "result": "any", "min_margin": null, "min_value": null, "starters_only": false, "bench_only": false, "game_pick": "all", "rank": "most", "per_game": true}
Example: "What was the score of the Pistons' last game?" teams: Pistons
{"focus_team": "Pistons", "result": "any", "min_margin": null, "min_value": null, "starters_only": false, "bench_only": false, "game_pick": "last", "rank": "most", "per_game": false}
"""


@dataclass
class Plan:
    """Filters chosen from the question's wording. All optional."""
    focus_team_id: int | None = None
    result: str = "any"
    min_margin: int | None = None
    min_value: int | None = None
    starters_only: bool = False
    bench_only: bool = False
    players_from: str = "both"      # focus | opponents | both
    game_pick: str = "all"
    rank: str = "most"
    per_game: bool = False
    game_number: int | None = None  # "the 3rd game": position after the other filters
    stat: str = "points"
    season_phase: str | None = None
    venue: str | None = None        # home | away, from the focus team's side

    def as_dict(self) -> dict:
        return asdict(self)


def plan_prompt(question: str, entities: Entities) -> str:
    teams = ", ".join(t["name"] for t in entities.teams) or "none"
    return f"{INSTRUCTIONS}\nQuestion: \"{question}\" teams: {teams}\n"


def _cued(key: str, value, question: str) -> bool:
    cue = PLAN_CUES.get((key, value))
    return cue is None or re.search(cue, question, re.IGNORECASE) is not None


def validate_plan(raw_reply: str | None, question: str, entities: Entities) -> Plan:
    """Keep only the plan values the question text supports; default the rest."""
    raw = parse_json_object(raw_reply)
    plan = Plan()
    if entities.teams:
        # Default to the first team named that isn't written as the opposition.
        own = [t for t in entities.teams if t["team_id"] not in entities.opponent_team_ids]
        plan.focus_team_id = (own or entities.teams)[0]["team_id"]
    if isinstance(raw, dict):
        focus = str(raw.get("focus_team") or "").strip().lower()
        for t in entities.teams:
            if focus and focus in (t["name"].lower(), f"{t['city']} {t['name']}".lower(), t["abbreviation"].lower()):
                plan.focus_team_id = t["team_id"]
    if plan.focus_team_id is not None:
        # Player questions are about the focus team's own players, unless the question
        # names that team only as the opposition ("...against the Thunder").
        alone_as_opponent = len(entities.teams) == 1 and plan.focus_team_id in entities.opponent_team_ids
        plan.players_from = "opponents" if alone_as_opponent else "focus"
    if isinstance(raw, dict):
        if raw.get("result") in RESULTS and _cued("result", raw["result"], question):
            plan.result = raw["result"]
        if raw.get("game_pick") in GAME_PICKS and _cued("game_pick", raw["game_pick"], question):
            plan.game_pick = raw["game_pick"]
        if raw.get("rank") == "least" and _cued("rank", "least", question):
            plan.rank = "least"
        for key in ("starters_only", "bench_only", "per_game"):
            setattr(plan, key, raw.get(key) is True and _cued(key, True, question))
        numbers = {int(n) for n in re.findall(r"(?<!\d)\d{1,3}(?!\d)", question)}
        for key in ("min_margin", "min_value"):
            value = raw.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 and int(value) in numbers:
                setattr(plan, key, int(value))
    # Cues the model missed but the wording makes unambiguous. "Beat" fits both sides, and
    # "who won" asks for the winner rather than filtering to wins, so neither counts.
    if plan.result == "any" and plan.focus_team_id is not None:
        wording = re.sub(r"\b(who|which team) won\b", " ", question, flags=re.IGNORECASE)
        won = re.search(STRICT_WIN, wording, re.IGNORECASE)
        lost = re.search(STRICT_LOSS, wording, re.IGNORECASE)
        if bool(won) != bool(lost):
            plan.result = "win" if won else "loss"
    if plan.game_pick == "all" and re.search(r"\b(opener|opening night)\b", question, re.IGNORECASE):
        plan.game_pick = "first"
    if not plan.starters_only and re.search(r"\bstarters?\b", question, re.IGNORECASE):
        plan.starters_only = True
    if not plan.bench_only and re.search(r"\b(bench|reserves?)\b", question, re.IGNORECASE):
        plan.bench_only = True
    for key, pattern in THRESHOLD_FALLBACKS.items():
        if getattr(plan, key) is None:
            m = re.search(pattern, question, re.IGNORECASE)
            value = int(next(g for g in m.groups() if g)) if m else None
            if value is not None and value not in (plan.min_margin, plan.min_value):
                setattr(plan, key, value)
    if plan.min_margin is not None and plan.min_margin == plan.min_value:
        # One number can't be both thresholds: "by 10" is a margin, anything else a stat value.
        if re.search(rf"by (?:at least )?{plan.min_margin}", question, re.IGNORECASE):
            plan.min_value = None
        else:
            plan.min_margin = None
    return plan
