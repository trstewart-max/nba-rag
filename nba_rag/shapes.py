"""Answer shapes for free-text chat, where a question arrives without an answer spec.

The chat endpoint picks one of these with a wording rule (route) or, failing that, the LLM,
then answers through exactly the same path as the evaluation sets.
"""
import re

GAMES, BOX, RECAPS, NOTES = "games", "box_scores", "recaps", "availability_notes"


def _shape(description: str, fields: dict, evidence: list[str]) -> dict:
    return {"description": description, "spec": {"fields": fields, "evidence": evidence}}


SHAPES = {
    "player_stat": _shape("how many points/rebounds/assists a named player had, in one game or over a period",
                          {"points": "int", "games": "int"}, [GAMES, BOX]),
    "leader": _shape("which player scored the most / led a team in a stat, in a game or over a period",
                     {"player_name": "str", "points": "int"}, [GAMES, BOX]),
    "most_games": _shape("which player had the most games reaching some number of a stat",
                         {"player_name": "str", "games": "int"}, [GAMES, BOX]),
    "game_result": _shape("the final score or winner of a game",
                          {"winner": "str", "score": "str"}, [GAMES]),
    "margin": _shape("a team's largest or smallest win or loss: opponent, margin and score",
                     {"opponent": "str", "margin": "int", "score": "str"}, [GAMES]),
    "comeback": _shape("the largest deficit a team overcame to win, from the game recaps",
                       {"opponent": "str", "deficit": "int", "score": "str"}, [GAMES, RECAPS]),
    "recap_play": _shape("which player made a play described in a recap, e.g. the go-ahead basket",
                         {"player_name": "str"}, [GAMES, RECAPS, BOX]),
    "availability": _shape("why a player missed a game, or whether they were available",
                           {"status": "str", "reason": "str"}, [NOTES, GAMES]),
}

# Wording that settles the shape outright, checked in order. A 2B model routes these
# unreliably ("who had the most rebounds" -> player_stat), and a rule is easy to audit.
# Questions no rule matches fall back to the LLM.
ROUTES = [
    ("availability", r"\bwhy\b.*\b(miss|out|sit|sat|absent)|\b(miss(ed)?|sat out|ruled out)\b.*\bgame\b|\bwhat kept\b"),
    ("recap_play", r"\bgo-ahead\b|\bgame-winn\w*\b|\bwinning (shot|basket|bucket)\b|\bput .* ahead for good\b"),
    ("comeback", r"\b(deficit|comeback|came back|overcame|overcome|rall(y|ied))\b"),
    ("margin", r"\b(largest|biggest|smallest|narrowest|worst|best)\b.*\b(margin|win|loss|defeat|victory|blowout)\b"),
    ("most_games", r"\bmost (games|times|often)\b|\bhow many games\b.*\b(or more|\d+\+)"),
    ("game_result", r"\bfinal score\b|\bwho won\b|\bwho beat\b|\bwhat was the score\b"),
    ("leader", r"^\s*(who|which)\b"),
    ("player_stat", r"\bhow many\b"),
]


def route(question: str) -> str | None:
    for kind, pattern in ROUTES:
        if re.search(pattern, question, re.IGNORECASE):
            return kind
    return None
