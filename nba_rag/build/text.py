"""Write the two text corpora from real events: game recaps and availability notes.

The source data has no prose, so it is generated here from play-by-play facts and the
box score's did-not-play comments. Phrasing varies (seeded by id, so builds are
reproducible) so that the pipeline has to read the text rather than pattern-match one
template. Every fact a recap states comes from game_flow or the box scores.
"""
import random
from datetime import date

import pandas as pd

ORDINAL_PERIOD = {1: "first quarter", 2: "second quarter", 3: "third quarter", 4: "fourth quarter"}
SHOT_WORDS = {"Layup": "layup", "Dunk": "dunk", "Hook": "hook shot", "Jump Shot": "jumper", "Tip": "tip-in"}


NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def surname(last_name: str) -> str:
    """'Butler III' -> 'Butler'; recaps refer to players by surname alone."""
    parts = last_name.split()
    while len(parts) > 1 and parts[-1].lower() in NAME_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def possessive(name: str) -> str:
    """"Lakers" -> "Lakers'", "Thunder" -> "Thunder's"."""
    return name + ("'" if name.endswith("s") else "'s")


def long_date(iso: str) -> str:
    d = date.fromisoformat(iso[:10])
    return f"{d:%B} {d.day}, {d.year}"


def period_name(period: int) -> str:
    if period <= 4:
        return ORDINAL_PERIOD[period]
    return "overtime" if period == 5 else f"the {['', '', 'second', 'third', 'fourth'][min(period - 4, 4)]} overtime"


def shot_word(action: str, shot: str) -> str:
    if action == "3pt":
        return "3-pointer"
    if action == "freethrow":
        return "free throw"
    return SHOT_WORDS.get(shot, "basket")


def write_recap(game: pd.Series, flow: pd.Series | None, box: pd.DataFrame,
                teams: dict, players: dict) -> str:
    rng = random.Random(int(game.game_id))
    home, away = teams[int(game.home_team_id)], teams[int(game.away_team_id)]
    winner_home = int(game.winner_team_id) == int(game.home_team_id)
    win, lose = (home, away) if winner_home else (away, home)
    wp, lp = max(game.home_points, game.away_points), min(game.home_points, game.away_points)
    when = long_date(game.game_date)
    ot = ""
    if flow is not None and flow.periods > 4:
        ot = {5: " in overtime", 6: " in double overtime", 7: " in triple overtime"}.get(
            int(flow.periods), f" in {int(flow.periods) - 4} overtimes")
    where = "at home" if winner_home else "on the road"
    stage = f" in {game.label}" if game.season_type in ("playoffs", "play-in", "cup-final") and game.label else ""

    s = []
    s.append(rng.choice([
        f"On {when}, the {win['city']} {win['name']} beat the {lose['city']} {lose['name']} {wp}-{lp} {where}{ot}{stage}.",
        f"The {win['name']} defeated the {lose['name']} {wp}-{lp}{ot} on {when}{stage}, winning {where}.",
        f"{when}: {win['city']} {win['name']} {wp}, {lose['city']} {lose['name']} {lp}{ot}{stage}.",
    ]))

    if flow is not None and pd.notna(flow.halftime_home):
        hh, ha = int(flow.halftime_home), int(flow.halftime_away)
        if hh == ha:
            s.append(f"The teams were tied {hh}-{ha} at halftime.")
        else:
            lead_team = home if hh > ha else away
            s.append(rng.choice([f"The {lead_team['name']} led {max(hh, ha)}-{min(hh, ha)} at halftime.",
                                 f"At the break it was {lead_team['name']} {max(hh, ha)}, "
                                 f"{(away if lead_team is home else home)['name']} {min(hh, ha)}."]))

    if flow is not None:
        deficit = int(flow.winner_max_deficit)
        if deficit == 0:
            s.append(rng.choice([f"The {win['name']} never trailed.", f"The {win['name']} led wire to wire."]))
        else:
            s.append(rng.choice([
                f"The {win['name']} rallied from a {deficit}-point deficit.",
                f"Down by as many as {deficit}, the {win['name']} came back to win.",
                f"The {win['name']} overcame a {deficit}-point hole.",
                f"The {lose['name']} led by as many as {deficit} before the {win['name']} took over.",
            ]))
        if rng.random() < 0.5 and int(flow.winner_max_lead) > 0:
            s.append(f"The {possessive(win['name'])} biggest lead was {int(flow.winner_max_lead)}.")
        if pd.notna(flow.go_ahead_player_id):
            pid = int(flow.go_ahead_player_id)
            p = players.get(pid)
            if p is not None:
                name = p["full_name"] if rng.random() < 0.6 else surname(p["last_name"])
                shot = shot_word(flow.go_ahead_action, flow.go_ahead_shot)
                if int(flow.go_ahead_period) >= 4:
                    left = f"with {flow.go_ahead_clock} left in {period_name(int(flow.go_ahead_period))}".replace(
                        "left in fourth", "left in the fourth")
                    s.append(rng.choice([
                        f"{name}'s {shot} {left} put the {win['name']} ahead for good.",
                        f"{name} hit the go-ahead {shot} {left}.",
                        f"The go-ahead {shot} came from {name} {left}.",
                    ]))
                else:
                    s.append(f"{name}'s {shot} in the {period_name(int(flow.go_ahead_period))} gave the "
                             f"{win['name']} a lead they never gave up.")
        if int(flow.lead_changes) > 0:
            lc, tt = int(flow.lead_changes), int(flow.times_tied)
            s.append(f"There {'was' if lc == 1 else 'were'} {lc} lead change{'' if lc == 1 else 's'} "
                     f"and {tt} tie{'' if tt == 1 else 's'}.")

    for team, verb in ((win, "led the"), (lose, "paced the")):
        rows = box[box.team_id == team["team_id"]].sort_values(["points", "player_id"], ascending=[False, True])
        if rows.empty:
            continue
        top = rows.iloc[0]
        name = players[int(top.player_id)]["full_name"]
        extra = ""
        if top.rebounds >= 10 or top.assists >= 10:
            extra = f", {int(top.rebounds)} rebounds and {int(top.assists)} assists"
        if team is win:
            s.append(rng.choice([f"{name} {verb} {team['name']} with {int(top.points)} points{extra}.",
                                 f"{name} had {int(top.points)} points{extra} for the {team['name']}."]))
        else:
            s.append(rng.choice([f"For the {team['name']}, {name} had {int(top.points)} points{extra}.",
                                 f"{name} {verb} {team['name']} with {int(top.points)} points{extra} in the loss."]))
    return " ".join(s)


# Box score comment code -> status, and the reason wording for the part after it.
STATUS_CODES = {"DNP": "did not play", "DND": "did not dress", "NWT": "not with team"}
REASONS = {
    "coach's decision": "coach's decision",
    "injury/illness": "injury/illness",
    "rest": "rest",
    "personal": "personal reasons",
    "not with team": "not with team",
    "return to competition reconditioning": "return-to-competition reconditioning",
    "team suspension": "team suspension",
}


def parse_comment(comment: str) -> tuple[str, str] | None:
    """'DND - Injury/Illness' -> ('did not dress', 'injury/illness')."""
    raw = str(comment).strip()
    code = raw[:3].upper()
    if code not in STATUS_CODES:
        return None
    rest = raw[3:].lstrip(" -_").replace("_", " ").strip().lower()
    return STATUS_CODES[code], REASONS.get(rest, rest or "unspecified")


def write_note(note_id: int, player: dict, team: dict, opp: dict, game_date: str, status: str, reason: str) -> str:
    rng = random.Random(note_id)
    when = long_date(game_date)
    name = player["full_name"]
    if status == "did not play":
        options = [
            f"{name} did not play in the {possessive(team['name'])} game against the {opp['name']} on {when} ({reason}).",
            f"{name} ({team['name']}) sat out on {when} against the {opp['name']}: {reason}.",
            f"{name} was available but did not play vs. the {opp['name']} on {when}; reason listed: {reason}.",
        ]
        if reason != "coach's decision":
            options = options[:2] + [f"{name} missed the {opp['name']} game on {when} due to {reason}."]
    elif status == "did not dress":
        options = [
            f"{name} did not dress for the {team['name']} against the {opp['name']} on {when} ({reason}).",
            f"{name} was ruled out for {when} vs. the {opp['name']} because of {reason}.",
            f"{team['name']} listed {name} as out against the {opp['name']} on {when}; reason: {reason}.",
        ]
    else:
        options = [
            f"{name} was not with the {team['name']} for their game against the {opp['name']} on {when} ({reason}).",
            f"{name} was away from the team on {when} and missed the {opp['name']} game: {reason}.",
        ]
    return rng.choice(options)
