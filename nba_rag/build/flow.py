"""Game-flow facts computed from play-by-play: leads, comebacks, the go-ahead basket.

These facts are what the recaps are written from, and what the eval's gold answers for
recap questions are computed from. The pipeline itself never reads this table; it has to
find the same facts in the recap text.
"""
import re

import pandas as pd

SCORING_ACTIONS = {"2pt", "3pt", "freethrow"}


def format_clock(clock: str) -> str:
    """'PT01M05.20S' -> '1:05'; seconds under a minute keep one decimal ('0:04.2')."""
    m = re.match(r"PT(\d+)M([\d.]+)S", str(clock))
    if not m:
        return ""
    minutes, seconds = int(m.group(1)), float(m.group(2))
    if minutes == 0 and seconds < 10 and seconds != int(seconds):
        return f"0:{seconds:04.1f}"
    return f"{minutes}:{int(seconds):02d}"


def game_flow(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row of flow facts per game whose play-by-play ends on the official final score."""
    finals = games.set_index("game_id")
    rows = []
    for gid, plays in pbp.groupby("gameId", sort=True):
        if gid not in finals.index:
            continue
        game = finals.loc[gid]
        plays = plays.sort_values("orderNumber")
        home, away = plays.scoreHome.to_numpy(), plays.scoreAway.to_numpy()
        if len(plays) == 0 or home[-1] != game.home_points or away[-1] != game.away_points:
            continue  # incomplete or inconsistent feed; better no recap facts than wrong ones
        margin = home.astype(int) - away.astype(int)
        winner_is_home = int(game.winner_team_id) == int(game.home_team_id)
        win_margin = margin if winner_is_home else -margin

        changed = (pd.Series(home).diff().fillna(home[0]) != 0) | (pd.Series(away).diff().fillna(away[0]) != 0)
        scoring = plays[changed.to_numpy()]
        signs = [s for s in (pd.Series(margin[changed.to_numpy()]).apply(lambda x: (x > 0) - (x < 0))) if s != 0]
        lead_changes = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        times_tied = int(((margin[changed.to_numpy()]) == 0).sum())

        first_half = plays[plays.period <= 2]
        half_home = int(first_half.scoreHome.iloc[-1]) if len(first_half) else None
        half_away = int(first_half.scoreAway.iloc[-1]) if len(first_half) else None

        # The go-ahead score: the last scoring play that took the winner from not leading
        # to leading. After it they never trailed or were tied again.
        win_after = win_margin[changed.to_numpy()]
        go_ahead = None
        before = 0
        for i, (idx, play) in enumerate(scoring.iterrows()):
            if before <= 0 < win_after[i]:
                go_ahead = play
            before = win_after[i]
        flow = {
            "game_id": int(gid),
            "home_max_lead": int(max(margin.max(), 0)),
            "away_max_lead": int(max(-margin.min(), 0)),
            "lead_changes": int(lead_changes),
            "times_tied": times_tied,
            "halftime_home": half_home,
            "halftime_away": half_away,
            "periods": int(plays.period.max()),
        }
        flow["winner_max_lead"] = flow["home_max_lead"] if winner_is_home else flow["away_max_lead"]
        flow["winner_max_deficit"] = flow["away_max_lead"] if winner_is_home else flow["home_max_lead"]
        if go_ahead is not None and go_ahead.actionType in SCORING_ACTIONS and str(go_ahead.personId) not in ("", "0"):
            flow.update({
                "go_ahead_player_id": int(go_ahead.personId),
                "go_ahead_action": go_ahead.actionType,
                "go_ahead_shot": str(go_ahead.subType or ""),
                "go_ahead_period": int(go_ahead.period),
                "go_ahead_clock": format_clock(go_ahead.clock),
            })
        rows.append(flow)
    out = pd.DataFrame(rows)
    for col in ("go_ahead_player_id", "go_ahead_period"):
        if col in out:
            out[col] = out[col].astype("Int64")
    return out
