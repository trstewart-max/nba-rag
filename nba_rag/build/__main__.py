"""Build the season's tables and text corpora from the Kaggle download.

usage:
    python -m nba_rag.build

Writes data/processed/{teams,players,games,box_scores,game_flow,recaps,availability_notes}.csv
"""
import logging

import pandas as pd

from nba_rag.build import flow as flow_mod
from nba_rag.build import raw, text
from nba_rag.config import PROCESSED_DIR

logger = logging.getLogger(__name__)


def build_box_scores(rows: pd.DataFrame) -> pd.DataFrame:
    played = rows[rows.comment.isna() & rows.numMinutes.notna() & (rows.numMinutes > 0)]
    return pd.DataFrame({
        "game_id": played.gameId.astype(int),
        "player_id": played.personId.astype(int),
        "team_id": played.playerteamId.astype(int),
        "opponent_team_id": played.opponentteamId.astype(int),
        "starter": played.startingPosition.notna(),
        "minutes": played.numMinutes.round(1),
        "points": played.points.astype(int),
        "rebounds": played.reboundsTotal.astype(int),
        "assists": played.assists.astype(int),
        "steals": played.steals.astype(int),
        "blocks": played.blocks.astype(int),
        "turnovers": played.turnovers.astype(int),
        "threes": played.threePointersMade.astype(int),
        "fg_made": played.fieldGoalsMade.astype(int),
        "fg_attempted": played.fieldGoalsAttempted.astype(int),
        "ft_made": played.freeThrowsMade.astype(int),
        "ft_attempted": played.freeThrowsAttempted.astype(int),
        "plus_minus": played.plusMinusPoints.fillna(0).astype(int),
    }).sort_values(["game_id", "team_id", "player_id"]).reset_index(drop=True)


def build_players(rows: pd.DataFrame) -> pd.DataFrame:
    names = (rows.groupby("personId")[["firstName", "lastName"]]
             .agg(lambda s: s.mode().iat[0] if not s.mode().empty else ""))
    players = names.reset_index().rename(columns={"personId": "player_id", "firstName": "first_name",
                                                  "lastName": "last_name"})
    return players.sort_values("player_id").reset_index(drop=True)


def build_notes(rows: pd.DataFrame, games: pd.DataFrame, teams: dict, players: dict) -> pd.DataFrame:
    dates = games.set_index("game_id").game_date
    out = []
    for r in rows[rows.comment.notna()].sort_values(["gameId", "playerteamId", "personId"]).itertuples():
        parsed = text.parse_comment(r.comment)
        if parsed is None:
            continue
        status, reason = parsed
        note_id = len(out) + 1
        team, opp = teams[int(r.playerteamId)], teams[int(r.opponentteamId)]
        out.append({
            "note_id": note_id, "game_id": int(r.gameId), "player_id": int(r.personId),
            "team_id": int(r.playerteamId), "status": status, "reason": reason,
            "text": text.write_note(note_id, players[int(r.personId)], team, opp, dates[int(r.gameId)], status, reason),
        })
    return pd.DataFrame(out)


def main() -> None:
    src = raw.data_dir()
    logger.info("Reading Kaggle data from %s", src)
    games = raw.load_games(src)
    game_ids = set(games.game_id)
    pbp = raw.load_play_by_play(src, game_ids)
    teams_df = raw.load_teams(src, games, pbp)
    rows = raw.load_box_rows(src, game_ids)
    box = build_box_scores(rows)
    players_df = build_players(rows)
    flows = flow_mod.game_flow(pbp, games)
    logger.info("Games %d, box rows %d, players %d, games with play-by-play facts %d",
                len(games), len(box), len(players_df), len(flows))

    teams = {int(t.team_id): t._asdict() for t in teams_df.itertuples(index=False)}
    players = {int(p.player_id): {**p._asdict(), "full_name": f"{p.first_name} {p.last_name}"}
               for p in players_df.itertuples(index=False)}
    flow_by_game = flows.set_index("game_id")
    box_by_game = {gid: g for gid, g in box.groupby("game_id")}
    recaps = []
    for g in games.itertuples(index=False):
        g = pd.Series(g._asdict())
        f = flow_by_game.loc[g.game_id] if g.game_id in flow_by_game.index else None
        recaps.append({"recap_id": len(recaps) + 1, "game_id": int(g.game_id),
                       "text": text.write_recap(g, f, box_by_game.get(g.game_id, box.iloc[:0]), teams, players)})
    notes = build_notes(rows, games, teams, players)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in (("teams", teams_df), ("players", players_df), ("games", games), ("box_scores", box),
                     ("game_flow", flows), ("recaps", pd.DataFrame(recaps)), ("availability_notes", notes)):
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        logger.info("Wrote %s: %d rows", name, len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
