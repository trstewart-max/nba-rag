"""Load the Kaggle download and cut it down to one season of clean tables."""
import logging
import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from nba_rag.config import KAGGLE_DATASET, NBA_DATA_DIR, SEASON_FIRST_DAY, SEASON_LAST_DAY

logger = logging.getLogger(__name__)

# Kaggle gameType -> our season_type. Preseason and All-Star games are dropped.
SEASON_TYPES = {
    "Regular Season": "regular",
    "Play-in Tournament": "play-in",
    "Playoffs": "playoffs",
    "Emirates NBA Cup": "cup-final",
}
PBP_COLUMNS = ["gameId", "period", "clock", "orderNumber", "actionType", "subType",
               "personId", "teamId", "scoreHome", "scoreAway"]


def data_dir() -> Path:
    if NBA_DATA_DIR:
        return Path(NBA_DATA_DIR)
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def _in_season(stamps: pd.Series) -> pd.Series:
    day = stamps.astype(str).str[:10]
    return (day >= SEASON_FIRST_DAY.isoformat()) & (day <= SEASON_LAST_DAY.isoformat())


def load_games(src: Path) -> pd.DataFrame:
    g = pd.read_csv(src / "Games.csv", low_memory=False)
    g = g[_in_season(g.gameDateTimeEst) & g.gameType.isin(SEASON_TYPES)].copy()
    label = (g.gameLabel.fillna("") + " " + g.gameSubLabel.fillna("")).str.strip()
    games = pd.DataFrame({
        "game_id": g.gameId.astype(int),
        "game_date": g.gameDateTimeEst.str[:10],
        "tipoff": g.gameDateTimeEst,
        "season_type": g.gameType.map(SEASON_TYPES),
        "label": label,
        "home_team_id": g.hometeamId.astype(int),
        "away_team_id": g.awayteamId.astype(int),
        "home_points": g.homeScore.astype(int),
        "away_points": g.awayScore.astype(int),
        "winner_team_id": g.winner.astype(int),
    })
    return games.sort_values(["tipoff", "game_id"]).reset_index(drop=True)


def load_teams(src: Path, games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """The 30 teams that played this season, named as the Games file names them.

    Abbreviations come from the play-by-play tricodes (e.g. SAS), which are the ones
    broadcasts use; TeamHistories has a few stale ones.
    """
    g = pd.read_csv(src / "Games.csv", low_memory=False, usecols=["gameId", "hometeamId", "hometeamCity", "hometeamName"])
    g = g[g.gameId.isin(games.game_id)]
    teams = (g.rename(columns={"hometeamId": "team_id", "hometeamCity": "city", "hometeamName": "name"})
              [["team_id", "city", "name"]].drop_duplicates("team_id"))
    codes = (pbp.dropna(subset=["teamTricode"]).assign(team_id=lambda d: d.teamId.astype(int))
             .groupby("team_id").teamTricode.agg(lambda s: s.mode().iat[0]))
    teams["abbreviation"] = teams.team_id.map(codes)
    return teams.sort_values("team_id").reset_index(drop=True)


def load_box_rows(src: Path, game_ids: set[int]) -> pd.DataFrame:
    """Every player row for the season's games, including did-not-play rows."""
    cols = ["firstName", "lastName", "personId", "gameId", "playerteamId", "opponentteamId", "numMinutes",
            "points", "assists", "blocks", "steals", "reboundsTotal", "turnovers", "threePointersMade",
            "fieldGoalsMade", "fieldGoalsAttempted", "freeThrowsMade", "freeThrowsAttempted",
            "plusMinusPoints", "comment", "startingPosition"]
    p = pd.read_csv(src / "PlayerStatistics.csv", low_memory=False, usecols=cols)
    p = p[p.gameId.isin(game_ids) & p.personId.notna()].copy()
    p["personId"] = p.personId.astype(int)
    # A few rows store numbers as strings; read them all as numbers.
    for col in cols[5:-2]:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p["comment"] = p.comment.where(p.comment.astype(str).str.strip() != "", None)
    return p


def load_play_by_play(src: Path, game_ids: set[int]) -> pd.DataFrame:
    """Play-by-play rows for the season's games (the file covers every season, ~1 GB)."""
    dataset = ds.dataset(src / "PlayByPlay.parquet")
    ids = pa_strings(sorted(game_ids))
    table = dataset.to_table(columns=PBP_COLUMNS + ["teamTricode"], filter=pc.field("gameId").isin(ids))
    pbp = table.to_pandas()
    pbp["gameId"] = pbp.gameId.astype(int)
    logger.info("Loaded %d play-by-play rows for %d games", len(pbp), pbp.gameId.nunique())
    return pbp


def pa_strings(values) -> list[str]:
    # gameId is stored as a string in the parquet file; some rows zero-pad it.
    out = []
    for v in values:
        out += [str(v), f"{int(v):010d}"]
    return out


def ascii_fold(s: str) -> str:
    """'Jokić' -> 'Jokic'. Used for matching, never for display."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
