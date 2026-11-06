"""Load data/processed/*.csv into Postgres, replacing whatever was there.

usage:
    python -m nba_rag.ingest
"""
import logging

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

from nba_rag.config import DB_DSN, PROCESSED_DIR

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS documents, availability_notes, recaps, box_scores, games, players, teams CASCADE;

CREATE TABLE teams (
    team_id      bigint PRIMARY KEY,
    city         text NOT NULL,
    name         text NOT NULL,
    abbreviation text NOT NULL
);
CREATE TABLE players (
    player_id  bigint PRIMARY KEY,
    first_name text NOT NULL,
    last_name  text NOT NULL
);
CREATE TABLE games (
    game_id        bigint PRIMARY KEY,
    game_date      date NOT NULL,
    tipoff         timestamp NOT NULL,
    season_type    text NOT NULL,          -- regular | play-in | playoffs | cup-final
    label          text,                   -- e.g. 'West Conf. Finals Game 3'
    home_team_id   bigint NOT NULL REFERENCES teams,
    away_team_id   bigint NOT NULL REFERENCES teams,
    home_points    int NOT NULL,
    away_points    int NOT NULL,
    winner_team_id bigint NOT NULL REFERENCES teams
);
CREATE TABLE box_scores (
    game_id          bigint NOT NULL REFERENCES games,
    player_id        bigint NOT NULL REFERENCES players,
    team_id          bigint NOT NULL REFERENCES teams,
    opponent_team_id bigint NOT NULL REFERENCES teams,
    starter          boolean NOT NULL,
    minutes          real NOT NULL,
    points int NOT NULL, rebounds int NOT NULL, assists int NOT NULL, steals int NOT NULL,
    blocks int NOT NULL, turnovers int NOT NULL, threes int NOT NULL,
    fg_made int NOT NULL, fg_attempted int NOT NULL, ft_made int NOT NULL, ft_attempted int NOT NULL,
    plus_minus int NOT NULL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX ON box_scores (player_id);
CREATE TABLE recaps (
    recap_id bigint PRIMARY KEY,
    game_id  bigint NOT NULL REFERENCES games,
    text     text NOT NULL
);
CREATE INDEX ON recaps (game_id);
CREATE TABLE availability_notes (
    note_id   bigint PRIMARY KEY,
    game_id   bigint NOT NULL REFERENCES games,
    player_id bigint NOT NULL REFERENCES players,
    team_id   bigint NOT NULL REFERENCES teams,
    status    text NOT NULL,               -- did not play | did not dress | not with team
    reason    text NOT NULL,
    text      text NOT NULL
);
CREATE INDEX ON availability_notes (game_id);
CREATE INDEX ON availability_notes (player_id);
-- Every embedded text, whatever table it came from. Filled by nba_rag.embed.
CREATE TABLE documents (
    source    text NOT NULL,               -- games | recaps | availability_notes
    source_id bigint NOT NULL,
    game_id   bigint NOT NULL REFERENCES games,
    body      text NOT NULL,
    embedding vector(768),
    PRIMARY KEY (source, source_id)
);
CREATE INDEX ON documents (game_id);
"""
# Load order respects the foreign keys.
TABLES = ["teams", "players", "games", "box_scores", "recaps", "availability_notes"]
MAX_BIND_PARAMS = 65535


def main() -> None:
    eng = sa.create_engine(DB_DSN)
    with eng.begin() as cx:
        for stmt in [s for s in SCHEMA.split(";") if s.strip()]:
            cx.execute(text(stmt))
        for table in TABLES:
            df = pd.read_csv(PROCESSED_DIR / f"{table}.csv")
            if table == "games":
                df["label"] = df["label"].where(df["label"].notna(), None)
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
                df["tipoff"] = pd.to_datetime(df["tipoff"])
            chunk = max(1, MAX_BIND_PARAMS // len(df.columns) - 1)
            df.to_sql(table, cx, if_exists="append", index=False, method="multi", chunksize=chunk)
            logger.info("Loaded %s: %d rows", table, len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
