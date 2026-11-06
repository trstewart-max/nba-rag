"""Embed games, recaps and availability notes into the documents table.

Box scores are deliberately not embedded: they are numbers, and the pipeline answers
numeric questions with SQL and Python. Embeddings earn their place on the text, where a
question like "who hit the go-ahead shot" has to find the right sentence.

usage:
    python -m nba_rag.embed [--limit N]
"""
import argparse
import logging

import sqlalchemy as sa
from sqlalchemy import text

from nba_rag.config import DB_DSN
from nba_rag.llm import DOCUMENT_PREFIX, embed_batch

logger = logging.getLogger(__name__)

# One SELECT per source producing (source_id, game_id, body).
SOURCES = {
    "games": """
        SELECT g.game_id AS source_id, g.game_id,
               to_char(g.game_date, 'FMMonth FMDD, YYYY') || ': ' || a.city || ' ' || a.name || ' at ' ||
               h.city || ' ' || h.name || '. Final score ' || a.name || ' ' || g.away_points || ', ' ||
               h.name || ' ' || g.home_points || '. ' || w.name || ' won. ' ||
               CASE g.season_type WHEN 'regular' THEN 'Regular season game.'
                    ELSE coalesce(g.label, g.season_type) || '.' END AS body
        FROM games g
        JOIN teams h ON h.team_id = g.home_team_id
        JOIN teams a ON a.team_id = g.away_team_id
        JOIN teams w ON w.team_id = g.winner_team_id
        ORDER BY g.game_id""",
    "recaps": "SELECT recap_id AS source_id, game_id, text AS body FROM recaps ORDER BY recap_id",
    "availability_notes": "SELECT note_id AS source_id, game_id, text AS body FROM availability_notes ORDER BY note_id",
}


def main(limit: int | None) -> None:
    eng = sa.create_engine(DB_DSN)
    with eng.begin() as cx:
        cx.execute(text("DELETE FROM documents"))
        for source, sql in SOURCES.items():
            rows = [dict(r) for r in cx.execute(text(sql)).mappings()]
            if limit:
                rows = rows[:limit]
            vectors = embed_batch([DOCUMENT_PREFIX + r["body"] for r in rows])
            cx.execute(
                text("INSERT INTO documents (source, source_id, game_id, body, embedding) "
                     "VALUES (:source, :source_id, :game_id, :body, (:embedding)::vector)"),
                [{**r, "source": source, "embedding": str(v)} for r, v in zip(rows, vectors)],
            )
            logger.info("Embedded %s: %d documents", source, len(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="embed only the first N rows of each source")
    main(parser.parse_args().limit)
