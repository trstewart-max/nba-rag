"""SQL retrieval of candidate rows, and vector ranking of text within those candidates.

Every query orders by a unique key so results are identical run to run. Vector search is
an exact scan (no ANN index) for the same reason; at ~8k documents it's milliseconds.
"""
from sqlalchemy import text
from sqlalchemy.engine import Connection

from nba_rag.llm import embed_query

GAME_COLUMNS = ("game_id, game_date, season_type, label, home_team_id, away_team_id, "
                "home_points, away_points, winner_team_id")
BOX_COLUMNS = ("game_id, player_id, team_id, opponent_team_id, starter, minutes, points, rebounds, "
               "assists, steals, blocks, turnovers, threes")


def _rows(cx: Connection, sql: str, params: dict | None = None) -> list[dict]:
    return [dict(r) for r in cx.execute(text(sql), params or {}).mappings()]


def team_games(cx: Connection, team_ids: list[int]) -> list[dict]:
    """Games involving every listed team: one team's schedule, or two teams' head-to-heads."""
    sql = f"SELECT {GAME_COLUMNS} FROM games WHERE TRUE"
    params = {}
    for i, tid in enumerate(team_ids):
        sql += f" AND :t{i} IN (home_team_id, away_team_id)"
        params[f"t{i}"] = int(tid)
    return _rows(cx, sql + " ORDER BY tipoff, game_id", params)


def player_games(cx: Connection, player_ids: list[int]) -> list[dict]:
    """Games a player has a box score or an availability note for."""
    return _rows(cx, f"""
        SELECT {GAME_COLUMNS} FROM games WHERE game_id IN (
            SELECT game_id FROM box_scores WHERE player_id = ANY(:p)
            UNION SELECT game_id FROM availability_notes WHERE player_id = ANY(:p))
        ORDER BY tipoff, game_id""", {"p": [int(p) for p in player_ids]})


def all_games(cx: Connection) -> list[dict]:
    return _rows(cx, f"SELECT {GAME_COLUMNS} FROM games ORDER BY tipoff, game_id")


def box_scores(cx: Connection, game_ids: list[int]) -> list[dict]:
    if not game_ids:
        return []
    return _rows(cx, f"SELECT {BOX_COLUMNS} FROM box_scores WHERE game_id = ANY(:g) ORDER BY game_id, player_id",
                 {"g": [int(g) for g in game_ids]})


def recaps(cx: Connection, game_ids: list[int]) -> list[dict]:
    if not game_ids:
        return []
    return _rows(cx, "SELECT recap_id, game_id, text FROM recaps WHERE game_id = ANY(:g) ORDER BY game_id, recap_id",
                 {"g": [int(g) for g in game_ids]})


def notes(cx: Connection, game_ids: list[int], player_ids: list[int] | None = None) -> list[dict]:
    if not game_ids:
        return []
    sql = ("SELECT note_id, game_id, player_id, team_id, status, reason, text FROM availability_notes "
           "WHERE game_id = ANY(:g)")
    params = {"g": [int(g) for g in game_ids]}
    if player_ids:
        sql += " AND player_id = ANY(:p)"
        params["p"] = [int(p) for p in player_ids]
    return _rows(cx, sql + " ORDER BY game_id, note_id", params)


def rank_documents(cx: Connection, question: str, source: str, game_ids: list[int], k: int,
                   source_ids: list[int] | None = None) -> list[dict]:
    """The k documents of one source most similar to the question, among the given games.

    SQL narrows to the candidate games first, then cosine similarity ranks within them.
    Ties break on source_id so the order never changes between runs.
    """
    if not game_ids:
        return []
    sql = ("SELECT source_id, game_id, body, 1 - (embedding <=> (:q)::vector) AS score FROM documents "
           "WHERE source = :s AND game_id = ANY(:g) AND embedding IS NOT NULL")
    params = {"q": str(embed_query(question)), "s": source, "g": [int(g) for g in game_ids], "k": k}
    if source_ids is not None:
        sql += " AND source_id = ANY(:ids)"
        params["ids"] = [int(i) for i in source_ids]
    sql += " ORDER BY embedding <=> (:q)::vector, source_id LIMIT :k"
    return _rows(cx, sql, params)


def search_documents(cx: Connection, question: str, k: int, game_ids: list[int] | None = None) -> list[dict]:
    """Plain similarity search over every document, optionally within some games."""
    sql = ("SELECT source, source_id, game_id, body, 1 - (embedding <=> (:q)::vector) AS score FROM documents "
           "WHERE embedding IS NOT NULL")
    params = {"q": str(embed_query(question)), "k": k}
    if game_ids is not None:
        sql += " AND game_id = ANY(:g)"
        params["g"] = [int(g) for g in game_ids]
    return _rows(cx, sql + " ORDER BY embedding <=> (:q)::vector, source, source_id LIMIT :k", params)
