"""HTTP API over the same answer path the evaluation uses, plus the static site.

    uvicorn nba_rag.server:app --reload    then open http://localhost:8000

POST /api/ask {"question": "...", "kind": optional shape name}
  A chat question has no answer spec, so a wording rule (or, failing that, the LLM) picks
  one of shapes.SHAPES, unless the caller names one; answer.answer_question then answers it exactly as it answers the eval.
"""
import logging
from pathlib import Path

import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from nba_rag import llm
from nba_rag.answer import answer_question
from nba_rag.catalog import Catalog, load_catalog
from nba_rag.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from nba_rag.parsing import STATS, detect_stat
from nba_rag.shapes import SHAPES, route

logger = logging.getLogger(__name__)
app = FastAPI(title="NBA RAG")
eng = sa.create_engine(DB_DSN)
_catalog: Catalog | None = None


def catalog(cx) -> Catalog:
    global _catalog
    if _catalog is None:
        _catalog = load_catalog(cx)
    return _catalog


def pick_shape(question: str) -> str | None:
    options = "\n".join(f"- {name}: {s['description']}" for name, s in SHAPES.items())
    raw = llm.generate("Pick the kind of answer this basketball question needs. Reply with one JSON object "
                       f"{{\"kind\": \"<name>\"}} using one of these names:\n{options}\n\nQuestion: {question}\nJSON:")
    kind = (llm.parse_json_object(raw) or {}).get("kind")
    return kind if kind in SHAPES else None


def spec_for(kind: str, question: str) -> dict:
    """The shape's spec, with its stat field swapped for the stat the question asks about."""
    spec = {"fields": dict(SHAPES[kind]["spec"]["fields"]), "evidence": list(SHAPES[kind]["spec"]["evidence"])}
    stat = detect_stat(question, {})
    for name in list(spec["fields"]):
        if name in STATS and name != stat:
            spec["fields"] = {(stat if k == name else k): v for k, v in spec["fields"].items()}
    return spec


def describe(cx, item: dict, cat: Catalog) -> str:
    """One readable line for an evidence row."""
    t = item["table"]
    if t == "games":
        r = cx.execute(text("SELECT game_date, home_team_id, away_team_id, home_points, away_points FROM games "
                            "WHERE game_id = :g"), {"g": item["game_id"]}).mappings().first()
        if r:
            home, away = cat.team_by_id[r["home_team_id"]]["name"], cat.team_by_id[r["away_team_id"]]["name"]
            return f"{r['game_date']}: {away} {r['away_points']} at {home} {r['home_points']}"
    elif t == "box_scores":
        r = cx.execute(text("SELECT points, rebounds, assists, minutes, starter FROM box_scores "
                            "WHERE game_id = :g AND player_id = :p"),
                       {"g": item["game_id"], "p": item["player_id"]}).mappings().first()
        if r:
            name = cat.player_by_id[item["player_id"]]["full_name"]
            return (f"{name}: {r['points']} pts, {r['rebounds']} reb, {r['assists']} ast in {r['minutes']:.0f} min "
                    f"({'starter' if r['starter'] else 'bench'})")
    elif t in ("recaps", "availability_notes"):
        key = "recap_id" if t == "recaps" else "note_id"
        r = cx.execute(text(f"SELECT text FROM {t} WHERE {key} = :i"), {"i": item[key]}).first()
        if r:
            return r[0]
    return ""


class AskRequest(BaseModel):
    question: str
    kind: str | None = None


@app.get("/api/health")
def health() -> dict:
    with eng.connect() as cx:
        docs = cx.execute(text("SELECT count(*) FROM documents WHERE embedding IS NOT NULL")).scalar()
    return {"ok": True, "llm": LLM_MODEL, "embed_model": EMBED_MODEL, "documents": docs,
            "shapes": {k: v["description"] for k, v in SHAPES.items()}}


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    kind = req.kind if req.kind in SHAPES else (route(req.question) or pick_shape(req.question))
    if kind is None:
        return {"kind": None, "answerable": False, "message": "I couldn't tell what kind of answer that needs.",
                "result": None, "evidence": [], "trace": {}}
    spec = spec_for(kind, req.question)
    with eng.connect() as cx:
        cat = catalog(cx)
        result, trace = answer_question(cx, req.question, spec, cat)
        evidence = [dict(e, text=describe(cx, e, cat)) for e in result["evidence"]]
    if result["answerable"]:
        message = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in result.items() if k not in ("answerable", "evidence"))
        if trace.get("ties"):
            message += f" (tied with {', '.join(str(t) for t in trace['ties'][1:])})"
    else:
        message = f"Can't answer from the data: {trace.get('abstain_reason', 'not stated')}."
    return {"kind": kind, "spec": spec, "answerable": result["answerable"], "message": message,
            "result": result, "evidence": evidence, "trace": trace}


# The static site doubles as the chat UI: served from here, it detects the API and goes live.
DOCS = Path(__file__).resolve().parent.parent / "docs"
if DOCS.exists():
    app.mount("/", StaticFiles(directory=DOCS, html=True), name="site")
