"""Answer a file of questions, or one question from the command line.

usage:
    python -m nba_rag.ask eval/sets/dev.json --out eval/results/dev_answers.json
    python -m nba_rag.ask --question "Who led the Knicks in scoring on March 3, 2026?" --kind leader

A question file is a list of {"id", "question", "answer": {"fields", "evidence"}}. Any
other keys (gold answers, tiers) are ignored here, so the pipeline never sees them.
"""
import argparse
import json
import logging
import time

import sqlalchemy as sa

from nba_rag.answer import answer_question
from nba_rag.catalog import load_catalog
from nba_rag.config import DB_DSN
from nba_rag.shapes import SHAPES

logger = logging.getLogger(__name__)


def run_file(path: str, out: str, limit: int | None = None) -> None:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)[:limit]
    eng = sa.create_engine(DB_DSN)
    outputs = []
    with eng.connect() as cx:
        catalog = load_catalog(cx)
        for q in questions:
            spec = {"fields": q["answer"]["fields"], "evidence": q["answer"]["evidence"]}
            t0 = time.perf_counter()
            result, trace = answer_question(cx, q["question"], spec, catalog)
            trace["seconds"] = round(time.perf_counter() - t0, 2)
            outputs.append({"id": q["id"], "result": result, "trace": trace})
            logger.info("%s: %s\n  -> %s", q["id"], q["question"],
                        json.dumps({k: v for k, v in result.items() if k != "evidence"}))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=1, default=str)
    logger.info("Wrote %d answers to %s", len(outputs), out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="question file")
    parser.add_argument("--out", default="answers.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question")
    parser.add_argument("--kind", choices=sorted(SHAPES), help="answer shape for --question")
    args = parser.parse_args()
    if args.question:
        eng = sa.create_engine(DB_DSN)
        with eng.connect() as cx:
            result, trace = answer_question(cx, args.question, SHAPES[args.kind]["spec"], load_catalog(cx))
        print(json.dumps({"result": result, "trace": trace}, indent=1, default=str, ensure_ascii=False))
    else:
        run_file(args.path, args.out, args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
