"""Answer an eval set, score it, and save answers, grades and the scorecard.

usage:
    python -m eval.run dev
    python -m eval.run heldout
"""
import json
import logging
import sys
from pathlib import Path

from eval.score import markdown, score
from nba_rag.ask import run_file

HERE = Path(__file__).resolve().parent


def main(split: str) -> None:
    questions_path = HERE / "sets" / f"{split}.json"
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    answers_path = results / f"{split}_answers.json"
    run_file(str(questions_path), str(answers_path))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    grades, summary = score(questions, answers)
    (results / f"{split}_scores.json").write_text(json.dumps({"summary": summary, "grades": grades}, indent=1),
                                                  encoding="utf-8")
    print(markdown(summary, f"{split} set"))
    for g in grades:
        if g["outcome"] != "correct":
            q = next(q for q in questions if q["id"] == g["id"])
            a = next(a for a in answers if a["id"] == g["id"])
            print(f"- {g['id']} [{g['kind']}] {g['outcome']}: {q['question']}\n    "
                  f"{g.get('diff') or a['trace'].get('abstain_reason')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
