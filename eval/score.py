"""Score pipeline answers against gold and write a scorecard.

An answer is correct when its answerable flag matches gold and, if answerable, every
field matches (case- and accent-insensitive for text; tied alternatives accepted).
Abstentions are broken out, because a confident wrong number is worse than "I don't know":

  false_abstain  - gold is answerable but the pipeline abstained
  hallucination  - gold is unanswerable but the pipeline answered
  wrong          - both answered, some field differs

usage:
    python -m eval.run dev        # answers + scores in one step (see eval/run.py)
    python -m eval.score eval/sets/dev.json eval/results/dev_answers.json
"""
import json
import sys
from collections import defaultdict

from nba_rag.catalog import fold


def _key(e: dict) -> tuple:
    return tuple(sorted(e.items()))


def _same(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return fold(str(a)).strip() == fold(str(b)).strip()
    return a == b


def grade(item: dict, result: dict) -> dict:
    gold = item["gold"]
    fields = item["answer"]["fields"]
    out = {"id": item["id"], "tier": item["tier"], "kind": item["kind"]}
    if not gold["answerable"]:
        out["outcome"] = "correct" if not result["answerable"] else "hallucination"
        return out
    if not result["answerable"]:
        out["outcome"] = "false_abstain"
        return out
    options = [gold] + [{**gold, **alt} for alt in gold.get("accept", [])]
    matched = next((o for o in options if all(_same(result.get(k), o[k]) for k in fields)), None)
    out["outcome"] = "correct" if matched else "wrong"
    if not matched:
        out["diff"] = {k: {"got": result.get(k), "gold": gold[k]} for k in fields if not _same(result.get(k), gold[k])}
    target = matched or gold
    got = {_key(e) for e in result["evidence"]}
    want = {_key(e) for e in target["evidence"]}
    out["evidence_precision"] = len(got & want) / len(got) if got else 0.0
    out["evidence_recall"] = len(got & want) / len(want) if want else 1.0
    return out


def summarize(grades: list[dict]) -> dict:
    def block(rows):
        n = len(rows)
        counts = defaultdict(int)
        for r in rows:
            counts[r["outcome"]] += 1
        ev = [r for r in rows if "evidence_precision" in r]
        return {
            "n": n,
            "accuracy": round(counts["correct"] / n, 3) if n else None,
            **{k: counts[k] for k in ("correct", "wrong", "false_abstain", "hallucination")},
            "evidence_precision": round(sum(r["evidence_precision"] for r in ev) / len(ev), 3) if ev else None,
            "evidence_recall": round(sum(r["evidence_recall"] for r in ev) / len(ev), 3) if ev else None,
        }
    by = lambda key: {k: block([g for g in grades if g[key] == k])  # noqa: E731
                      for k in sorted({g[key] for g in grades})}
    return {"overall": block(grades), "by_tier": by("tier"), "by_kind": by("kind")}


def score(questions: list[dict], answers: list[dict]) -> tuple[list[dict], dict]:
    by_id = {a["id"]: a for a in answers}
    grades = [grade(q, by_id[q["id"]]["result"]) for q in questions if q["id"] in by_id]
    return grades, summarize(grades)


def markdown(summary: dict, title: str) -> str:
    lines = [f"### {title}", "", "| Slice | n | Accuracy | Wrong | False abstain | Hallucination | Evidence P / R |",
             "|---|---|---|---|---|---|---|"]

    def row(name, b):
        ev = f"{b['evidence_precision']:.2f} / {b['evidence_recall']:.2f}" if b["evidence_precision"] is not None else "-"
        lines.append(f"| {name} | {b['n']} | {b['accuracy']:.0%} | {b['wrong']} | {b['false_abstain']} | "
                     f"{b['hallucination']} | {ev} |")
    row("**Overall**", summary["overall"])
    for tier in ("core", "extended", "stretch"):
        if tier in summary["by_tier"]:
            row(f"tier: {tier}", summary["by_tier"][tier])
    return "\n".join(lines)


if __name__ == "__main__":
    qs = json.load(open(sys.argv[1], encoding="utf-8"))
    ans = json.load(open(sys.argv[2], encoding="utf-8"))
    _, summary = score(qs, ans)
    print(markdown(summary, sys.argv[1]))
