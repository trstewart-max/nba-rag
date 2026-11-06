"""Bundle eval sets, answers and scores into docs/data/results.json for the static site.

usage:
    python -m eval.export_site [--repo-url https://github.com/<you>/nba-rag]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from eval.score import grade, summarize
from nba_rag.config import EMBED_MODEL, LLM_MODEL, PROCESSED_DIR, ROOT

HERE = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "data" / "results.json"
TRACE_KEYS = ("abstain_reason", "context", "llm_calls", "seconds", "plan", "retrieval", "ties", "answer_row")


class Describer:
    """One readable line per evidence row, from the processed CSVs."""

    def __init__(self):
        read = lambda n: pd.read_csv(PROCESSED_DIR / f"{n}.csv")  # noqa: E731
        self.teams = read("teams").set_index("team_id").name.to_dict()
        p = read("players")
        self.players = dict(zip(p.player_id, p.first_name + " " + p.last_name))
        self.games = read("games").set_index("game_id")
        self.box = read("box_scores").set_index(["game_id", "player_id"])
        self.recaps = read("recaps").set_index("recap_id").text.to_dict()
        self.notes = read("availability_notes").set_index("note_id").text.to_dict()
        self.counts = {"games": len(self.games), "box_scores": len(self.box), "players": len(p),
                       "recaps": len(self.recaps), "notes": len(self.notes)}

    def __call__(self, e: dict) -> str:
        t = e["table"]
        if t == "games" and e["game_id"] in self.games.index:
            g = self.games.loc[e["game_id"]]
            return (f"{g.game_date}: {self.teams[g.away_team_id]} {g.away_points} at "
                    f"{self.teams[g.home_team_id]} {g.home_points}")
        if t == "box_scores" and (e["game_id"], e["player_id"]) in self.box.index:
            b = self.box.loc[(e["game_id"], e["player_id"])]
            return (f"{self.players[e['player_id']]}: {b.points} pts, {b.rebounds} reb, {b.assists} ast "
                    f"in {b.minutes:.0f} min ({'starter' if b.starter else 'bench'})")
        if t == "recaps":
            return self.recaps.get(e["recap_id"], "")
        if t == "availability_notes":
            return self.notes.get(e["note_id"], "")
        return ""


def main(repo_url: str | None) -> None:
    describe = Describer()
    questions, scorecards = [], {}
    for split in ("dev", "heldout", "test"):
        answers_path = HERE / "results" / f"{split}_answers.json"
        if not answers_path.exists():
            continue
        qs = json.loads((HERE / "sets" / f"{split}.json").read_text(encoding="utf-8"))
        answers = {a["id"]: a for a in json.loads(answers_path.read_text(encoding="utf-8"))}
        grades = []
        for q in qs:
            a = answers.get(q["id"])
            if a is None:
                continue
            g = grade(q, a["result"])
            grades.append(g)
            questions.append({
                "split": split, "id": q["id"], "tier": q["tier"], "kind": q["kind"], "question": q["question"],
                "fields": q["answer"]["fields"], "outcome": g["outcome"],
                "answer": {k: v for k, v in a["result"].items() if k != "evidence"},
                "gold": {k: v for k, v in q["gold"].items() if k not in ("evidence", "accept")},
                "evidence": [dict(e, text=describe(e)) for e in a["result"]["evidence"]],
                "trace": {k: a["trace"][k] for k in TRACE_KEYS if k in a["trace"]},
            })
        scorecards[split] = summarize(grades)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": date.today().isoformat(), "repo_url": repo_url,
        "models": {"llm": LLM_MODEL, "embed": EMBED_MODEL},
        "dataset": describe.counts, "scorecards": scorecards, "questions": questions,
    }, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote {len(questions)} questions to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url")
    main(parser.parse_args().repo_url)
