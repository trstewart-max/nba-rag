import pandas as pd

from eval.score import grade, summarize
from nba_rag.build.flow import format_clock, game_flow
from nba_rag.build.text import parse_comment, possessive, surname


def test_game_flow_finds_comeback_and_go_ahead():
    games = pd.DataFrame([{"game_id": 1, "home_team_id": 1, "away_team_id": 2, "home_points": 9,
                           "away_points": 8, "winner_team_id": 1}])
    # Away team runs up 8-0; home comes back, ties at 8, then goes ahead for good on a free throw.
    plays = [(1, "PT11M00.00S", "3pt", 0, 3, 20), (1, "PT10M00.00S", "3pt", 0, 6, 20),
             (2, "PT05M00.00S", "2pt", 0, 8, 21), (3, "PT05M00.00S", "3pt", 3, 8, 10),
             (4, "PT01M00.00S", "3pt", 6, 8, 10), (4, "PT00M30.00S", "2pt", 8, 8, 11),
             (4, "PT00M04.20S", "freethrow", 9, 8, 12)]
    pbp = pd.DataFrame([{"gameId": 1, "period": p, "clock": c, "orderNumber": i, "actionType": a, "subType": "",
                         "personId": str(pid), "teamId": "", "scoreHome": h, "scoreAway": aw}
                        for i, (p, c, a, h, aw, pid) in enumerate(plays)])
    flow = game_flow(pbp, games).iloc[0]
    assert flow.winner_max_deficit == 8 and flow.winner_max_lead == 1
    assert flow.go_ahead_player_id == 12 and flow.go_ahead_period == 4 and flow.go_ahead_clock == "0:04.2"
    assert flow.halftime_home == 0 and flow.halftime_away == 8
    assert flow.lead_changes == 1 and flow.times_tied == 1


def test_game_flow_skips_feeds_that_disagree_with_the_final():
    games = pd.DataFrame([{"game_id": 1, "home_team_id": 1, "away_team_id": 2, "home_points": 100,
                           "away_points": 90, "winner_team_id": 1}])
    pbp = pd.DataFrame([{"gameId": 1, "period": 1, "clock": "PT11M00.00S", "orderNumber": 0, "actionType": "2pt",
                         "subType": "", "personId": "5", "teamId": "", "scoreHome": 2, "scoreAway": 0}])
    assert game_flow(pbp, games).empty


def test_text_helpers():
    assert format_clock("PT01M05.20S") == "1:05"
    assert parse_comment("DND - Injury/Illness") == ("did not dress", "injury/illness")
    assert parse_comment("DNP - Coach's Decision") == ("did not play", "coach's decision")
    assert parse_comment("NWT_TEAM_SUSPENSION") == ("not with team", "team suspension")
    assert parse_comment("Inactive") is None
    assert surname("Butler III") == "Butler" and surname("Jackson Jr.") == "Jackson"
    assert possessive("Lakers") == "Lakers'" and possessive("Thunder") == "Thunder's"


def _item(gold, fields, kind="k", tier="core"):
    return {"id": "x", "tier": tier, "kind": kind, "answer": {"fields": fields}, "gold": gold}


def test_grading_outcomes():
    fields = {"opponent": "str", "score": "str"}
    gold = {"answerable": True, "opponent": "Nuggets", "score": "120-99",
            "evidence": [{"table": "games", "game_id": 1}],
            "accept": [{"opponent": "Jazz", "score": "110-89", "evidence": [{"table": "games", "game_id": 2}]}]}
    ok = grade(_item(gold, fields), {"answerable": True, "opponent": "jazz", "score": "110-89",
                                     "evidence": [{"table": "games", "game_id": 2}]})
    assert ok["outcome"] == "correct" and ok["evidence_recall"] == 1.0
    wrong = grade(_item(gold, fields), {"answerable": True, "opponent": "Jazz", "score": "120-99", "evidence": []})
    assert wrong["outcome"] == "wrong"
    assert grade(_item(gold, fields), {"answerable": False, "evidence": []})["outcome"] == "false_abstain"
    none = {"answerable": False, "evidence": []}
    assert grade(_item(none, fields), {"answerable": True, "evidence": []})["outcome"] == "hallucination"
    assert summarize([ok, wrong])["overall"]["accuracy"] == 0.5
