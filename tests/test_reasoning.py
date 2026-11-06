import json
from datetime import date

from nba_rag import planner
from nba_rag.answer import abstain_result, coerce, resolve_player, value_supported
from nba_rag.catalog import link_entities
from nba_rag.reasoning import ContextRow, aggregate_players, derive_games, filter_games, unexplained_numbers


def _game(gid, day, home, away, hp, ap, season_type="regular"):
    return {"game_id": gid, "game_date": day, "season_type": season_type, "label": None, "home_team_id": home,
            "away_team_id": away, "home_points": hp, "away_points": ap, "winner_team_id": home if hp > ap else away}


GAMES = [
    _game(1, date(2025, 10, 21), 1, 2, 110, 100),
    _game(2, date(2025, 10, 23), 2, 1, 120, 99),
    _game(3, date(2025, 10, 25), 1, 6, 101, 104),
    _game(4, date(2026, 4, 20), 1, 2, 115, 90, "playoffs"),
]


def _box(gid, pid, tid, pts, starter=True, reb=5):
    return {"game_id": gid, "player_id": pid, "team_id": tid, "starter": starter, "points": pts, "rebounds": reb,
            "assists": 0, "steals": 0, "blocks": 0, "threes": 0}


def test_derive_games_numbers_each_phase_separately():
    games = derive_games(GAMES, focus_team_id=1)
    assert [g["game_number"] for g in games] == [1, 2, 3, 1]
    assert games[1]["focus_result"] == "loss" and games[1]["opponent_id"] == 2 and games[1]["margin"] == 21
    assert games[1]["score"] == "120-99" and games[0]["focus_venue"] == "home"


def test_filters():
    games = derive_games(GAMES, focus_team_id=1)
    plan = planner.Plan(focus_team_id=1, result="loss", season_phase="regular")
    assert [g["game_id"] for g in filter_games(games, plan, None, [])] == [2, 3]
    plan.min_margin = 10
    assert [g["game_id"] for g in filter_games(games, plan, None, [])] == [2]
    plan = planner.Plan(focus_team_id=1, game_pick="first", season_phase="regular")
    assert [g["game_id"] for g in filter_games(games, plan, None, [])] == [1]
    plan = planner.Plan(focus_team_id=1, season_phase="playoffs")
    assert [g["game_id"] for g in filter_games(games, plan, None, [])] == [4]
    plan = planner.Plan(focus_team_id=1, venue="away")
    assert [g["game_id"] for g in filter_games(games, plan, None, [])] == [2]


def test_aggregate_totals_thresholds_and_single_game_best():
    games = derive_games(GAMES[:3], focus_team_id=1)
    box = [_box(1, 10, 1, 30), _box(2, 10, 1, 20), _box(3, 10, 1, 31), _box(1, 11, 1, 25, reb=15),
           _box(2, 11, 1, 35, starter=False), _box(1, 99, 2, 50)]
    plan = planner.Plan(focus_team_id=1, players_from="focus")
    aggs = aggregate_players(box, games, plan, [])
    assert [(a["player_id"], a["total"]) for a in aggs] == [(10, 81), (11, 60)]
    plan.starters_only = True
    assert [(a["player_id"], a["total"]) for a in aggregate_players(box, games, plan, [])] == [(10, 81), (11, 25)]
    plan = planner.Plan(focus_team_id=1, players_from="focus", min_value=30)
    aggs = aggregate_players(box, games, plan, [])
    assert [(a["player_id"], len(a["hit_rows"])) for a in aggs] == [(10, 2), (11, 1)]
    plan = planner.Plan(focus_team_id=1, players_from="focus", per_game=True, stat="rebounds")
    assert aggregate_players(box, games, plan, [])[0]["player_id"] == 11
    plan = planner.Plan(focus_team_id=1, players_from="opponents")
    assert [a["player_id"] for a in aggregate_players(box, games, plan, [])] == [99]


def test_plan_validation_rejects_uncued_filters(catalog):
    q = "What was the final score of the Thunder's game against the Lakers on April 12, 2026?"
    e = link_entities(q, catalog)
    reply = json.dumps({"focus_team": "Lakers", "result": "loss", "game_pick": "last", "min_margin": 7,
                        "min_value": None, "rank": "least"})
    plan = planner.validate_plan(reply, q, e)
    # "final score" is not a "last game", 7 isn't in the question, nothing says "least".
    assert plan.game_pick == "all" and plan.min_margin is None and plan.rank == "most"
    assert plan.focus_team_id == 2 and plan.result == "any"


def test_plan_fallbacks_without_the_model(catalog):
    q = "Which Thunder starter scored the most total points in games the Thunder lost by 10 or more points?"
    plan = planner.validate_plan("not json", q, link_entities(q, catalog))
    assert plan.starters_only and plan.min_margin == 10 and plan.min_value is None
    q = "Which player had the most games with 30 or more points against the Thunder?"
    e = link_entities(q, catalog)
    plan = planner.validate_plan(None, q, e)
    assert plan.min_value == 30 and plan.players_from == "opponents"
    assert unexplained_numbers(q, e, plan) == []


def test_unexplained_number_triggers_abstention(catalog):
    q = "How many points did Chet Holmgren score in games with 12 blocks?"
    e = link_entities(q, catalog)
    plan = planner.validate_plan(None, q, e)
    assert unexplained_numbers(q, e, plan) == [12]


def test_value_supported_and_coerce():
    row = ContextRow(label="N1", line="Chet Holmgren was ruled out for March 2, 2026 because of injury/illness.",
                     values={"status": "did not dress"}, evidence=[])
    assert value_supported("status", "Did Not Dress", row)
    assert not value_supported("status", "out", row)
    assert value_supported("reason", "injury/illness", row)
    assert not value_supported("reason", "sprained ankle", row)
    assert value_supported("deficit", 2, row) and not value_supported("deficit", 17, row)
    assert coerce("true", "bool") is True and coerce(3.6, "int") == 4
    assert abstain_result({"fields": {"points": "int", "player_name": "str"}}) == {
        "answerable": False, "points": 0, "player_name": "", "evidence": []}


def test_resolve_player_by_surname(catalog):
    box = [_box(1, 11, 1, 10), _box(1, 15, 1, 12), _box(1, 16, 1, 3)]
    assert resolve_player("Holmgren", catalog, box, [1]) == 11
    assert resolve_player("Chet Holmgren", catalog, box, [1]) == 11
    assert resolve_player("Williams", catalog, box, [1]) is None  # two Williamses played
    assert resolve_player("Holmgren", catalog, box, [2]) is None  # not in that game


def test_plan_reads_unambiguous_result_words(catalog):
    q = "What was the Thunder's largest margin of victory this season, and who did they beat?"
    assert planner.validate_plan(None, q, link_entities(q, catalog)).result == "win"
    q = "Who won the Thunder's game against the Lakers?"
    assert planner.validate_plan(None, q, link_entities(q, catalog)).result == "any"


def test_quote_grounding_and_player_naming(catalog):
    from nba_rag.answer import ground_quote, names_player
    row = ContextRow(label="R1", line="recap of game 1: The Thunder won 101-99. Holmgren's 3-pointer with 0:40 left "
                     "put the Thunder ahead for good. Shai Gilgeous-Alexander had 35 points.", values={}, evidence=[])
    sentence = ground_quote("Holmgren's 3-pointer with 0:40 left put the Thunder ahead", row)
    assert sentence.startswith("Holmgren's 3-pointer")
    assert ground_quote("Williams hit a buzzer beater from half court", row) is None
    assert names_player(sentence, catalog.player_by_id[11])
    assert not names_player(sentence, catalog.player_by_id[10])
