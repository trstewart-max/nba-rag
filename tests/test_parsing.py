from datetime import date

from nba_rag import parsing
from nba_rag.catalog import link_entities


def test_date_forms(catalog):
    w = parsing.parse_date_window
    assert w("on April 12, 2026?", catalog) == (date(2026, 4, 12), date(2026, 4, 12))
    assert w("on Nov. 3rd", catalog) == (date(2025, 11, 3), date(2025, 11, 3))
    assert w("on 2026-01-05", catalog) == (date(2026, 1, 5), date(2026, 1, 5))
    assert w("in January 2026", catalog) == (date(2026, 1, 1), date(2026, 1, 31))
    assert w("in December", catalog) == (date(2025, 12, 1), date(2025, 12, 31))
    assert w("after the All-Star break", catalog) == (date(2026, 2, 19), date(2026, 4, 12))
    assert w("before the break", catalog) == (date(2025, 10, 21), date(2026, 2, 12))
    assert w("Who may have scored the most?", catalog) is None


def test_ordinals_scores_numbers():
    assert parsing.parse_game_ordinal("in their 3rd game") == 3
    assert parsing.parse_game_ordinal("the tenth game of the year") == 10
    assert parsing.parse_game_ordinal("their first game") is None  # the opener is a plan filter
    assert parsing.parse_scores("the 125-124 win") == [(125, 124)]
    assert parsing.standalone_numbers("125-124 win on March 3, 2026 with 30 points and 4 3-pointers") == [30, 4]
    assert parsing.standalone_numbers("in the 2025-26 season") == []


def test_stat_phase_venue():
    assert parsing.detect_stat("Who had the most boards?", {}) == "rebounds"
    assert parsing.detect_stat("How many did he score?", {"assists": "int"}) == "assists"
    assert parsing.detect_stat("Who made the most 3-pointers?", {}) == "threes"
    assert parsing.detect_season_phase("in the West Conf. Finals") == "playoffs"
    assert parsing.detect_season_phase("in the play-in game") == "play-in"
    assert parsing.detect_season_phase("this season") is None
    assert parsing.detect_venue("their record at home") == "home"
    assert parsing.detect_venue("in road games") == "away"


def test_teams_link_and_opponent_side(catalog):
    e = link_entities("How did the Thunder do against the Lakers?", catalog)
    assert [t["name"] for t in e.teams] == ["Thunder", "Lakers"]
    assert e.opponent_team_ids == {2}
    e = link_entities("Did the Sixers beat the Blazers?", catalog)
    assert [t["name"] for t in e.teams] == ["76ers", "Trail Blazers"]
    # "Los Angeles" alone could be either team, so it names neither.
    assert link_entities("Who won in Los Angeles?", catalog).teams == []
    assert [t["name"] for t in link_entities("the Los Angeles Clippers", catalog).teams] == ["Clippers"]
    assert [t["name"] for t in link_entities("OKC vs DEN", catalog).teams] == ["Thunder", "Nuggets"]


def test_players_link_with_accents_suffixes_and_nicknames(catalog):
    e = link_entities("How many points did Nikola Jokic score?", catalog)
    assert [p["player_id"] for p in e.players] == [12]
    e = link_entities("Did Jaren Jackson play?", catalog)
    assert [p["player_id"] for p in e.players] == [13]
    e = link_entities("What about Steph Curry?", catalog)
    assert [p["player_id"] for p in e.players] == [14] and not e.unresolved_names
    e = link_entities("How many assists did De'Aaron Fox's backup have?", catalog)
    assert [p["player_id"] for p in e.players] == [17]


def test_unknown_names_and_surnames(catalog):
    e = link_entities("How many points did Marcus Holmgren score?", catalog)
    assert e.unresolved_names == ["Marcus Holmgren"]
    e = link_entities("Who hit the shot after Holmgren fouled out?", catalog)
    assert e.surnames == ["holmgren"] and not e.players
    # A surname two players share is kept as a surname, to be resolved within the games.
    e = link_entities("Did Williams start for the Thunder?", catalog)
    assert e.surnames == ["williams"]


def test_team_names_with_digits_are_not_numbers(catalog):
    from nba_rag.parsing import mask_spans, standalone_numbers
    q = "Who led the 76ers in scoring?"
    e = link_entities(q, catalog)
    assert standalone_numbers(mask_spans(q, e.spans)) == []


def test_possessives_and_opponent_phrasing(catalog):
    e = link_entities("What was LeBron James' point total in the opener?", catalog)
    assert [p["player_id"] for p in e.players] == [18] and not e.unresolved_names
    e = link_entities("What was the season-high for rebounds by an opponent of the Thunder?", catalog)
    assert e.opponent_team_ids == {1}


def test_numbers_in_recap_sentences():
    sn = parsing.standalone_numbers
    assert sn("At the break it was Raptors 51, Heat 44.") == []
    assert sn("The Raptors overcame a 3-point hole.") == [3]
    assert sn("The Raptors' biggest lead was 23.") == [23]
    assert sn("He made 4 3-pointers and shot 2.5 per game") == [4]


def test_holiday_dates(catalog):
    w = parsing.parse_date_window
    assert w("against the Lakers on Christmas?", catalog) == (date(2025, 12, 25), date(2025, 12, 25))
    assert w("on New Year's Eve", catalog) == (date(2025, 12, 31), date(2025, 12, 31))
    assert w("on New Year's Day", catalog) == (date(2026, 1, 1), date(2026, 1, 1))
