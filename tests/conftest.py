from datetime import date

import pytest

from nba_rag.catalog import Catalog, strip_suffix


def _team(tid, city, name, abbr):
    from nba_rag.catalog import CITY_ALIASES, TEAM_ALIASES
    return {"team_id": tid, "city": city, "name": name, "abbreviation": abbr,
            "aliases": TEAM_ALIASES.get(name, []), "city_aliases": CITY_ALIASES.get(city, [])}


def _player(pid, first, last):
    return {"player_id": pid, "first_name": first, "last_name": last, "full_name": f"{first} {last}",
            "surname": strip_suffix(last)}


@pytest.fixture
def catalog() -> Catalog:
    teams = [
        _team(1, "Oklahoma City", "Thunder", "OKC"),
        _team(2, "Los Angeles", "Lakers", "LAL"),
        _team(3, "LA", "Clippers", "LAC"),
        _team(4, "Philadelphia", "76ers", "PHI"),
        _team(5, "Portland", "Trail Blazers", "POR"),
        _team(6, "Denver", "Nuggets", "DEN"),
    ]
    players = [
        _player(10, "Shai", "Gilgeous-Alexander"),
        _player(11, "Chet", "Holmgren"),
        _player(12, "Nikola", "Jokić"),
        _player(13, "Jaren", "Jackson Jr."),
        _player(14, "Stephen", "Curry"),
        _player(15, "Jalen", "Williams"),
        _player(16, "Jaylin", "Williams"),
        _player(17, "De'Aaron", "Fox"),
        _player(18, "LeBron", "James"),
    ]
    return Catalog(teams=teams, players=players, season_start=date(2025, 10, 21),
                   regular_season_end=date(2026, 4, 12), season_end=date(2026, 6, 13),
                   all_star_break=(date(2026, 2, 13), date(2026, 2, 18)))
