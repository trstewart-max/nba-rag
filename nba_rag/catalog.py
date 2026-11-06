"""Teams, players and season dates, plus matching the names a question uses to them."""
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.engine import Connection

# Common short names. Only the forms fans actually write; anything else must use the
# mascot, "City Mascot", or the abbreviation.
TEAM_ALIASES = {
    "76ers": ["Sixers"], "Trail Blazers": ["Blazers"], "Timberwolves": ["Wolves"],
    "Cavaliers": ["Cavs"], "Mavericks": ["Mavs"],
}
# Extra city spellings. "Los Angeles" alone is then ambiguous and never matches a team.
CITY_ALIASES = {"LA": ["Los Angeles"]}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
MONTH_WORDS = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
               "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
               "sep", "sept", "oct", "nov", "dec"}
# Capitalized words that start a question or clause, never part of a player's name.
FUNCTION_WORDS = {"did", "does", "do", "how", "who", "whom", "whose", "which", "what", "why", "when", "where",
                  "was", "were", "is", "has", "had", "have", "in", "on", "for", "against", "after", "before",
                  "during", "since", "and", "or", "but", "then", "the", "a", "an", "of", "at", "by", "vs",
                  "between", "among", "list", "name", "give", "show", "tell", "can", "could", "would", "all",
                  "star", "all-star", "nba", "game", "season", "playoffs", "finals", "conference", "west",
                  "east", "western", "eastern", "cup", "regular", "christmas", "day", "opener"}


def fold(s: str) -> str:
    """Lowercase, strip accents and periods, straighten apostrophes: 'A.J. Jokić' -> 'aj jokic'."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.replace("’", "'").replace(".", "").lower()


def strip_suffix(last_name: str) -> str:
    parts = last_name.split()
    while len(parts) > 1 and fold(parts[-1]) in NAME_SUFFIXES:
        parts.pop()
    return " ".join(parts)


@dataclass
class Catalog:
    """Everything loaded once per run: teams, players and the season's key dates."""
    teams: list[dict]
    players: list[dict]
    season_start: date          # first regular-season game
    regular_season_end: date    # last regular-season game
    season_end: date            # last game of any kind
    all_star_break: tuple[date, date]

    def __post_init__(self):
        self.team_by_id = {t["team_id"]: t for t in self.teams}
        self.player_by_id = {p["player_id"]: p for p in self.players}
        variants: dict[str, set[int]] = {}
        for p in self.players:
            for v in {fold(p["full_name"]), fold(f"{p['first_name']} {p['surname']}")}:
                variants.setdefault(v, set()).add(p["player_id"])
        # A spelling two players share can't identify either one.
        self.player_variants = {v: next(iter(ids)) for v, ids in variants.items() if len(ids) == 1}
        self.first_names = {fold(p["first_name"]) for p in self.players}
        self.surnames = {fold(p["surname"]) for p in self.players}
        self.team_words = {fold(w) for t in self.teams
                           for w in " ".join([t["city"], t["name"], *t["aliases"], *t["city_aliases"]]).split()}


def load_catalog(cx: Connection) -> Catalog:
    teams = [dict(r) for r in cx.execute(text(
        "SELECT team_id, city, name, abbreviation FROM teams ORDER BY team_id")).mappings()]
    for t in teams:
        t["aliases"] = TEAM_ALIASES.get(t["name"], [])
        t["city_aliases"] = CITY_ALIASES.get(t["city"], [])
    players = [dict(r) for r in cx.execute(text(
        "SELECT player_id, first_name, last_name FROM players ORDER BY player_id")).mappings()]
    for p in players:
        p["full_name"] = f"{p['first_name']} {p['last_name']}"
        p["surname"] = strip_suffix(p["last_name"])
    rows = cx.execute(text("SELECT game_date, season_type FROM games")).all()
    regular = sorted({d for d, t in rows if t == "regular"})
    return Catalog(teams=teams, players=players, season_start=regular[0], regular_season_end=regular[-1],
                   season_end=max(d for d, _ in rows), all_star_break=find_all_star_break(regular))


def find_all_star_break(game_days: list[date]) -> tuple[date, date]:
    """The longest stretch of February with no regular-season games: the All-Star break."""
    gaps = [(b - a, a, b) for a, b in zip(game_days, game_days[1:]) if a.month == 2 or b.month == 2]
    if not gaps:
        return game_days[0], game_days[0] - timedelta(days=1)
    _, last_before, first_after = max(gaps)
    return last_before + timedelta(days=1), first_after - timedelta(days=1)


@dataclass
class Entities:
    """What a question names: teams and players in order of mention, and names it can't place."""
    teams: list[dict] = field(default_factory=list)
    players: list[dict] = field(default_factory=list)
    unresolved_names: list[str] = field(default_factory=list)
    # Teams written as the opposition ("against the Thunder", "vs. OKC").
    opponent_team_ids: set[int] = field(default_factory=set)
    # Known surnames written alone ("Did Holmgren..."), resolved once the games are known.
    surnames: list[str] = field(default_factory=list)
    # Character spans of the question taken by names, so numbers inside them ("76ers") are ignored.
    spans: list[tuple[int, int]] = field(default_factory=list)


OPPONENT_CUE = r"(?:against|vs\.?|versus|facing|playing|over|beat|to|opponents? of)"
NAME_TOKEN = r"[A-Z][\w'’.-]*"


def _team_patterns(t: dict) -> list[tuple[str, int]]:
    """(regex, flags) for every way a team may be written, most specific first."""
    mascots = [t["name"], *t["aliases"]]
    cities = [t["city"], *t["city_aliases"]]
    pats = [(re.escape(f"{c} {m}"), re.IGNORECASE) for c in cities for m in mascots]
    pats += [(re.escape(m), re.IGNORECASE) for m in mascots]
    pats.append((re.escape(t["abbreviation"]), 0))
    return pats


def link_entities(question: str, catalog: Catalog) -> Entities:
    """Match the teams and players a question names against the catalog.

    Teams match on "City Mascot", the mascot or an alias (any case), the abbreviation
    (exact case), or the city alone when no other team shares it. Players match on their
    full name with accents, periods and suffixes optional. A capitalized pair that looks
    like a player's name but isn't one is reported as unresolved, so the caller abstains
    rather than answering about somebody else.
    """
    e = Entities()
    found_teams = []
    for t in catalog.teams:
        for pat, flags in _team_patterns(t):
            for m in re.finditer(rf"(?<![\w-]){pat}(?![\w-])", question, flags):
                found_teams.append((m.start(), t))
                e.spans.append(m.span())
                before = question[max(0, m.start() - 20):m.start()]
                if re.search(rf"\b{OPPONENT_CUE}\s+(?:the\s+)?$", before, re.IGNORECASE):
                    e.opponent_team_ids.add(t["team_id"])
    city_count: dict[str, int] = {}
    for t in catalog.teams:
        for c in {t["city"], *t["city_aliases"]}:
            city_count[c] = city_count.get(c, 0) + 1
    for t in catalog.teams:
        if city_count[t["city"]] > 1 or len(t["city"]) <= 2:
            continue
        for m in re.finditer(rf"(?<![\w-]){re.escape(t['city'])}(?![\w-])", question, re.IGNORECASE):
            if not any(s <= m.start() < end for s, end in e.spans):
                found_teams.append((m.start(), t))
                e.spans.append(m.span())
    seen = set()
    for _, t in sorted(found_teams, key=lambda x: (x[0], x[1]["team_id"])):
        if t["team_id"] not in seen:
            seen.add(t["team_id"])
            e.teams.append(t)

    # Drop possessives: "Fox's" -> "Fox", "Hawkins'" -> "Hawkins".
    folded = re.sub(r"'s(?![a-z])|(?<=s)'(?![a-z])", "", fold(question))
    # fold() drops periods and accents, so offsets in `folded` don't line up with `question`.
    # Names are matched in folded text; their spans are re-found in the original below.
    found_players = []
    for variant, pid in catalog.player_variants.items():
        m = re.search(rf"(?<![\w'-]){re.escape(variant)}(?![\w'-])", folded)
        if m:
            found_players.append((m.start(), m.end(), pid))
    # Prefer the longest match at a position ("jaren jackson jr" over "jaren jackson").
    found_players.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    taken: list[tuple[int, int]] = []
    for s, end, pid in found_players:
        if any(a <= s < b for a, b in taken):
            continue
        taken.append((s, end))
        if pid not in [p["player_id"] for p in e.players]:
            e.players.append(catalog.player_by_id[pid])

    tokens = list(re.finditer(rf"(?<![\w'’-]){NAME_TOKEN}", question))
    token_spans_in_names = set()
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if question[a.end():b.start()] != " " or i in token_spans_in_names:
            continue
        pair = fold(f"{a.group(0)} {re.sub(r'[’\']s?$', '', b.group(0))}")
        if pair in catalog.player_variants:
            token_spans_in_names |= {i, i + 1}
            e.spans += [a.span(), b.span()]
            continue
        first, last = fold(a.group(0)), fold(re.sub(r"[’']s?$", "", b.group(0)))
        if _not_a_name(first, catalog) or _not_a_name(last, catalog):
            continue
        if first in catalog.first_names or last in catalog.surnames:
            # "Steph Curry": a shortened first name that leaves one candidate is accepted.
            cands = [p for p in catalog.players if fold(p["surname"]) == last
                     and (fold(p["first_name"]).startswith(first) or first.startswith(fold(p["first_name"])))]
            if len(cands) == 1:
                if cands[0]["player_id"] not in [p["player_id"] for p in e.players]:
                    e.players.append(cands[0])
            else:
                e.unresolved_names.append(f"{a.group(0)} {b.group(0)}")
            token_spans_in_names |= {i, i + 1}
            e.spans += [a.span(), b.span()]

    known_full = {fold(p["full_name"]) for p in e.players}
    for i, tok in enumerate(tokens):
        if i in token_spans_in_names:
            continue
        word = fold(re.sub(r"[’']s?$", "", tok.group(0)))
        if word in catalog.surnames and not _not_a_name(word, catalog):
            if any(word in name.split() or name.endswith(word) for name in known_full):
                continue  # part of a full name matched above
            e.spans.append(tok.span())
            if word not in e.surnames:
                e.surnames.append(word)
    return e


def _not_a_name(word: str, catalog: Catalog) -> bool:
    return word in catalog.team_words or word in MONTH_WORDS or word in FUNCTION_WORDS or not word[:1].isalpha()
