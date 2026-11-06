"""Deterministic reading of a question: dates, ordinals, scores, stats, season phase, venue.

Anything a regular expression can read reliably is read here, not by the LLM. The LLM is
only asked about wording that needs judgment (see planner.py).
"""
import calendar
import re
from datetime import date, timedelta

from nba_rag.catalog import Catalog

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
ORDINAL_WORDS = {w: i for i, w in enumerate(
    ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
     "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
     "eighteenth", "nineteenth", "twentieth"], start=1)}

# Box-score stats a question can ask about, and the words that name each one.
STAT_WORDS = {
    "points": r"\bpoints?\b|\bscor(?:e|ed|es|ing|er)\b|\bpts\b",
    "rebounds": r"\brebound(?:s|ed|ing)?\b|\bboards\b",
    "assists": r"\bassist(?:s|ed)?\b|\bdimes\b",
    "steals": r"\bsteals?\b",
    "blocks": r"\bblock(?:s|ed)?\b|\bblocked shots\b",
    "threes": r"\bthrees\b|\b3-pointers?\b|\bthree-pointers?\b|\b3s\b|\b3PM\b",
}
STATS = tuple(STAT_WORDS)

# Game days fans name by holiday. New Year's Eve is checked before New Year's Day.
HOLIDAYS = {
    r"\bchristmas(?: day)?\b": (12, 25),
    r"\bnew year'?s eve\b": (12, 31),
    r"\bnew year'?s(?: day)?\b(?! eve)": (1, 1),
}

SEASON_PHASES = [
    ("play-in", r"\bplay-?in\b"),
    ("cup-final", r"\b(?:nba )?cup (?:final|championship)\b"),
    ("playoffs", r"\b(?:playoffs?|postseason|post-season|first round|second round|conf(?:erence|\.)? "
                 r"(?:semi-?)?finals?|nba finals|series)\b"),
    ("regular", r"\bregular season\b"),
]


def detect_stat(question: str, fields: dict) -> str:
    """The stat a question is about: a stat-named answer field first, else the first stat word."""
    for name in fields:
        if name in STATS:
            return name
    hits = [(m.start(), stat) for stat, pat in STAT_WORDS.items()
            for m in [re.search(pat, question, re.IGNORECASE)] if m]
    return min(hits)[1] if hits else "points"


def detect_season_phase(question: str) -> str | None:
    """'playoffs', 'play-in', 'cup-final', 'regular', or None when the question doesn't say."""
    for phase, pat in SEASON_PHASES:
        if re.search(pat, question, re.IGNORECASE):
            return phase
    return None


def detect_venue(question: str) -> str | None:
    if re.search(r"\bat home\b|\bhome games?\b|\bhome court\b", question, re.IGNORECASE):
        return "home"
    if re.search(r"\bon the road\b|\baway games?\b|\broad games?\b", question, re.IGNORECASE):
        return "away"
    return None


def parse_game_ordinal(question: str) -> int | None:
    """The n in "the 3rd game", "their tenth game" or "game #12"."""
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th) game\b", question, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(rf"\b({'|'.join(ORDINAL_WORDS)}) game\b", question, re.IGNORECASE)
    if m and m.group(1).lower() != "first":  # "first game" is the opener, handled by the plan
        return ORDINAL_WORDS[m.group(1).lower()]
    m = re.search(r"\bgame (?:#|no\.?\s*|number\s+)(\d{1,2})\b", question, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _season_year(month: int, catalog: Catalog) -> int:
    """Seasons span two calendar years; October belongs to the first, April to the second."""
    for year in range(catalog.season_start.year, catalog.season_end.year + 1):
        last = calendar.monthrange(year, month)[1]
        if date(year, month, last) >= catalog.season_start and date(year, month, 1) <= catalog.season_end:
            return year
    return catalog.season_start.year


def parse_date_window(question: str, catalog: Catalog) -> tuple[date, date] | None:
    """All the dates a question mentions, as one inclusive (start, end) window, or None.

    Reads "April 12, 2026", "April 12", "Apr. 12th", "January 2026", "in January",
    "2026-04-12", and before/after the All-Star break.
    """
    spans, taken = [], []

    def free(m):
        return not any(m.start() < e and s < m.end() for s, e in taken)

    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", question):
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        spans.append((d, d)); taken.append(m.span())
    for m in re.finditer(rf"\b({MONTH_RE})\.? (\d{{1,2}})(?:st|nd|rd|th)?(?:,? (\d{{4}}))?\b", question, re.IGNORECASE):
        if not free(m):
            continue
        month = MONTHS[m.group(1).lower()]
        year = int(m.group(3)) if m.group(3) else _season_year(month, catalog)
        try:
            d = date(year, month, int(m.group(2)))
        except ValueError:
            continue
        spans.append((d, d)); taken.append(m.span())
    for m in re.finditer(rf"\b({MONTH_RE})\b\.?(?:,? (\d{{4}}))?", question, re.IGNORECASE):
        if not free(m):
            continue
        word = m.group(1)
        # "may", "march" and "mar" are ordinary words too; only count them capitalized.
        if word.lower() in ("may", "march", "mar") and not word[0].isupper():
            continue
        month = MONTHS[word.lower()]
        year = int(m.group(2)) if m.group(2) else _season_year(month, catalog)
        spans.append((date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])))
        taken.append(m.span())
    for pattern, (month, day) in HOLIDAYS.items():
        m = re.search(pattern, question, re.IGNORECASE)
        if m and free(m):
            d = date(_season_year(month, catalog), month, day)
            spans.append((d, d)); taken.append(m.span())
    brk_start, brk_end = catalog.all_star_break
    if re.search(r"\b(after|since|following|post)\b[^.?]*\b(all-star )?break\b", question, re.IGNORECASE):
        spans.append((brk_end + timedelta(days=1), catalog.regular_season_end))
    if re.search(r"\b(before|prior to|until|leading up to)\b[^.?]*\b(all-star )?break\b", question, re.IGNORECASE):
        spans.append((catalog.season_start, brk_start - timedelta(days=1)))
    if not spans:
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


def parse_scores(question: str) -> list[tuple[int, int]]:
    """Final scores written like "125-124", as (higher, lower)."""
    out = []
    for m in re.finditer(r"(?<![\d-])(\d{2,3})-(\d{2,3})(?![\d-])", question):
        a, b = int(m.group(1)), int(m.group(2))
        out.append((max(a, b), min(a, b)))
    return out


def standalone_numbers(text_: str) -> list[int]:
    """Numbers other than years, days in dates, season labels, 3-pointers and parts of scores."""
    text_ = re.sub(r"\b\d{4}-\d{2}-\d{2}\b|\b20\d\d-\d\d\b", " ", text_)
    text_ = re.sub(rf"\b({MONTH_RE})\.? \d{{1,2}}(?:st|nd|rd|th)?(?!\d)(?:,? \d{{4}})?", " ", text_, flags=re.IGNORECASE)
    text_ = re.sub(r"\b(?:3|three)-pointers?\b|\b3-point (?:shots?|attempts?|range|shooting)\b|\b3s\b|\b3PM\b",
                   " ", text_, flags=re.IGNORECASE)
    # Line scores written "Celtics 98, Heat 96" (at the half, or final).
    text_ = re.sub(r"\b\d{2,3}, (?:[A-Z][\w'.]*\s)+\d{2,3}\b", " ", text_)
    return [int(n) for n in re.findall(r"(?<![\d.-])(?<!\d-)(\d{1,3})(?!\d|\.\d)(?!-\d)", text_)]


def mask_spans(question: str, spans: list[tuple[int, int]]) -> str:
    """Blank out the given character spans (team and player names) before reading numbers."""
    chars = list(question)
    for s, e in spans:
        for i in range(s, min(e, len(chars))):
            chars[i] = " "
    return "".join(chars)
