"""Settings, all overridable with environment variables."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DB_DSN = os.getenv("DB_DSN", "postgresql+psycopg://nba:nba@localhost:5433/nba")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:2b")

KAGGLE_DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"
# Point this at an already-downloaded copy of the dataset to skip kagglehub.
NBA_DATA_DIR = os.getenv("NBA_DATA_DIR")
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", ROOT / "data" / "processed"))

# The season this index covers. Raw rows outside these dates are dropped at build time.
SEASON_LABEL = "2025-26"
SEASON_FIRST_DAY = date(2025, 10, 1)
SEASON_LAST_DAY = date(2026, 6, 30)
