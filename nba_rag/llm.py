"""Thin Ollama client: deterministic generation and batched embeddings."""
import json
import logging
import re
import time

import requests

from nba_rag.config import EMBED_MODEL, LLM_MODEL, OLLAMA_HOST

logger = logging.getLogger(__name__)

# Ollama sometimes rejects a request while it swaps models in and out of memory; the same
# request succeeds a moment later, so these are retried unchanged.
RETRY_STATUSES = {400, 429, 500, 502, 503, 504}
RETRY_DELAYS = (1, 3, 10)

# nomic-embed-text is asymmetric: indexed text and search text get different prefixes.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def _post(path: str, payload: dict) -> dict:
    for attempt, delay in enumerate((*RETRY_DELAYS, None), start=1):
        try:
            r = requests.post(f"{OLLAMA_HOST}{path}", json=payload, timeout=300)
        except requests.exceptions.ConnectionError as e:
            if delay is None:
                raise RuntimeError(f"Can't reach Ollama at {OLLAMA_HOST}. Is it running?") from e
            logger.warning("Ollama %s connection failed (attempt %d); retrying in %ss", path, attempt, delay)
        else:
            if r.status_code not in RETRY_STATUSES or delay is None:
                r.raise_for_status()
                return r.json()
            logger.warning("Ollama %s returned %s (attempt %d); retrying in %ss", path, r.status_code, attempt, delay)
        time.sleep(delay)
    raise AssertionError("unreachable")


def generate(prompt: str, model: str = LLM_MODEL) -> str:
    """One completion with temperature 0 and a fixed seed, so reruns reproduce answers."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # qwen3.5 is a thinking model; with thinking on, a 2B model can spend its whole
        # budget reasoning and return nothing. Grounded extraction doesn't need it.
        "think": False,
        "options": {"temperature": 0, "seed": 0},
    }
    return _post("/api/generate", payload)["response"]


def embed_batch(texts: list[str], model: str = EMBED_MODEL, batch_size: int = 64) -> list[list[float]]:
    """Embed texts in order, one request per batch. Callers add the nomic prefix."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(_post("/api/embed", {"model": model, "input": texts[i:i + batch_size]})["embeddings"])
    return out


def embed_query(text_: str, model: str = EMBED_MODEL) -> list[float]:
    return embed_batch([QUERY_PREFIX + text_], model)[0]


def parse_json_object(raw: str | None) -> dict | None:
    """The outermost {...} in a model reply, parsed; None if there isn't a valid one.

    Small models wrap JSON in prose or code fences even when told not to.
    """
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
