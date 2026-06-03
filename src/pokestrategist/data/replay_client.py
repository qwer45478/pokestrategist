"""Showdown replay downloader.

Public endpoints (no auth required):
* Search:   ``GET https://replay.pokemonshowdown.com/search.json?format={fmt}&page={n}``
              -> JSON list of ``{id, format, players, rating, uploadtime, ...}``.
* Replay:   ``GET https://replay.pokemonshowdown.com/{id}.json``
              -> JSON ``{id, format, players, rating?, uploadtime, log, ...}`` where
                 ``log`` is the newline-separated battle protocol stream.

Behavior
--------
* Disk cache: every replay is stored as ``{cache_dir}/{id}.json`` and reused on hit.
* Polite: configurable inter-request delay; a single ``requests.Session`` is reused so
  TCP/TLS state is kept warm.
* Best-effort retries on transient HTTP errors (5xx and timeouts).

This module intentionally does **not** parse the protocol log — that is
``pokepilot.data.protocol``'s job. The downloader only deals with the JSON envelope.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import requests

logger = logging.getLogger(__name__)

REPLAY_BASE = "https://replay.pokemonshowdown.com"
DEFAULT_USER_AGENT = "PokePilot/0.0.1 (research; +https://github.com/)"


@dataclass(frozen=True)
class ReplaySearchHit:
    """A single entry returned by the replay search endpoint."""

    replay_id: str
    format: str
    players: tuple[str, ...]
    rating: Optional[int]
    upload_time: Optional[int]


class ReplayClient:
    """Downloads (and caches) Showdown replays.

    Parameters
    ----------
    cache_dir:
        Directory used to persist raw replay JSON. Created if missing.
    request_delay:
        Seconds to sleep between consecutive HTTP calls. Be a good citizen of the
        Showdown infrastructure — replays are free, so do not hammer.
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Number of retries on transient errors (timeouts and HTTP 5xx).
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/raw/replays",
        request_delay: float = 0.5,
        timeout: float = 15.0,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session or requests.Session()
        self._session.headers.setdefault("User-Agent", DEFAULT_USER_AGENT)
        self._last_request_at = 0.0

    # ------------------------------------------------------------------ HTTP
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_at = time.monotonic()

    def _get_json(self, url: str, params: Optional[dict] = None) -> object:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                logger.warning("Transient error (%s) on %s, attempt %d", exc, url, attempt)
                continue
            if resp.status_code >= 500:
                last_exc = RuntimeError(f"HTTP {resp.status_code} from {url}")
                logger.warning("Server error %d on %s, attempt %d", resp.status_code, url, attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        # Out of retries.
        raise RuntimeError(f"Failed to GET {url} after {self.max_retries} attempts") from last_exc

    # --------------------------------------------------------------- search
    def search(
        self,
        battle_format: str = "gen9ou",
        max_pages: int = 1,
        min_rating: Optional[int] = None,
    ) -> Iterator[ReplaySearchHit]:
        """Iterate replay search results, paginating until exhausted or `max_pages` hit.

        ``min_rating`` filters client-side because the search endpoint does not.
        Passing ``max_pages <= 0`` switches to exhaustive mode and keeps scanning
        until the search endpoint returns an empty page.
        Replays with no rating (private ladder / unranked) are dropped when filtering.
        """
        page = 1
        while True:
            if max_pages > 0 and page > max_pages:
                return
            payload = self._get_json(
                f"{REPLAY_BASE}/search.json",
                params={"format": battle_format, "page": page},
            )
            if not isinstance(payload, list) or not payload:
                return
            for entry in payload:
                rating = entry.get("rating")
                if min_rating is not None and (rating is None or rating < min_rating):
                    continue
                players = entry.get("players") or []
                yield ReplaySearchHit(
                    replay_id=str(entry["id"]),
                    format=str(entry.get("format", battle_format)),
                    players=tuple(str(p) for p in players),
                    rating=int(rating) if rating is not None else None,
                    upload_time=int(entry["uploadtime"]) if entry.get("uploadtime") else None,
                )
            page += 1

    # ----------------------------------------------------------------- fetch
    def _cache_path(self, replay_id: str) -> Path:
        # Showdown ids are already filesystem-safe (alnum + dashes).
        return self.cache_dir / f"{replay_id}.json"

    def fetch(self, replay_id: str, force: bool = False) -> dict:
        """Return the full replay JSON for `replay_id`, using the disk cache by default."""
        cache_path = self._cache_path(replay_id)
        if cache_path.exists() and not force:
            with cache_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        payload = self._get_json(f"{REPLAY_BASE}/{replay_id}.json")
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected replay payload type for {replay_id}: {type(payload)}")
        # Atomic write — never leave a half-written file behind on Ctrl-C.
        tmp_path = cache_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        tmp_path.replace(cache_path)
        return payload

    def fetch_many(
        self,
        replay_ids: Iterable[str],
        force: bool = False,
    ) -> Iterator[dict]:
        for rid in replay_ids:
            try:
                yield self.fetch(rid, force=force)
            except Exception:  # noqa: BLE001 — keep the batch flowing
                logger.exception("Failed to fetch replay %s; skipping", rid)
