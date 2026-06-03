"""Download a filtered subset of the metamon raw replay archive."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import duckdb
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


DATASET_PARQUET_API = "https://datasets-server.huggingface.co/parquet"
DEFAULT_DATASET = "jakegrigsby/metamon-raw-replays"
DEFAULT_USER_AGENT = "pokestrategist/0.0.1 (research; +https://github.com/)"


@dataclass(slots=True)
class DownloadStats:
    shards_seen: int = 0
    rows_matched: int = 0
    files_written: int = 0
    files_skipped: int = 0
    rows_skipped_existing: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a filtered metamon raw replay subset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--formatid", default="gen9ou")
    parser.add_argument("--min-rating", type=int, default=1550)
    parser.add_argument("--output-dir", default="data/raw/metamon_gen9ou_1550")
    parser.add_argument("--jsonl-output", default=None)
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recent-first", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--skip-cache-files", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-total", type=int, default=5)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--insecure", action="store_true")
    return parser.parse_args()


def _session(*, retry_total: int, retry_backoff: float, insecure: bool) -> requests.Session:
    session = requests.Session()
    session.headers.setdefault("User-Agent", DEFAULT_USER_AGENT)
    session.verify = not insecure
    retries = Retry(
        total=max(retry_total, 0),
        connect=max(retry_total, 0),
        read=max(retry_total, 0),
        backoff_factor=max(retry_backoff, 0.0),
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _load_existing_jsonl_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            replay_id = str(payload.get("id") or "").strip()
            if replay_id:
                existing_ids.add(replay_id)
    return existing_ids


def _parquet_urls(session: requests.Session, dataset: str, timeout: float) -> list[str]:
    response = session.get(DATASET_PARQUET_API, params={"dataset": dataset}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return [str(row["url"]) for row in payload["parquet_files"]]


def _select_parquet_urls(parquet_urls: Sequence[str], *, shard_start: int, max_shards: int | None) -> list[str]:
    selected = list(parquet_urls[shard_start:])
    if max_shards is not None:
        selected = selected[:max_shards]
    return selected


def _should_query_shards_incrementally(*, limit: int | None, recent_first: bool) -> bool:
    return limit is None and not recent_first


def _resolve_signed_url(session: requests.Session, parquet_url: str, timeout: float) -> str:
    response = session.head(parquet_url, allow_redirects=True, timeout=timeout)
    response.raise_for_status()
    return str(response.url)


def _connect(extension_dir: Path) -> duckdb.DuckDBPyConnection:
    extension_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA disable_progress_bar;")
    con.execute(f"SET extension_directory='{extension_dir.as_posix()}';")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    return con


def _read_parquet_expr(signed_urls: Sequence[str]) -> str:
    url_list = ", ".join(repr(url) for url in signed_urls)
    return f"read_parquet([{url_list}], union_by_name=true)"


def _filtered_select_sql(signed_urls: Sequence[str]) -> str:
    return (
        "SELECT id, format, players, log, uploadtime, rating "
        f"FROM {_read_parquet_expr(signed_urls)} "
        "WHERE formatid = ? AND TRY_CAST(rating AS INTEGER) >= ?"
    )


def _count_rows(
    con: duckdb.DuckDBPyConnection,
    signed_urls: Sequence[str],
    formatid: str,
    min_rating: int,
    *,
    limit: int | None = None,
    recent_first: bool = False,
) -> int:
    select_sql = _filtered_select_sql(signed_urls)
    params: list[object] = [formatid, min_rating]
    if recent_first:
        select_sql += " ORDER BY TRY_CAST(uploadtime AS BIGINT) DESC"
    if limit is not None:
        select_sql += " LIMIT ?"
        params.append(limit)
    query = f"SELECT COUNT(*) FROM ({select_sql}) AS filtered_rows"
    return int(con.execute(query, params).fetchone()[0])


def _iter_rows(
    con: duckdb.DuckDBPyConnection,
    signed_urls: Sequence[str],
    formatid: str,
    min_rating: int,
    batch_size: int,
    *,
    limit: int | None = None,
    recent_first: bool = False,
):
    query = _filtered_select_sql(signed_urls)
    params: list[object] = [formatid, min_rating]
    if recent_first:
        query += " ORDER BY TRY_CAST(uploadtime AS BIGINT) DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    cursor = con.execute(query, params)
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            return
        yield from rows


def _write_replay(output_dir: Path, replay_id: str, payload: dict[str, object], force: bool) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{replay_id}.json"
    if path.exists() and not force:
        return False
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    tmp_path.replace(path)
    return True


def main() -> None:
    args = _parse_args()
    if not args.count_only and args.skip_cache_files and not args.jsonl_output:
        raise SystemExit("--skip-cache-files requires --jsonl-output unless --count-only is set")
    if args.shard_start < 0:
        raise SystemExit("--shard-start must be a non-negative integer")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")
    if args.resume and args.force:
        raise SystemExit("--resume and --force cannot be used together")

    session = _session(retry_total=args.retry_total, retry_backoff=args.retry_backoff, insecure=args.insecure)
    parquet_urls = _parquet_urls(session, args.dataset, args.timeout)
    parquet_urls = _select_parquet_urls(
        parquet_urls,
        shard_start=args.shard_start,
        max_shards=args.max_shards,
    )

    con = _connect(Path(".duckdb_extensions"))
    stats = DownloadStats()
    output_dir = Path(args.output_dir)
    jsonl_path = Path(args.jsonl_output) if args.jsonl_output else None
    jsonl_handle = None
    existing_jsonl_ids: set[str] = set()
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume:
            existing_jsonl_ids = _load_existing_jsonl_ids(jsonl_path)
            jsonl_mode = "a"
        else:
            jsonl_mode = "w" if args.force else "x"
        jsonl_handle = jsonl_path.open(jsonl_mode, encoding="utf-8")

    try:
        if args.count_only:
            if _should_query_shards_incrementally(limit=args.limit, recent_first=args.recent_first):
                for parquet_url in tqdm(parquet_urls, desc="count shards", unit="shard"):
                    signed_url = _resolve_signed_url(session, parquet_url, args.timeout)
                    stats.shards_seen += 1
                    stats.rows_matched += _count_rows(
                        con,
                        [signed_url],
                        args.formatid,
                        args.min_rating,
                    )
            else:
                signed_urls = []
                for parquet_url in tqdm(parquet_urls, desc="resolve shards", unit="shard"):
                    signed_urls.append(_resolve_signed_url(session, parquet_url, args.timeout))
                    stats.shards_seen += 1
                stats.rows_matched = _count_rows(
                    con,
                    signed_urls,
                    args.formatid,
                    args.min_rating,
                    limit=args.limit,
                    recent_first=args.recent_first,
                )
        else:
            if not _should_query_shards_incrementally(limit=args.limit, recent_first=args.recent_first):
                print(
                    "Querying filtered rows across all shards"
                    + (" ordered by newest first" if args.recent_first else "")
                    + (f" with limit={args.limit}" if args.limit is not None else "")
                    + "...",
                    flush=True,
                )
                signed_urls = []
                for parquet_url in tqdm(parquet_urls, desc="resolve shards", unit="shard"):
                    signed_urls.append(_resolve_signed_url(session, parquet_url, args.timeout))
                    stats.shards_seen += 1
                row_iterable = _iter_rows(
                    con,
                    signed_urls,
                    args.formatid,
                    args.min_rating,
                    args.batch_size,
                    limit=args.limit,
                    recent_first=args.recent_first,
                )
            else:
                def _iter_signed_urls():
                    for parquet_url in tqdm(parquet_urls, desc="query shards", unit="shard"):
                        signed_url = _resolve_signed_url(session, parquet_url, args.timeout)
                        stats.shards_seen += 1
                        yield signed_url

                row_iterable = (
                    row
                    for signed_url in _iter_signed_urls()
                    for row in _iter_rows(
                        con,
                        [signed_url],
                        args.formatid,
                        args.min_rating,
                        args.batch_size,
                    )
                )

            for replay_id, replay_format, players, log, uploadtime, rating in row_iterable:
                stats.rows_matched += 1
                if replay_id in existing_jsonl_ids:
                    stats.rows_skipped_existing += 1
                    continue
                payload = {
                    "id": replay_id,
                    "format": replay_format,
                    "players": players,
                    "log": log,
                    "uploadtime": int(uploadtime),
                    "rating": int(rating),
                }
                if jsonl_handle is not None:
                    jsonl_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    existing_jsonl_ids.add(str(replay_id))
                if args.skip_cache_files:
                    continue
                if _write_replay(output_dir, replay_id, payload, force=args.force):
                    stats.files_written += 1
                else:
                    stats.files_skipped += 1
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    stats_json = json.dumps(stats.to_dict(), indent=2)
    print(stats_json)
    if args.stats_output:
        Path(args.stats_output).write_text(stats_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()