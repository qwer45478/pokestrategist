from __future__ import annotations

import json

from pokestrategist.cli.download_metamon_subset import (
    _load_existing_jsonl_ids,
    _select_parquet_urls,
    _should_query_shards_incrementally,
)


def test_load_existing_jsonl_ids_reads_ids_once(tmp_path):
    jsonl_path = tmp_path / "metamon.jsonl"
    rows = [
        {"id": "replay-a", "format": "gen9ou"},
        {"id": "replay-b", "format": "gen9ou"},
        {"id": "replay-a", "format": "gen9ou"},
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    existing_ids = _load_existing_jsonl_ids(jsonl_path)

    assert existing_ids == {"replay-a", "replay-b"}


def test_select_parquet_urls_supports_shard_offset_and_limit():
    parquet_urls = [f"shard-{index}" for index in range(6)]

    selected = _select_parquet_urls(parquet_urls, shard_start=2, max_shards=3)

    assert selected == ["shard-2", "shard-3", "shard-4"]


def test_should_query_shards_incrementally_only_without_global_ordering():
    assert _should_query_shards_incrementally(limit=None, recent_first=False) is True
    assert _should_query_shards_incrementally(limit=100, recent_first=False) is False
    assert _should_query_shards_incrementally(limit=None, recent_first=True) is False