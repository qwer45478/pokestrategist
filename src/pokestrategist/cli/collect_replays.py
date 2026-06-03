"""Collect raw replay JSON from public Showdown search pages."""

from __future__ import annotations

import argparse
import json

from pokestrategist.data.replay_client import ReplayClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw public Pokemon Showdown replay JSON.")
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--max-replays", type=int, default=None)
    parser.add_argument("--min-rating", type=int, default=1550)
    parser.add_argument("--cache-dir", default="data/raw/replays")
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client = ReplayClient(cache_dir=args.cache_dir, request_delay=args.request_delay)
    hits = client.search(battle_format=args.format, max_pages=args.max_pages, min_rating=args.min_rating)
    count = 0
    for hit in hits:
        client.fetch(hit.replay_id, force=args.force_refresh)
        count += 1
        if args.max_replays is not None and count >= args.max_replays:
            break
    print(json.dumps({"raw_replays_fetched": count, "cache_dir": args.cache_dir}, indent=2))


if __name__ == "__main__":
    main()