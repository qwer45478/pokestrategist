"""Build a clean v1 decision dataset from cached replay JSON or replay JSONL."""

from __future__ import annotations

import argparse
import json

from pokestrategist.data.dataset_builder import build_dataset_from_path
from pokestrategist.data.schema import PlayerSide


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v1 decision-model JSONL from replay cache directories or replay JSONL files.")
    parser.add_argument("--input", dest="input_path", default="data/raw/replays")
    parser.add_argument("--input-dir", dest="input_path")
    parser.add_argument("--output", default="data/processed/pokestrategist_v1.jsonl")
    parser.add_argument("--perspectives", choices=("both", "p1", "p2"), default="both")
    return parser.parse_args()


def _perspectives(name: str) -> tuple[PlayerSide, ...]:
    if name == "both":
        return (PlayerSide.P1, PlayerSide.P2)
    if name == "p1":
        return (PlayerSide.P1,)
    return (PlayerSide.P2,)


def main() -> None:
    args = _parse_args()
    stats = build_dataset_from_path(args.input_path, args.output, perspectives=_perspectives(args.perspectives))
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()