"""Build full Gen 9 static rule catalogs plus a Gen 9 OU active subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokestrategist.data.static_catalog_builder import ActiveSubsetConfig
from pokestrategist.data.static_catalog_builder import build_catalogs_from_sources
from pokestrategist.data.static_catalog_builder import collect_observed_usage_from_processed
from pokestrategist.data.static_catalog_builder import previous_month
from pokestrategist.data.static_catalog_builder import write_json

DEFAULT_USER_AGENT = "pokestrategist/0.0.1 (static-catalog; research)"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Pokemon Showdown static data and Smogon Gen 9 OU usage stats."
    )
    parser.add_argument("--output-dir", default="data/static/gen9")
    parser.add_argument("--usage-output-dir", default="data/static/gen9ou")
    parser.add_argument("--formatid", default="gen9ou")
    parser.add_argument("--stats-month", default=previous_month())
    parser.add_argument("--usage-rating-cutoff", type=int, default=0)
    parser.add_argument("--species-min-usage", type=float, default=0.05)
    parser.add_argument("--species-min-raw-count", type=int, default=100)
    parser.add_argument("--move-min-usage", type=float, default=1.0)
    parser.add_argument("--item-min-usage", type=float, default=1.0)
    parser.add_argument("--ability-min-usage", type=float, default=1.0)
    parser.add_argument("--max-species", type=int, default=None)
    parser.add_argument("--processed-data", default=None, help="Optional local decision JSONL whose observed facts are always preserved.")
    parser.add_argument("--processed-max-samples", type=int, default=None)
    parser.add_argument("--include-nonstandard", action="store_true")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    observed_usage = None
    if args.processed_data:
        observed_usage = collect_observed_usage_from_processed(args.processed_data, max_samples=args.processed_max_samples)

    payloads = build_catalogs_from_sources(
        month=args.stats_month,
        formatid=args.formatid,
        rating_cutoff=args.usage_rating_cutoff,
        timeout=args.timeout,
        user_agent=args.user_agent,
        include_nonstandard=args.include_nonstandard,
        active_config=ActiveSubsetConfig(
            species_min_usage=args.species_min_usage,
            species_min_raw_count=args.species_min_raw_count,
            move_min_usage=args.move_min_usage,
            item_min_usage=args.item_min_usage,
            ability_min_usage=args.ability_min_usage,
            max_species=args.max_species,
        ),
        observed_usage=observed_usage,
    )

    output_dir = Path(args.output_dir)
    usage_output_dir = Path(args.usage_output_dir)
    write_json(output_dir / "species.json", payloads["species"], indent=args.indent)
    write_json(output_dir / "moves.json", payloads["moves"], indent=args.indent)
    write_json(output_dir / "items.json", payloads["items"], indent=args.indent)
    write_json(output_dir / "abilities.json", payloads["abilities"], indent=args.indent)
    write_json(output_dir / "manifest.json", payloads["manifest"], indent=args.indent)
    write_json(usage_output_dir / "usage.json", payloads["usage"], indent=args.indent)
    write_json(usage_output_dir / "active_subset.json", payloads["active_subset"], indent=args.indent)
    write_json(usage_output_dir / "observed_usage.json", payloads["observed_usage"], indent=args.indent)

    print(json.dumps(payloads["manifest"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()