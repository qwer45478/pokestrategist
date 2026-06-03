from __future__ import annotations

import argparse
import json
from pathlib import Path

from pokestrategist.data.team_preview_prior import build_team_preview_prior_from_processed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a team-preview hidden-set prior from processed decision JSONL.")
    parser.add_argument("--data", required=True, help="Processed decision JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSON path for the preview prior catalog.")
    parser.add_argument("--min-team-size", type=int, default=2)
    parser.add_argument("--min-evidence-fields", type=int, default=1)
    parser.add_argument("--min-template-count", type=int, default=2)
    parser.add_argument("--teammate-alpha", type=float, default=0.25)
    parser.add_argument("--team-condition-weight", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    catalog = build_team_preview_prior_from_processed(
        args.data,
        min_team_size=args.min_team_size,
        min_evidence_fields=args.min_evidence_fields,
        min_template_count=args.min_template_count,
        teammate_alpha=args.teammate_alpha,
        team_condition_weight=args.team_condition_weight,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "example_count": catalog.example_count,
                "species_with_templates": len(catalog.templates_by_species),
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()