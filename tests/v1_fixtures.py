from __future__ import annotations

import json
from pathlib import Path

from pokestrategist.data.dataset_builder import build_samples_from_replay
from pokestrategist.data.protocol import parse_replay_payload


def sample_replay_payload(replay_id: str = "gen9ou-test") -> dict[str, object]:
    log_lines = [
        "|player|p1|Alice",
        "|player|p2|Bob",
        "|clearpoke",
        "|poke|p1|Great Tusk|",
        "|poke|p1|Dragapult|",
        "|poke|p1|Kingambit|",
        "|poke|p1|Rotom-Wash|",
        "|poke|p1|Cinderace|",
        "|poke|p1|Gliscor|",
        "|poke|p2|Gholdengo|",
        "|poke|p2|Ting-Lu|",
        "|poke|p2|Dragonite|",
        "|poke|p2|Rillaboom|",
        "|poke|p2|Corviknight|",
        "|poke|p2|Samurott-Hisui|",
        "|teampreview",
        "|teamsize|p1|6",
        "|teamsize|p2|6",
        "|start",
        "|switch|p1a: Great Tusk|Great Tusk, M|100/100",
        "|switch|p2a: Gholdengo|Gholdengo|100/100",
        "|turn|1",
        "|move|p1a: Great Tusk|Headlong Rush|p2a: Gholdengo",
        "|-damage|p2a: Gholdengo|55/100",
        "|move|p2a: Gholdengo|Make It Rain|p1a: Great Tusk",
        "|-damage|p1a: Great Tusk|45/100",
        "|turn|2",
        "|move|p1a: Great Tusk|Knock Off|p2a: Gholdengo",
        "|-damage|p2a: Gholdengo|35/100",
        "|move|p2a: Gholdengo|Nasty Plot|p2a: Gholdengo",
        "|turn|3",
        "|switch|p1a: Dragapult|Dragapult|100/100",
        "|move|p2a: Gholdengo|Shadow Ball|p1a: Dragapult",
        "|-damage|p1a: Dragapult|80/100",
        "|move|p1a: Dragapult|Draco Meteor|p2a: Gholdengo",
        "|-damage|p2a: Gholdengo|0 fnt",
        "|faint|p2a: Gholdengo",
        "|win|Alice",
    ]
    return {
        "id": replay_id,
        "format": "gen9ou",
        "players": ["Alice", "Bob"],
        "rating": 1700,
        "uploadtime": 1700000000,
        "log": "\n".join(log_lines),
    }


def write_dataset(path: Path, replay_ids: tuple[str, ...] = ("replay-a", "replay-b")) -> Path:
    lines: list[str] = []
    for replay_id in replay_ids:
        parsed = parse_replay_payload(sample_replay_payload(replay_id))
        lines.extend(sample.model_dump_json() for sample in build_samples_from_replay(parsed))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
