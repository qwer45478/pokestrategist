from __future__ import annotations

from pokestrategist.data.protocol import parse_replay_payload

from tests.v1_fixtures import sample_replay_payload


def test_parse_replay_payload_splits_turns_and_winner():
    parsed = parse_replay_payload(sample_replay_payload())

    assert parsed.meta.replay_id == "gen9ou-test"
    assert parsed.meta.battle_format == "gen9ou"
    assert parsed.meta.winner == "Alice"
    assert [turn.turn for turn in parsed.turns] == [1, 2, 3]
    assert parsed.preamble[0] == "|player|p1|Alice"
