from __future__ import annotations

from pokestrategist.data.dataset_builder import build_samples_from_replay
from pokestrategist.data.schema import ActionHead
from pokestrategist.data.protocol import parse_replay_payload
from pokestrategist.data.schema import PlayerSide

from tests.v1_fixtures import sample_replay_payload


def test_build_samples_from_replay_emits_legal_actions():
    parsed = parse_replay_payload(sample_replay_payload())

    samples = build_samples_from_replay(parsed)

    assert len(samples) == 6
    p1_turn2 = next(sample for sample in samples if sample.perspective is PlayerSide.P1 and sample.turn == 2)
    assert p1_turn2.observation.self_side.active_species == "Great Tusk"
    assert any(action.move_token == "Headlong Rush" for action in p1_turn2.recoverable_actions)
    assert not any(action.move_token == "Knock Off" and action.head is ActionHead.MOVE for action in p1_turn2.legal_actions)
    assert any(action.switch_species == "Rotom-Wash" and action.head is ActionHead.SWITCH for action in p1_turn2.legal_actions)
    assert p1_turn2.hidden_move_slots == 3
    assert p1_turn2.family_target == "utility"
