"""Dataset building for the clean v1 decision model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence, TypeVar

from pokepilot.data.protocol import ParsedReplay, TurnWindow, parse_replay_payload
from pokepilot.data.schema import (
    ActionHead,
    BattleObservation,
    BeliefSummaryTarget,
    DecisionSample,
    FieldSummary,
    FutureSummaryTarget,
    ParticlePosteriorTarget,
    PhaseLabel,
    PlanHypothesis,
    PlanLabel,
    PlayerSide,
    RevealLikelihoodTarget,
    ResponseCluster,
    ResourceLedgerTarget,
    SemanticIntent,
    SideSummary,
    StructuredAction,
    TypedConsequenceTarget,
)

HISTORY_WINDOW = 16
MAX_MOVE_SLOTS = 4
COARSE_FAMILIES = {"attack", "hazard", "recovery", "pivot", "setup", "status", "utility", "protect", "switch"}
PROTECTIVE_MOVES = {"protect", "detect", "kings shield", "spiky shield", "baneful bunker"}
PIVOT_MOVES = {"u-turn", "volt switch", "flip turn", "teleport", "parting shot", "chilly reception"}
RECOVERY_MOVES = {"recover", "roost", "slack off", "wish", "soft-boiled", "moonlight", "morning sun"}
SETUP_MOVES = {"swords dance", "nasty plot", "dragon dance", "calm mind", "bulk up", "quiver dance", "agility", "trailblaze"}
HAZARD_MOVES = {"stealth rock", "spikes", "toxic spikes", "sticky web"}
STATUS_MOVES = {"will-o-wisp", "thunder wave", "toxic", "glare", "spore", "sleep powder", "stun spore", "yawn"}
UTILITY_MOVES = {"taunt", "encore", "knock off", "trick", "substitute", "defog", "rapid spin", "mortal spin", "court change", "parting shot"}
HAZARD_KEYS = {"stealth rock": "stealthrock", "spikes": "spikes", "toxic spikes": "toxicspikes", "sticky web": "stickyweb"}
SCREEN_KEYS = {"reflect": "reflect", "light screen": "lightscreen", "aurora veil": "auroraveil", "safeguard": "safeguard"}
PLAN_LABEL_ORDER = [PlanLabel.PRESSURE, PlanLabel.STABILIZE, PlanLabel.PRESERVE, PlanLabel.CONVERT]
PHASE_LABEL_ORDER = [PhaseLabel.SCOUT, PhaseLabel.RESOURCE, PhaseLabel.TEMPO, PhaseLabel.CONVERT_TRANSITION]
_LabelT = TypeVar("_LabelT")


@dataclass
class BuildStats:
    replays_seen: int = 0
    replays_written: int = 0
    replays_failed: int = 0
    samples_written: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class _MutableSideState:
    active_species: str | None = None
    active_hp: float = 1.0
    active_status: str | None = None
    fainted_species: list[str] | None = None
    tera_used: bool = False
    team_order: list[str] | None = None
    species_hp: dict[str, float] | None = None
    species_status: dict[str, str | None] | None = None
    revealed_moves: dict[str, set[str]] | None = None
    revealed_items: dict[str, str] | None = None
    revealed_abilities: dict[str, str] | None = None
    revealed_tera_types: dict[str, str] | None = None
    hazards: dict[str, int] | None = None
    screens: dict[str, bool] | None = None

    def __post_init__(self) -> None:
        self.fainted_species = self.fainted_species or []
        self.team_order = self.team_order or []
        self.species_hp = self.species_hp or {}
        self.species_status = self.species_status or {}
        self.revealed_moves = self.revealed_moves or {}
        self.revealed_items = self.revealed_items or {}
        self.revealed_abilities = self.revealed_abilities or {}
        self.revealed_tera_types = self.revealed_tera_types or {}
        self.hazards = self.hazards or {"stealthrock": 0, "spikes": 0, "toxicspikes": 0, "stickyweb": 0}
        self.screens = self.screens or {"reflect": False, "lightscreen": False, "auroraveil": False, "safeguard": False}

    def to_summary(self) -> SideSummary:
        return SideSummary(
            active_species=self.active_species,
            active_hp=self.active_hp,
            active_status=self.active_status,
            fainted_count=len(self.fainted_species),
            tera_used=self.tera_used,
            team_order=list(self.team_order),
            fainted_species=list(self.fainted_species),
            species_hp=dict(self.species_hp),
            species_status=dict(self.species_status),
            revealed_moves={species: sorted(moves) for species, moves in self.revealed_moves.items()},
            revealed_items=dict(self.revealed_items),
            revealed_abilities=dict(self.revealed_abilities),
            revealed_tera_types=dict(self.revealed_tera_types),
            hazards=dict(self.hazards),
            screens=dict(self.screens),
        )


class _TraceState:
    def __init__(self) -> None:
        self.sides = {PlayerSide.P1: _MutableSideState(), PlayerSide.P2: _MutableSideState()}
        self.field = FieldSummary()
        self.history_actions: list[str] = []
        self.pending_forced_switch: set[PlayerSide] = set()

    def _ensure_species(self, side: PlayerSide, species: str | None) -> None:
        if not species:
            return
        state = self.sides[side]
        if species not in state.team_order:
            state.team_order.append(species)
        state.species_hp.setdefault(species, 1.0)
        state.species_status.setdefault(species, None)

    def snapshot(self, parsed: ParsedReplay, turn: int, perspective: PlayerSide, protocol_window: list[str]) -> BattleObservation:
        return BattleObservation(
            replay_id=parsed.meta.replay_id,
            battle_format=parsed.meta.battle_format,
            turn=turn,
            perspective=perspective,
            rating=parsed.meta.rating,
            self_side=self.sides[perspective].to_summary(),
            opp_side=self.sides[perspective.opponent].to_summary(),
            field=self.field.model_copy(deep=True),
            history_actions=list(self.history_actions[-HISTORY_WINDOW:]),
            protocol_window=list(protocol_window),
        )

    def recoverable_actions(self, perspective: PlayerSide) -> list[StructuredAction]:
        state = self.sides[perspective]
        actions: list[StructuredAction] = []
        active = state.active_species
        if active:
            for move_token in sorted(state.revealed_moves.get(active, set())):
                family = _coarse_move_family(move_token)
                actions.append(StructuredAction(actor=perspective, head=ActionHead.MOVE, move_token=move_token, move_family=family))
                if not state.tera_used:
                    actions.append(
                        StructuredAction(
                            actor=perspective,
                            head=ActionHead.TERA_MOVE,
                            move_token=move_token,
                            move_family=family,
                            tera=True,
                        )
                    )
        for slot, species in enumerate(state.team_order):
            if species == active or species in state.fainted_species:
                continue
            actions.append(
                StructuredAction(
                    actor=perspective,
                    head=ActionHead.SWITCH,
                    move_family="switch",
                    switch_slot=slot,
                    switch_species=species,
                )
            )
        deduped: dict[str, StructuredAction] = {}
        for action in actions:
            deduped.setdefault(action.key(), action)
        return list(deduped.values())

    def legal_actions(self, perspective: PlayerSide) -> list[StructuredAction]:
        state = self.sides[perspective]
        actions: list[StructuredAction] = []
        active = state.active_species
        if active:
            for move_token in sorted(state.revealed_moves.get(active, set())):
                family = _coarse_move_family(move_token)
                actions.append(StructuredAction(actor=perspective, head=ActionHead.MOVE, move_token=move_token, move_family=family))
                if not state.tera_used:
                    actions.append(
                        StructuredAction(
                            actor=perspective,
                            head=ActionHead.TERA_MOVE,
                            move_token=move_token,
                            move_family=family,
                            tera=True,
                        )
                    )
        for slot, species in enumerate(state.team_order):
            if species == active or species in state.fainted_species:
                continue
            actions.append(
                StructuredAction(
                    actor=perspective,
                    head=ActionHead.SWITCH,
                    move_family="switch",
                    switch_slot=slot,
                    switch_species=species,
                )
            )
        deduped: dict[str, StructuredAction] = {}
        for action in actions:
            deduped.setdefault(action.key(), action)
        ordered = sorted(
            deduped.values(),
            key=lambda action: (
                0 if action.head is ActionHead.MOVE else 1 if action.head is ActionHead.TERA_MOVE else 2,
                action.move_token or "",
                action.switch_slot if action.switch_slot is not None else 99,
                action.switch_species or "",
            ),
        )
        return ordered

    def extract_turn_actions(self, window: TurnWindow) -> dict[PlayerSide, StructuredAction]:
        actions: dict[PlayerSide, StructuredAction] = {}
        ignored_forced = set(self.pending_forced_switch)
        tera_sides = {
            _parse_player_side(parts[2])
            for parts in (_split(line) for line in window.messages)
            if len(parts) > 2 and parts[1] == "-terastallize" and _parse_player_side(parts[2]) is not None
        }
        for line in window.messages:
            parts = _split(line)
            if len(parts) < 2:
                continue
            cmd = parts[1]
            if cmd == "drag":
                continue
            if cmd == "switch" and len(parts) > 3:
                side = _parse_player_side(parts[2])
                if side is None:
                    continue
                species = _normalize_species(parts[3]) or _extract_species(parts[2])
                self._ensure_species(side, species)
                if side in ignored_forced:
                    ignored_forced.discard(side)
                    continue
                if side not in actions:
                    slot = self.sides[side].team_order.index(species) if species in self.sides[side].team_order else None
                    actions[side] = StructuredAction(actor=side, head=ActionHead.SWITCH, move_family="switch", switch_slot=slot, switch_species=species)
            if cmd == "move" and len(parts) > 3:
                side = _parse_player_side(parts[2])
                if side is None or side in actions:
                    continue
                move_token = parts[3]
                head = ActionHead.TERA_MOVE if side in tera_sides else ActionHead.MOVE
                actions[side] = StructuredAction(
                    actor=side,
                    head=head,
                    move_token=move_token,
                    move_family=_coarse_move_family(move_token),
                    tera=head is ActionHead.TERA_MOVE,
                )
        return actions

    def apply_turn(self, window: TurnWindow, actions: dict[PlayerSide, StructuredAction]) -> None:
        for line in window.messages:
            parts = _split(line)
            if len(parts) < 2:
                continue
            cmd = parts[1]
            if cmd == "poke" and len(parts) > 3:
                side = _parse_player_side(parts[2])
                species = _normalize_species(parts[3])
                if side is not None:
                    self._ensure_species(side, species)
                continue
            if cmd in {"switch", "drag"} and len(parts) > 3:
                side = _parse_player_side(parts[2])
                if side is None:
                    continue
                species = _normalize_species(parts[3]) or _extract_species(parts[2])
                hp, status = _parse_hp_status(parts[3])
                self._ensure_species(side, species)
                state = self.sides[side]
                state.active_species = species
                state.active_hp = hp
                state.active_status = status
                state.species_hp[species] = hp
                state.species_status[species] = status
                self.pending_forced_switch.discard(side)
            elif cmd == "move" and len(parts) > 3:
                side = _parse_player_side(parts[2])
                if side is None:
                    continue
                state = self.sides[side]
                species = state.active_species or _extract_species(parts[2]) or "unknown"
                self._ensure_species(side, species)
                state.revealed_moves.setdefault(species, set()).add(parts[3])
            elif cmd in {"-damage", "-heal"} and len(parts) > 3:
                side = _parse_player_side(parts[2])
                if side is None:
                    continue
                hp, status = _parse_hp_status(parts[3])
                state = self.sides[side]
                state.active_hp = hp
                if state.active_species:
                    state.species_hp[state.active_species] = hp
                if status is not None:
                    state.active_status = status
                    if state.active_species:
                        state.species_status[state.active_species] = status
            elif cmd == "-status" and len(parts) > 3:
                side = _parse_player_side(parts[2])
                if side is not None:
                    self.sides[side].active_status = parts[3]
                    if self.sides[side].active_species:
                        self.sides[side].species_status[self.sides[side].active_species] = parts[3]
            elif cmd == "-curestatus" and len(parts) > 2:
                side = _parse_player_side(parts[2])
                if side is not None:
                    self.sides[side].active_status = None
                    if self.sides[side].active_species:
                        self.sides[side].species_status[self.sides[side].active_species] = None
            elif cmd == "faint" and len(parts) > 2:
                side = _parse_player_side(parts[2])
                if side is not None:
                    state = self.sides[side]
                    species = state.active_species or _extract_species(parts[2])
                    if species and species not in state.fainted_species:
                        state.fainted_species.append(species)
                    if species:
                        state.species_hp[species] = 0.0
                        state.species_status[species] = "fnt"
                    state.active_species = None
                    state.active_hp = 0.0
                    self.pending_forced_switch.add(side)
            elif cmd in {"-item", "-enditem"} and len(parts) > 3:
                side = _parse_player_side(parts[2])
                if side is not None:
                    species = self.sides[side].active_species or _extract_species(parts[2]) or "unknown"
                    self.sides[side].revealed_items[species] = parts[3]
            elif cmd == "-ability" and len(parts) > 3:
                side = _parse_player_side(parts[2])
                if side is not None:
                    species = self.sides[side].active_species or _extract_species(parts[2]) or "unknown"
                    self.sides[side].revealed_abilities[species] = parts[3]
            elif cmd == "-terastallize" and len(parts) > 3:
                side = _parse_player_side(parts[2])
                if side is not None:
                    species = self.sides[side].active_species or _extract_species(parts[2]) or "unknown"
                    self.sides[side].tera_used = True
                    self.sides[side].revealed_tera_types[species] = parts[3]
            elif cmd == "-sidestart" and len(parts) > 3:
                side = _parse_side_marker(parts[2])
                if side is not None:
                    _apply_side_effect(self.sides[side], parts[3], start=True)
            elif cmd == "-sideend" and len(parts) > 3:
                side = _parse_side_marker(parts[2])
                if side is not None:
                    _apply_side_effect(self.sides[side], parts[3], start=False)
            elif cmd == "-weather" and len(parts) > 2:
                self.field.weather = "none" if parts[2].lower() == "none" else parts[2].lower()
            elif cmd == "-fieldstart" and len(parts) > 2:
                effect = parts[2].lower()
                if "trick room" in effect:
                    self.field.trick_room = True
                elif "terrain" in effect:
                    self.field.terrain = effect.replace("move:", "").strip()
            elif cmd == "-fieldend" and len(parts) > 2:
                effect = parts[2].lower()
                if "trick room" in effect:
                    self.field.trick_room = False
                elif "terrain" in effect:
                    self.field.terrain = "none"
        for side, action in actions.items():
            token = action.move_token or action.switch_species or action.head.value
            self.history_actions.append(f"{side.value}:{action.head.value}:{token}")


def _split(line: str) -> list[str]:
    return line.split("|")


def _parse_player_side(actor: str) -> PlayerSide | None:
    token = actor.strip().lower()
    if token.startswith("p1"):
        return PlayerSide.P1
    if token.startswith("p2"):
        return PlayerSide.P2
    return None


def _parse_side_marker(actor: str) -> PlayerSide | None:
    token = actor.strip().lower()
    if token.startswith("p1"):
        return PlayerSide.P1
    if token.startswith("p2"):
        return PlayerSide.P2
    return None


def _extract_species(actor: str) -> str | None:
    if ":" not in actor:
        return None
    return actor.split(":", 1)[1].split(",", 1)[0].strip() or None


def _normalize_species(text: str) -> str | None:
    token = text.split(",", 1)[0].strip()
    return token or None


def _parse_hp_status(hp_text: str) -> tuple[float, str | None]:
    token = hp_text.strip()
    if token in {"0 fnt", "0/100 fnt"}:
        return 0.0, "fnt"
    parts = token.split()
    hp_part = parts[0]
    status = parts[1] if len(parts) > 1 else None
    if "/" in hp_part:
        current, maximum = hp_part.split("/", 1)
        try:
            current_hp = float(current)
            max_hp = float(maximum)
            return (current_hp / max_hp) if max_hp else 0.0, status
        except ValueError:
            return 1.0, status
    if hp_part.endswith("%"):
        try:
            return float(hp_part[:-1]) / 100.0, status
        except ValueError:
            return 1.0, status
    return 1.0, status


def _coarse_move_family(move_token: str | None) -> str:
    token = (move_token or "").strip().lower()
    if token in HAZARD_MOVES:
        return "hazard"
    if token in RECOVERY_MOVES:
        return "recovery"
    if token in PIVOT_MOVES:
        return "pivot"
    if token in SETUP_MOVES:
        return "setup"
    if token in PROTECTIVE_MOVES:
        return "protect"
    if token in STATUS_MOVES:
        return "status"
    if token in UTILITY_MOVES:
        return "utility"
    return "attack"


def _apply_side_effect(side_state: _MutableSideState, effect_token: str, *, start: bool) -> None:
    effect = effect_token.replace("move:", "").strip().lower()
    if effect in HAZARD_KEYS:
        key = HAZARD_KEYS[effect]
        if start:
            max_count = 3 if key == "spikes" else 2 if key == "toxicspikes" else 1
            side_state.hazards[key] = min(side_state.hazards[key] + 1, max_count)
        else:
            side_state.hazards[key] = 0
        return
    if effect in SCREEN_KEYS:
        side_state.screens[SCREEN_KEYS[effect]] = start


def _normalize_weights(items: list[float]) -> list[float]:
    total = sum(items)
    if total <= 0.0:
        return [1.0 / len(items)] * len(items)
    return [value / total for value in items]


def _top_plan_hypotheses(weights: list[float]) -> list[PlanHypothesis]:
    weighted = list(zip(PLAN_LABEL_ORDER, weights))
    weighted.sort(key=lambda pair: pair[1], reverse=True)
    return [PlanHypothesis(label=label, weight=weight) for label, weight in weighted[:3] if weight > 0.05]


def _belief_summary(observation: BattleObservation) -> BeliefSummaryTarget:
    opp = observation.opp_side
    active = opp.active_species or ""
    known_moves = len(opp.revealed_moves.get(active, [])) / 4.0 if active else 0.0
    known_speed = 0.5 if observation.turn >= 4 else 0.0
    role_confidence = min(1.0, 0.35 * known_moves + 0.25 * float(active in opp.revealed_items) + 0.2 * float(active in opp.revealed_abilities))
    return BeliefSummaryTarget(
        item=1.0 if active in opp.revealed_items else 0.0,
        ability=1.0 if active in opp.revealed_abilities else 0.0,
        speed=known_speed,
        moves=known_moves,
        role=role_confidence,
        tera=1.0 if active in opp.revealed_tera_types else 0.0,
    )


def _belief_target(observation: BattleObservation) -> list[float]:
    return _belief_summary(observation).as_vector()


def _bounded(value: float, scale: float = 1.0) -> float:
    if scale <= 0.0:
        return max(-1.0, min(value, 1.0))
    return max(-1.0, min(value / scale, 1.0))


def _opponent_archetype_mix(observation: BattleObservation) -> tuple[float, float, float]:
    opp = observation.opp_side
    revealed_moves = [move for species in opp.team_order for move in opp.revealed_moves.get(species, [])]
    setup = sum(1 for move in revealed_moves if _coarse_move_family(move) == "setup")
    pivot = sum(1 for move in revealed_moves if _coarse_move_family(move) == "pivot")
    recovery = sum(1 for move in revealed_moves if _coarse_move_family(move) == "recovery")
    hazards = sum(1 for move in revealed_moves if _coarse_move_family(move) == "hazard")
    status = sum(1 for move in revealed_moves if _coarse_move_family(move) == "status")
    offense = 1.0 + 0.8 * setup + 0.4 * pivot + 0.2 * float(observation.turn <= 5)
    balance = 1.0 + 0.5 * pivot + 0.45 * hazards + 0.35 * recovery + 0.15 * setup
    stall = 1.0 + 0.8 * recovery + 0.55 * hazards + 0.45 * status
    normalized = _normalize_weights([offense, balance, stall])
    return normalized[0], normalized[1], normalized[2]


def _particle_posterior(observation: BattleObservation) -> ParticlePosteriorTarget:
    opp = observation.opp_side
    roster = list(opp.team_order)
    if not roster and opp.active_species:
        roster = [opp.active_species]
    team_size = max(len(roster), 1)
    moves_known = sum(min(len(opp.revealed_moves.get(species, [])), MAX_MOVE_SLOTS) for species in roster) / float(MAX_MOVE_SLOTS * team_size)
    item_known = sum(1 for species in roster if species in opp.revealed_items) / float(team_size)
    ability_known = sum(1 for species in roster if species in opp.revealed_abilities) / float(team_size)
    tera_known = sum(1 for species in roster if species in opp.revealed_tera_types) / float(team_size)
    active = opp.active_species or ""
    active_axes = [
        min(len(opp.revealed_moves.get(active, [])) / float(MAX_MOVE_SLOTS), 1.0) if active else 0.0,
        1.0 if active in opp.revealed_items else 0.0,
        1.0 if active in opp.revealed_abilities else 0.0,
        1.0 if active in opp.revealed_tera_types else 0.0,
    ]
    archetype_offense, archetype_balance, archetype_stall = _opponent_archetype_mix(observation)
    return ParticlePosteriorTarget(
        team_seen=min(len(roster) / 6.0, 1.0),
        team_alive=max(0.0, float(team_size - opp.fainted_count) / float(team_size)),
        moves_known=moves_known,
        item_known=item_known,
        ability_known=ability_known,
        tera_known=tera_known,
        active_set_known=sum(active_axes) / len(active_axes),
        archetype_offense=archetype_offense,
        archetype_balance=archetype_balance,
        archetype_stall=archetype_stall,
    )


def _resource_ledger(observation: BattleObservation) -> ResourceLedgerTarget:
    own = observation.self_side
    opp = observation.opp_side
    keep = 1.0 if own.fainted_count <= opp.fainted_count else 0.0
    hazard = float(sum(opp.hazards.values()) - sum(own.hazards.values()))
    hp = own.active_hp - opp.active_hp
    tera = float(not own.tera_used)
    sac = max(0.0, 1.0 - own.fainted_count / 6.0)
    speed = 0.5 if observation.field.trick_room else float(own.active_hp >= opp.active_hp)
    return ResourceLedgerTarget(keep=keep, hazard=hazard, speed=speed, hp=hp, tera=tera, sac=sac)


def _resource_target(observation: BattleObservation) -> list[float]:
    return _resource_ledger(observation).as_vector()


def _semantic_intent_for_action(action: StructuredAction, observation: BattleObservation) -> SemanticIntent:
    if action.head is ActionHead.SWITCH:
        return SemanticIntent.PRESERVE if observation.self_side.active_hp < 0.4 else SemanticIntent.TEMPO
    token = (action.move_token or "").lower()
    if token in PROTECTIVE_MOVES:
        return SemanticIntent.SCOUT
    if token in HAZARD_MOVES:
        return SemanticIntent.PUNISH
    if token in PIVOT_MOVES:
        return SemanticIntent.TEMPO
    if token in RECOVERY_MOVES:
        return SemanticIntent.PRESERVE
    if token in SETUP_MOVES:
        return SemanticIntent.CONVERT
    return SemanticIntent.SAFE


def _unlock_target(observation: BattleObservation) -> float:
    own = observation.self_side
    opp = observation.opp_side
    lead = (opp.fainted_count - own.fainted_count) / 6.0
    pressure = own.active_hp - opp.active_hp
    tera_bonus = 0.15 if not own.tera_used else 0.0
    return max(0.0, min(1.0, 0.5 + 0.5 * lead + 0.25 * pressure + tera_bonus))


def _plan_posterior(observation: BattleObservation, semantic: SemanticIntent, unlock_target: float) -> list[float]:
    own = observation.self_side
    opp = observation.opp_side
    pressure = 0.15 + 0.4 * float(semantic in {SemanticIntent.PUNISH, SemanticIntent.TEMPO}) + 0.2 * max(0.0, own.active_hp - opp.active_hp)
    stabilize = 0.2 + 0.2 * float(semantic is SemanticIntent.SAFE) + 0.1 * float(observation.turn <= 3)
    preserve = 0.15 + 0.45 * float(semantic is SemanticIntent.PRESERVE) + 0.25 * float(own.active_hp < 0.45)
    convert = 0.05 + 0.55 * unlock_target + 0.35 * float(semantic is SemanticIntent.CONVERT)
    return _normalize_weights([pressure, stabilize, preserve, convert])


def _phase_posterior(observation: BattleObservation, semantic: SemanticIntent, unlock_target: float) -> list[float]:
    opp = observation.opp_side
    scout = 0.15 + 0.5 * float(observation.turn <= 4) + 0.35 * float(semantic is SemanticIntent.SCOUT) + 0.15 * float(len(opp.revealed_moves.get(opp.active_species or "", [])) <= 1)
    resource = 0.15 + 0.35 * float(semantic is SemanticIntent.PRESERVE) + 0.25 * float(observation.self_side.active_hp < 0.5)
    tempo = 0.15 + 0.45 * float(semantic in {SemanticIntent.TEMPO, SemanticIntent.PUNISH}) + 0.15 * float(sum(opp.hazards.values()) >= sum(observation.self_side.hazards.values()))
    convert = 0.05 + 0.8 * unlock_target + 0.25 * float(semantic is SemanticIntent.CONVERT)
    return _normalize_weights([scout, resource, tempo, convert])


def _argmax_label(weights: list[float], labels: list[_LabelT]) -> _LabelT:
    best_index = max(range(len(weights)), key=lambda index: weights[index])
    return labels[best_index]


def _phase_label(observation: BattleObservation, semantic: SemanticIntent, unlock_target: float) -> PhaseLabel:
    return _argmax_label(_phase_posterior(observation, semantic, unlock_target), PHASE_LABEL_ORDER)


def _plan_label(observation: BattleObservation, semantic: SemanticIntent, unlock_target: float) -> PlanLabel:
    return _argmax_label(_plan_posterior(observation, semantic, unlock_target), PLAN_LABEL_ORDER)


def _response_cluster(action: StructuredAction | None) -> ResponseCluster:
    if action is None:
        return ResponseCluster.HOLD
    if action.head is ActionHead.SWITCH:
        if action.switch_slot is not None and action.switch_slot <= 1:
            return ResponseCluster.DEFENSIVE_SWITCH
        return ResponseCluster.OFFENSIVE_SWITCH
    token = (action.move_token or "").lower()
    if token in PROTECTIVE_MOVES:
        return ResponseCluster.PROTECTIVE_MOVE
    if token in SETUP_MOVES:
        return ResponseCluster.SETUP
    return ResponseCluster.AGGRESSIVE_MOVE


def _family_target(action: StructuredAction) -> str:
    if action.head is ActionHead.SWITCH:
        return "switch"
    family = action.move_family or _coarse_move_family(action.move_token)
    return family if family in COARSE_FAMILIES else "utility"


def _future_summary(current: DecisionSample, future: DecisionSample | None) -> FutureSummaryTarget:
    if future is None:
        return FutureSummaryTarget()
    current_obs = current.observation
    future_obs = future.observation
    current_plan = current.plan_posterior or [1.0 if label is current.plan_label else 0.0 for label in PLAN_LABEL_ORDER]
    future_plan = future.plan_posterior or [1.0 if label is future.plan_label else 0.0 for label in PLAN_LABEL_ORDER]
    current_phase = current.phase_posterior or [1.0 if label is current.phase_label else 0.0 for label in PHASE_LABEL_ORDER]
    future_phase = future.phase_posterior or [1.0 if label is future.phase_label else 0.0 for label in PHASE_LABEL_ORDER]
    return FutureSummaryTarget(
        delta_plan=0.5 * sum(abs(future_value - current_value) for current_value, future_value in zip(current_plan, future_plan)),
        delta_belief=current.belief_summary.uncertainty() - future.belief_summary.uncertainty(),
        delta_resource=future.resource_ledger.scalar_value() - current.resource_ledger.scalar_value(),
        delta_phase=0.5 * sum(abs(future_value - current_value) for current_value, future_value in zip(current_phase, future_phase)),
        delta_unlock=future.unlock_target - current.unlock_target,
        safe_score=future_obs.self_side.active_hp - current_obs.self_side.active_hp,
        tempo_score=1.0 if future_obs.opp_side.active_species != current_obs.opp_side.active_species else 0.0,
        convert_score=future.unlock_target,
    )


def _reveal_likelihood(current: DecisionSample, future_window: Sequence[DecisionSample]) -> RevealLikelihoodTarget:
    if not future_window:
        return RevealLikelihoodTarget()
    current_opp = current.observation.opp_side
    active = current_opp.active_species or ""
    if not active:
        return RevealLikelihoodTarget()
    current_moves = set(current_opp.revealed_moves.get(active, []))
    move_reveal = 0.0
    item_reveal = 0.0
    ability_reveal = 0.0
    tera_reveal = 0.0
    switch_reveal = 0.0
    for future in future_window:
        future_opp = future.observation.opp_side
        future_moves = set(future_opp.revealed_moves.get(active, []))
        move_reveal = max(move_reveal, float(len(future_moves - current_moves) > 0))
        item_reveal = max(item_reveal, float(active not in current_opp.revealed_items and active in future_opp.revealed_items))
        ability_reveal = max(ability_reveal, float(active not in current_opp.revealed_abilities and active in future_opp.revealed_abilities))
        tera_reveal = max(tera_reveal, float(active not in current_opp.revealed_tera_types and active in future_opp.revealed_tera_types))
        switch_reveal = max(switch_reveal, float(bool(future_opp.active_species) and future_opp.active_species != active))
    return RevealLikelihoodTarget(
        move=move_reveal,
        item=item_reveal,
        ability=ability_reveal,
        tera=tera_reveal,
        switch=switch_reveal,
    )


def _typed_consequence(current: DecisionSample, future: DecisionSample | None) -> TypedConsequenceTarget:
    if future is None:
        return TypedConsequenceTarget()
    current_obs = current.observation
    future_obs = future.observation
    current_self_hazards = sum(current_obs.self_side.hazards.values())
    future_self_hazards = sum(future_obs.self_side.hazards.values())
    current_opp_hazards = sum(current_obs.opp_side.hazards.values())
    future_opp_hazards = sum(future_obs.opp_side.hazards.values())
    ko_swing = (future_obs.opp_side.fainted_count - current_obs.opp_side.fainted_count) - (
        future_obs.self_side.fainted_count - current_obs.self_side.fainted_count
    )
    tera_swing = (float(future_obs.opp_side.tera_used) - float(current_obs.opp_side.tera_used)) - (
        float(future_obs.self_side.tera_used) - float(current_obs.self_side.tera_used)
    )
    position_swing = float(future_obs.opp_side.active_species != current_obs.opp_side.active_species) - 0.5 * float(
        future_obs.self_side.active_species != current_obs.self_side.active_species
    )
    return TypedConsequenceTarget(
        hp_self=_bounded(future_obs.self_side.active_hp - current_obs.self_side.active_hp),
        hp_opp=_bounded(current_obs.opp_side.active_hp - future_obs.opp_side.active_hp),
        ko=_bounded(ko_swing, scale=2.0),
        hazard=_bounded((future_opp_hazards - current_opp_hazards) - (future_self_hazards - current_self_hazards), scale=4.0),
        speed=_bounded(future.resource_ledger.speed - current.resource_ledger.speed),
        resource=_bounded(future.resource_ledger.scalar_value() - current.resource_ledger.scalar_value(), scale=2.0),
        info=_bounded(current.particle_posterior.uncertainty() - future.particle_posterior.uncertainty()),
        tera=_bounded(tera_swing),
        position=_bounded(position_swing),
        unlock=_bounded(future.unlock_target - current.unlock_target),
    )


def build_samples_from_replay(parsed: ParsedReplay, perspectives: Sequence[PlayerSide] = (PlayerSide.P1, PlayerSide.P2)) -> list[DecisionSample]:
    tracker = _TraceState()
    side_samples = {PlayerSide.P1: [], PlayerSide.P2: []}
    all_samples: list[DecisionSample] = []
    if parsed.preamble:
        tracker.apply_turn(TurnWindow(turn=0, messages=parsed.preamble), {})
    for window in parsed.turns:
        actions = tracker.extract_turn_actions(window)
        for perspective in perspectives:
            our_action = actions.get(perspective)
            if our_action is None:
                continue
            observation = tracker.snapshot(parsed, window.turn, perspective, window.messages)
            semantic = _semantic_intent_for_action(our_action, observation)
            unlock = _unlock_target(observation)
            plan_posterior = _plan_posterior(observation, semantic, unlock)
            phase_posterior = _phase_posterior(observation, semantic, unlock)
            plan = _argmax_label(plan_posterior, PLAN_LABEL_ORDER)
            phase = _argmax_label(phase_posterior, PHASE_LABEL_ORDER)
            belief_summary = _belief_summary(observation)
            resource_ledger = _resource_ledger(observation)
            legal_actions = tracker.legal_actions(perspective)
            visible_unique_moves = len(observation.self_side.revealed_moves.get(observation.self_side.active_species or "", []))
            hidden_move_slots = max(0, MAX_MOVE_SLOTS - visible_unique_moves)
            sample = DecisionSample(
                replay_id=parsed.meta.replay_id,
                battle_format=parsed.meta.battle_format,
                turn=window.turn,
                perspective=perspective,
                observation=observation,
                our_action=our_action,
                opponent_action=actions.get(perspective.opponent),
                family_target=_family_target(our_action),
                semantic_intent=semantic,
                plan_hypotheses=_top_plan_hypotheses(plan_posterior),
                plan_posterior=plan_posterior,
                plan_label=plan,
                phase_posterior=phase_posterior,
                phase_label=phase,
                unlock_target=unlock,
                response_cluster=_response_cluster(actions.get(perspective.opponent)),
                legal_actions=legal_actions,
                recoverable_actions=tracker.recoverable_actions(perspective),
                hidden_move_slots=hidden_move_slots,
                candidate_policy="honest-v4",
                particle_posterior=_particle_posterior(observation),
                belief_summary=belief_summary,
                resource_ledger=resource_ledger,
                belief_target=belief_summary.as_vector(),
                resource_target=resource_ledger.as_vector(),
            )
            side_samples[perspective].append(sample)
            all_samples.append(sample)
        tracker.apply_turn(window, actions)
    for samples in side_samples.values():
        for index, sample in enumerate(samples):
            future = samples[index + 1] if index + 1 < len(samples) else None
            sample.future_target = _future_summary(sample, future)
            sample.reveal_likelihood = _reveal_likelihood(sample, samples[index + 1 : index + 3])
            sample.typed_consequence = _typed_consequence(sample, future)
    return all_samples


def _iter_replay_payloads(input_path: Path) -> Iterator[Mapping[str, object]]:
    if input_path.is_dir():
        for replay_path in sorted(input_path.glob("*.json")):
            yield json.loads(replay_path.read_text(encoding="utf-8"))
        return

    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        return

    if input_path.suffix.lower() == ".json":
        yield json.loads(input_path.read_text(encoding="utf-8"))
        return

    raise ValueError(f"Unsupported replay input path: {input_path}")


def build_dataset_from_path(input_path: str | Path, output_path: str | Path, *, perspectives: Sequence[PlayerSide] = (PlayerSide.P1, PlayerSide.P2)) -> BuildStats:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = BuildStats()
    with output_path.open("w", encoding="utf-8") as fh:
        for payload in _iter_replay_payloads(input_path):
            stats.replays_seen += 1
            try:
                parsed = parse_replay_payload(payload)
                samples = build_samples_from_replay(parsed, perspectives=perspectives)
            except Exception:
                stats.replays_failed += 1
                continue
            if not samples:
                continue
            for sample in samples:
                fh.write(sample.model_dump_json())
                fh.write("\n")
            stats.replays_written += 1
            stats.samples_written += len(samples)
    return stats


def build_dataset_from_directory(input_dir: str | Path, output_path: str | Path, *, perspectives: Sequence[PlayerSide] = (PlayerSide.P1, PlayerSide.P2)) -> BuildStats:
    return build_dataset_from_path(input_dir, output_path, perspectives=perspectives)