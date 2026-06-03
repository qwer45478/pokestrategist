"""Core schema for the v1 PokePilot decision pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PlayerSide(str, Enum):
    P1 = "p1"
    P2 = "p2"

    @property
    def opponent(self) -> "PlayerSide":
        return PlayerSide.P2 if self is PlayerSide.P1 else PlayerSide.P1


class ActionHead(str, Enum):
    MOVE = "move"
    SWITCH = "switch"
    TERA_MOVE = "tera-move"


class SemanticIntent(str, Enum):
    SAFE = "safe"
    PUNISH = "punish"
    SCOUT = "scout"
    TEMPO = "tempo"
    PRESERVE = "preserve"
    CONVERT = "convert"


class PhaseLabel(str, Enum):
    SCOUT = "scout"
    RESOURCE = "resource"
    TEMPO = "tempo"
    CONVERT_TRANSITION = "convert-transition"


class PlanLabel(str, Enum):
    PRESSURE = "pressure"
    STABILIZE = "stabilize"
    PRESERVE = "preserve"
    CONVERT = "convert"


class ResponseCluster(str, Enum):
    HOLD = "hold"
    DEFENSIVE_SWITCH = "defensive-switch"
    OFFENSIVE_SWITCH = "offensive-switch"
    PROTECTIVE_MOVE = "protective-move"
    AGGRESSIVE_MOVE = "aggressive-move"
    SETUP = "setup"


class PlanHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: PlanLabel
    weight: float = Field(default=0.0, ge=0.0, le=1.0)


def _hazard_defaults() -> dict[str, int]:
    return {"stealthrock": 0, "spikes": 0, "toxicspikes": 0, "stickyweb": 0}


def _screen_defaults() -> dict[str, bool]:
    return {"reflect": False, "lightscreen": False, "auroraveil": False, "safeguard": False}


class SideSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_species: str | None = None
    active_hp: float = 1.0
    active_status: str | None = None
    fainted_count: int = 0
    tera_used: bool = False
    team_order: list[str] = Field(default_factory=list)
    fainted_species: list[str] = Field(default_factory=list)
    species_hp: dict[str, float] = Field(default_factory=dict)
    species_status: dict[str, str | None] = Field(default_factory=dict)
    revealed_moves: dict[str, list[str]] = Field(default_factory=dict)
    revealed_items: dict[str, str] = Field(default_factory=dict)
    revealed_abilities: dict[str, str] = Field(default_factory=dict)
    revealed_tera_types: dict[str, str] = Field(default_factory=dict)
    hazards: dict[str, int] = Field(default_factory=_hazard_defaults)
    screens: dict[str, bool] = Field(default_factory=_screen_defaults)


class FieldSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weather: str = "none"
    terrain: str = "none"
    trick_room: bool = False


class StructuredAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actor: PlayerSide | None = None
    head: ActionHead
    move_token: str | None = None
    move_family: str | None = None
    tera: bool = False
    switch_slot: int | None = None
    switch_species: str | None = None
    candidate_source: str | None = None
    candidate_prior_prob: float = 0.0
    candidate_support_score: float = 0.0
    candidate_is_gold_injected: bool = False

    def key(self) -> str:
        return "|".join(
            [
                self.head.value,
                self.move_token or "",
                self.move_family or "",
                "1" if self.tera else "0",
                "" if self.switch_slot is None else str(self.switch_slot),
                self.switch_species or "",
            ]
        )


class BattleObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replay_id: str
    battle_format: str
    turn: int
    perspective: PlayerSide
    rating: int | None = None
    self_side: SideSummary
    opp_side: SideSummary
    field: FieldSummary
    history_actions: list[str] = Field(default_factory=list)
    protocol_window: list[str] = Field(default_factory=list)


class FutureSummaryTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta_plan: float = 0.0
    delta_belief: float = 0.0
    delta_resource: float = 0.0
    delta_phase: float = 0.0
    delta_unlock: float = 0.0
    safe_score: float = 0.0
    tempo_score: float = 0.0
    convert_score: float = 0.0

    def as_vector(self) -> list[float]:
        return [
            self.delta_plan,
            self.delta_belief,
            self.delta_resource,
            self.delta_phase,
            self.delta_unlock,
            self.safe_score,
            self.tempo_score,
            self.convert_score,
        ]

    def tail_vector(self) -> list[float]:
        return [
            min(self.delta_plan, 0.0),
            min(self.delta_belief, 0.0),
            min(self.delta_resource, 0.0),
            min(self.delta_phase, 0.0),
            min(self.delta_unlock, 0.0),
            min(self.safe_score, 0.0),
            min(self.tempo_score, 0.0),
            min(self.convert_score, 0.0),
        ]


class ParticlePosteriorTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team_seen: float = 0.0
    team_alive: float = 0.0
    moves_known: float = 0.0
    item_known: float = 0.0
    ability_known: float = 0.0
    tera_known: float = 0.0
    active_set_known: float = 0.0
    archetype_offense: float = 0.0
    archetype_balance: float = 0.0
    archetype_stall: float = 0.0

    def as_vector(self) -> list[float]:
        return [
            self.team_seen,
            self.team_alive,
            self.moves_known,
            self.item_known,
            self.ability_known,
            self.tera_known,
            self.active_set_known,
            self.archetype_offense,
            self.archetype_balance,
            self.archetype_stall,
        ]

    def uncertainty(self) -> float:
        known_axes = [self.moves_known, self.item_known, self.ability_known, self.tera_known, self.active_set_known]
        return sum(1.0 - value for value in known_axes) / len(known_axes)


class RevealLikelihoodTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    move: float = 0.0
    item: float = 0.0
    ability: float = 0.0
    tera: float = 0.0
    switch: float = 0.0

    def as_vector(self) -> list[float]:
        return [self.move, self.item, self.ability, self.tera, self.switch]


class TypedConsequenceTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hp_self: float = 0.0
    hp_opp: float = 0.0
    ko: float = 0.0
    hazard: float = 0.0
    speed: float = 0.0
    resource: float = 0.0
    info: float = 0.0
    tera: float = 0.0
    position: float = 0.0
    unlock: float = 0.0

    def as_vector(self) -> list[float]:
        return [
            self.hp_self,
            self.hp_opp,
            self.ko,
            self.hazard,
            self.speed,
            self.resource,
            self.info,
            self.tera,
            self.position,
            self.unlock,
        ]

    def interaction_vector(self) -> list[float]:
        return [
            self.hp_opp * self.ko,
            self.hazard * self.position,
            self.info * self.unlock,
            self.speed * self.resource,
            self.tera * self.position,
        ]

    def bin_indices(self) -> list[int]:
        thresholds = (-0.35, -0.1, 0.1, 0.35)
        bins: list[int] = []
        for value in self.as_vector():
            if value <= thresholds[0]:
                bins.append(0)
            elif value <= thresholds[1]:
                bins.append(1)
            elif value <= thresholds[2]:
                bins.append(2)
            elif value <= thresholds[3]:
                bins.append(3)
            else:
                bins.append(4)
        return bins


class BeliefSummaryTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: float = 0.0
    ability: float = 0.0
    speed: float = 0.0
    moves: float = 0.0
    role: float = 0.0
    tera: float = 0.0

    def as_vector(self) -> list[float]:
        return [self.item, self.ability, self.speed, self.moves, self.role, self.tera]

    def uncertainty(self) -> float:
        return sum(1.0 - value for value in self.as_vector()) / 6.0


class ResourceLedgerTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keep: float = 0.0
    hazard: float = 0.0
    speed: float = 0.0
    hp: float = 0.0
    tera: float = 0.0
    sac: float = 0.0

    def as_vector(self) -> list[float]:
        return [self.keep, self.hazard, self.speed, self.hp, self.tera, self.sac]

    def scalar_value(self) -> float:
        return 0.25 * self.keep + 0.15 * self.hazard + 0.15 * self.speed + 0.2 * self.hp + 0.1 * self.tera + 0.15 * self.sac


class DecisionSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replay_id: str
    battle_format: str
    turn: int
    perspective: PlayerSide
    observation: BattleObservation
    our_action: StructuredAction
    opponent_action: StructuredAction | None = None
    family_target: str
    semantic_intent: SemanticIntent
    plan_hypotheses: list[PlanHypothesis] = Field(default_factory=list)
    plan_posterior: list[float] = Field(default_factory=list)
    plan_label: PlanLabel
    phase_posterior: list[float] = Field(default_factory=list)
    phase_label: PhaseLabel
    unlock_target: float
    response_cluster: ResponseCluster
    legal_actions: list[StructuredAction] = Field(default_factory=list)
    recoverable_actions: list[StructuredAction] = Field(default_factory=list)
    hidden_move_slots: int = 0
    candidate_policy: str = "honest-v3"
    future_target: FutureSummaryTarget = Field(default_factory=FutureSummaryTarget)
    particle_posterior: ParticlePosteriorTarget = Field(default_factory=ParticlePosteriorTarget)
    reveal_likelihood: RevealLikelihoodTarget = Field(default_factory=RevealLikelihoodTarget)
    typed_consequence: TypedConsequenceTarget = Field(default_factory=TypedConsequenceTarget)
    belief_summary: BeliefSummaryTarget = Field(default_factory=BeliefSummaryTarget)
    resource_ledger: ResourceLedgerTarget = Field(default_factory=ResourceLedgerTarget)
    belief_target: list[float] = Field(default_factory=list)
    resource_target: list[float] = Field(default_factory=list)