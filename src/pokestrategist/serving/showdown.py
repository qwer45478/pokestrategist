from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from pokestrategist.data.hidden_move_prior import HiddenMovePrior
from pokestrategist.data.schema import (
    ActionHead,
    BattleObservation,
    DecisionSample,
    FieldSummary,
    FutureSummaryTarget,
    ParticlePosteriorTarget,
    PhaseLabel,
    PlanLabel,
    PlayerSide,
    ResourceLedgerTarget,
    ResponseCluster,
    RevealLikelihoodTarget,
    SemanticIntent,
    SideSummary,
    StructuredAction,
    TypedConsequenceTarget,
    BeliefSummaryTarget,
)
from pokestrategist.data.static_rules import load_static_rule_catalog
from pokestrategist.data.team_preview_prior import TeamPreviewPriorCatalog
from pokestrategist.data.usage_priors import load_usage_prior_catalog
from pokestrategist.models.decision_model import PokeStrategistDecisionConfig, build_model_for_checkpoint
from pokestrategist.training.dataset import collate_decision_batch, tensorize_decision_sample

PLAN_LABELS = tuple(label.value for label in PlanLabel)
PHASE_LABELS = tuple(label.value for label in PhaseLabel)
LINE_LABELS = tuple(label.value for label in SemanticIntent)
RESPONSE_LABELS = tuple(label.value for label in ResponseCluster)
USED_MOVE_PATTERN = re.compile(r"^(The opposing )?(?P<species>.+?) used (?P<move>.+?)!$", re.IGNORECASE)
SWITCH_PATTERN = re.compile(r"^(The opposing )?(?P<species>.+?) (?:switched in|went back to|was sent out)!$", re.IGNORECASE)

# Patterns that definitively signal Team Preview is over and the battle has begun.
# Showdown prints "Go! <Pokemon>!" for your lead and "<Player> sent out <Pokemon>!"
# for the opponent lead.  Chinese UI uses "去吧！" / "派出了".
_BATTLE_START_SENTINELS = re.compile(
    r"\bGo!\s+\S|sent out|去吧！|派出了",
    re.IGNORECASE,
)


def _detect_battle_start_from_log(lines: list[str]) -> bool:
    """Return True if the battle log contains sent-out messages indicating
    Team Preview has concluded and the first turn is about to begin."""
    if not lines:
        return False
    combined = " ".join(lines)
    return bool(_BATTLE_START_SENTINELS.search(combined))


def _resolve_is_team_preview(snapshot: ShowdownBattleSnapshot) -> bool:
    """Determine whether the snapshot is still in Team Preview phase.

    Primary signal: ``is_team_preview`` field populated by the browser extension.
    Fallback: inspect the recent battle log for sent-out markers (Go! / sent out)."""
    if snapshot.is_team_preview:
        return True
    # Fallback: if the extension didn't set the flag, check the battle log.
    if not snapshot.legal_moves and snapshot.legal_switches:
        if not _detect_battle_start_from_log(snapshot.recent_log):
            return True
    return False


class ShowdownActionSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    disabled: bool = False
    tooltip: str = ""


class ShowdownSideSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: str | None = None
    hp: str | None = None
    status: str | None = None
    revealed_moves: list[str] = Field(default_factory=list, alias="revealedMoves")
    item: str | None = None
    ability: str | None = None
    tera_type: str | None = Field(default=None, alias="teraType")


class ShowdownFieldSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    weather: str = "none"
    terrain: str = "none"
    trick_room: bool = Field(default=False, alias="trickRoom")


class ShowdownBattleSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    source: str = "showdown-dom"
    page_title: str | None = Field(default=None, alias="pageTitle")
    turn: int | str | None = None
    forced_switch: bool = Field(default=False, alias="forcedSwitch")
    is_team_preview: bool = Field(default=False, alias="isTeamPreview")
    self_side: ShowdownSideSnapshot = Field(alias="self")
    opponent_side: ShowdownSideSnapshot = Field(alias="opponent")
    legal_moves: list[ShowdownActionSnapshot] = Field(default_factory=list, alias="legalMoves")
    legal_switches: list[ShowdownActionSnapshot] = Field(default_factory=list, alias="legalSwitches")
    recent_log: list[str] = Field(default_factory=list, alias="recentLog")
    observed_at: str | None = Field(default=None, alias="observedAt")
    url: str | None = None
    field: ShowdownFieldSnapshot = Field(default_factory=ShowdownFieldSnapshot)
    format_id: str | None = Field(default=None, alias="formatId")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _parse_turn(turn: int | str | None) -> int:
    if isinstance(turn, int):
        return max(1, turn)
    if isinstance(turn, str):
        match = re.search(r"(\d+)", turn)
        if match:
            return max(1, int(match.group(1)))
    return 1


def _parse_hp_fraction(text: str | None) -> float:
    if not text:
        return 1.0
    normalized = text.strip().lower()
    if normalized.endswith("%"):
        try:
            return max(0.0, min(float(normalized[:-1]) / 100.0, 1.0))
        except ValueError:
            return 1.0
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", normalized)
    if match:
        current = float(match.group(1))
        total = max(float(match.group(2)), 1.0)
        return max(0.0, min(current / total, 1.0))
    if "fnt" in normalized:
        return 0.0
    return 1.0


def _clean_switch_species(label: str | None) -> str | None:
    if not label:
        return None
    cleaned = re.sub(r"\s+\d+%.*$", "", label).strip()
    cleaned = re.sub(r"\s+\([^)]*\)$", "", cleaned).strip()
    return cleaned or None


def _infer_move_family(move_name: str | None) -> str:
    catalog = load_static_rule_catalog()
    entry = catalog.move_entry(move_name)
    if not entry:
        return "utility"
    tags = {str(tag).lower() for tag in entry.get("tags", [])}
    if "pivot" in tags:
        return "pivot"
    if "recovery" in tags:
        return "recovery"
    if "setup" in tags or "boost" in tags:
        return "setup"
    if "hazard" in tags:
        return "hazard"
    if "protect" in tags:
        return "protect"
    if "status" in tags:
        return "status"
    if str(entry.get("category", "")).lower() == "status":
        return "utility"
    if float(entry.get("power") or 0.0) > 0.0:
        return "attack"
    return "utility"


def _battle_format(snapshot: ShowdownBattleSnapshot) -> str:
    if snapshot.format_id:
        return snapshot.format_id
    title = (snapshot.page_title or "").lower()
    if "gen 9" in title and "ou" in title:
        return "gen9ou"
    return "gen9ou"


def _replay_id(snapshot: ShowdownBattleSnapshot) -> str:
    if snapshot.url:
        return snapshot.url.rstrip("/").split("/")[-1] or "showdown-live"
    if snapshot.page_title:
        return snapshot.page_title.strip().replace(" ", "-").lower()
    return "showdown-live"


def _history_from_log(lines: list[str]) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    history_tokens: list[str] = []
    self_moves: dict[str, list[str]] = {}
    opp_moves: dict[str, list[str]] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("turn"):
            history_tokens.append(lowered)
            continue
        used_match = USED_MOVE_PATTERN.match(line)
        if used_match:
            move_name = used_match.group("move").strip()
            species = used_match.group("species").strip()
            is_opp = bool(used_match.group(1))
            owner = opp_moves if is_opp else self_moves
            owner.setdefault(species, [])
            if move_name not in owner[species]:
                owner[species].append(move_name)
            history_tokens.append(f"{'opp' if is_opp else 'self'}:move:{move_name.lower()}")
            continue
        switch_match = SWITCH_PATTERN.match(line)
        if switch_match:
            species = switch_match.group("species").strip()
            is_opp = bool(switch_match.group(1))
            history_tokens.append(f"{'opp' if is_opp else 'self'}:switch:{species.lower()}")
            continue
        history_tokens.append(lowered)
    return history_tokens[-16:], self_moves, opp_moves


def _build_legal_actions(snapshot: ShowdownBattleSnapshot) -> list[StructuredAction]:
    actions: list[StructuredAction] = []
    self_name = (snapshot.self_side.name or "").strip()
    for move in snapshot.legal_moves:
        if move.disabled or not move.label.strip():
            continue
        actions.append(
            StructuredAction(
                actor=PlayerSide.P1,
                head=ActionHead.MOVE,
                move_token=move.label.strip(),
                move_family=_infer_move_family(move.label),
            )
        )
    for index, switch in enumerate(snapshot.legal_switches, start=1):
        if switch.disabled or not switch.label.strip():
            continue
        lowered_label = switch.label.lower()
        if "(fainted)" in lowered_label or "(active)" in lowered_label:
            continue
        species = _clean_switch_species(switch.label)
        if species and self_name and species.lower() == self_name.lower():
            continue
        actions.append(
            StructuredAction(
                actor=PlayerSide.P1,
                head=ActionHead.SWITCH,
                move_family="switch",
                switch_slot=index,
                switch_species=species,
            )
        )
    return actions


def snapshot_to_decision_sample(snapshot: ShowdownBattleSnapshot) -> DecisionSample:
    legal_actions = _build_legal_actions(snapshot)
    if not legal_actions:
        raise ValueError("Snapshot did not contain any enabled legal moves or switches.")

    history_actions, self_log_moves, opp_log_moves = _history_from_log(snapshot.recent_log)
    self_name = snapshot.self_side.name or "Unknown"
    opp_name = snapshot.opponent_side.name or "Unknown"
    self_revealed_moves = list(snapshot.self_side.revealed_moves) or [action.move_token for action in legal_actions if action.head is ActionHead.MOVE and action.move_token]

    self_side = SideSummary(
        active_species=self_name,
        active_hp=_parse_hp_fraction(snapshot.self_side.hp),
        active_status=snapshot.self_side.status,
        tera_used=False,
        team_order=[self_name] + [action.switch_species for action in legal_actions if action.head is ActionHead.SWITCH and action.switch_species and action.switch_species != self_name],
        species_hp={self_name: _parse_hp_fraction(snapshot.self_side.hp)},
        species_status={self_name: snapshot.self_side.status},
        revealed_moves={self_name: self_revealed_moves},
        revealed_items={self_name: snapshot.self_side.item} if snapshot.self_side.item else {},
        revealed_abilities={self_name: snapshot.self_side.ability} if snapshot.self_side.ability else {},
        revealed_tera_types={self_name: snapshot.self_side.tera_type} if snapshot.self_side.tera_type else {},
    )
    opp_side = SideSummary(
        active_species=opp_name,
        active_hp=_parse_hp_fraction(snapshot.opponent_side.hp),
        active_status=snapshot.opponent_side.status,
        tera_used=False,
        team_order=[opp_name],
        species_hp={opp_name: _parse_hp_fraction(snapshot.opponent_side.hp)},
        species_status={opp_name: snapshot.opponent_side.status},
        revealed_moves={opp_name: list(snapshot.opponent_side.revealed_moves) or opp_log_moves.get(opp_name, [])},
        revealed_items={opp_name: snapshot.opponent_side.item} if snapshot.opponent_side.item else {},
        revealed_abilities={opp_name: snapshot.opponent_side.ability} if snapshot.opponent_side.ability else {},
        revealed_tera_types={opp_name: snapshot.opponent_side.tera_type} if snapshot.opponent_side.tera_type else {},
    )
    observation = BattleObservation(
        replay_id=_replay_id(snapshot),
        battle_format=_battle_format(snapshot),
        turn=0 if _resolve_is_team_preview(snapshot) else _parse_turn(snapshot.turn),
        perspective=PlayerSide.P1,
        self_side=self_side,
        opp_side=opp_side,
        field=FieldSummary(
            weather=snapshot.field.weather,
            terrain=snapshot.field.terrain,
            trick_room=snapshot.field.trick_room,
        ),
        history_actions=history_actions,
        protocol_window=snapshot.recent_log[-16:],
    )
    dummy_action = legal_actions[0]
    family_target = dummy_action.move_family or ("switch" if dummy_action.head is ActionHead.SWITCH else "utility")
    return DecisionSample(
        replay_id=observation.replay_id,
        battle_format=observation.battle_format,
        turn=observation.turn,
        perspective=PlayerSide.P1,
        observation=observation,
        our_action=dummy_action,
        family_target=family_target,
        semantic_intent=SemanticIntent.PRESERVE if snapshot.forced_switch else SemanticIntent.TEMPO,
        plan_label=PlanLabel.STABILIZE if snapshot.forced_switch else PlanLabel.PRESSURE,
        phase_label=PhaseLabel.RESOURCE if snapshot.forced_switch else PhaseLabel.TEMPO,
        unlock_target=0.0,
        response_cluster=ResponseCluster.HOLD,
        legal_actions=legal_actions,
        recoverable_actions=list(legal_actions),
        hidden_move_slots=0,
        candidate_policy="showdown-local-v1",
        future_target=FutureSummaryTarget(),
        particle_posterior=ParticlePosteriorTarget(),
        reveal_likelihood=RevealLikelihoodTarget(),
        typed_consequence=TypedConsequenceTarget(),
        belief_summary=BeliefSummaryTarget(),
        resource_ledger=ResourceLedgerTarget(),
    )


def _to_device(batch: dict[str, torch.Tensor | list[str]], device: torch.device) -> dict[str, torch.Tensor | list[str]]:
    moved: dict[str, torch.Tensor | list[str]] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _action_display(entry: dict[str, Any]) -> str:
    if entry.get("head") == "switch":
        return f"Switch -> {entry.get('switch_species') or 'Unknown'}"
    label = entry.get("move_token") or "Unknown move"
    if entry.get("tera"):
        return f"Tera {label}"
    return str(label)


def _reason_text(entry: dict[str, Any], metadata: dict[str, Any]) -> str:
    head = entry.get("head", "move")
    if head == "switch":
        species = entry.get("switch_species") or "???"
        return (
            f"线路={metadata['line_label']} 计划={metadata['plan_label']} 阶段={metadata['phase_label']} | "
            f"换上 {species} 以改善对位并获取后续回合主动权"
        )
    move = entry.get("move_token") or "???"
    family = entry.get("move_family") or "utility"
    return (
        f"线路={metadata['line_label']} 计划={metadata['plan_label']} 阶段={metadata['phase_label']} | "
        f"{move} (family={family}) — 预期对手回应={metadata.get('response_label','hold')}"
    )


def _structured_reason_chain(
    entry: dict[str, Any],
    metadata: dict[str, Any],
    action_index: int | None = None,
    outputs: dict[str, Any] | None = None,
    is_lead: bool = False,
    opp_team: list[str] | None = None,
    team_preview_prior: Any = None,
) -> dict[str, Any]:
    """Build a structured reason chain dict for rich UI rendering."""
    head = entry.get("head", "move")
    chain: dict[str, Any] = {
        "head": head,
        "line": metadata.get("line_label", "unknown"),
        "plan": metadata.get("plan_label", "unknown"),
        "phase": metadata.get("phase_label", "unknown"),
        "is_lead": is_lead,
    }

    if head == "switch":
        chain["action_summary"] = f"换上 {entry.get('switch_species') or '???'}"
        chain["family"] = "switch"
        species = entry.get("switch_species")
        # Lead-specific: add usage-based lead rate if available
        if is_lead and species and team_preview_prior is not None:
            try:
                templates = team_preview_prior.top_templates(species, limit=1)
                if templates:
                    chain["lead_hint"] = f"使用率 {templates[0].prior_prob:.1%}"
            except Exception:
                pass
        # Lead-specific: add opponent matchup summary
        if is_lead and opp_team:
            chain["opp_team"] = opp_team[:6]
    else:
        chain["action_summary"] = entry.get("move_token") or "???"
        chain["family"] = entry.get("move_family") or "utility"
        chain["tera"] = bool(entry.get("tera"))

    # Per-action response distribution
    if outputs is not None and action_index is not None:
        resp_probs = outputs.get("candidate_response_probs")
        if resp_probs is not None and resp_probs.dim() >= 2:
            probs = resp_probs[0, action_index].tolist()
            labels = ["attack", "防御换人", "进攻换人", "保护", "进攻", "强化"]
            chain["response"] = [
                {"label": labels[i] if i < len(labels) else f"r{i}", "prob": round(float(p), 3)}
                for i, p in enumerate(probs) if p > 0.05
            ][:3]

        # Consequence delta
        future = outputs.get("typed_consequence_values")
        if future is not None and future.dim() >= 2:
            vals = future[0, action_index].tolist()
            axes = ["HPself", "HPopp", "KO", "hazard", "speed", "resource", "info", "tera", "position", "unlock"]
            deltas = []
            for i, v in enumerate(vals):
                if i < len(axes) and abs(float(v)) > 0.1:
                    delta = "↑" if float(v) > 0 else "↓"
                    deltas.append(f"{axes[i]}{delta}")
            chain["consequence"] = deltas[:4] if deltas else ([] if is_lead else ["无明显后果变化"])

        # Plan/phase confidence
        plan_probs = outputs.get("plan_posterior")
        if plan_probs is not None:
            plan_labels = ["pressure", "stabilize", "preserve", "convert"]
            chain["plan_dist"] = [
                {"label": plan_labels[i] if i < len(plan_labels) else f"p{i}", "prob": round(float(p), 3)}
                for i, p in enumerate(plan_probs[0].tolist())
            ][:4]

    return chain


class PokeStrategistLocalPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
        hidden_candidate_topk: int | None = None,
        use_team_preview_prior: bool | None = None,
        use_usage_priors: bool | None = None,
        usage_data_dir: str | Path | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = _resolve_device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        trainer_config = checkpoint.get("trainer_config", {})
        self.hidden_move_prior = HiddenMovePrior.from_dict(checkpoint.get("hidden_move_prior"))
        self.hidden_candidate_topk = int(hidden_candidate_topk or trainer_config.get("hidden_candidate_topk", 4))
        if use_team_preview_prior is None:
            self.use_team_preview_prior = bool(trainer_config.get("use_team_preview_prior", True))
        else:
            self.use_team_preview_prior = bool(use_team_preview_prior)
        self.team_preview_prior = (
            TeamPreviewPriorCatalog.from_dict(checkpoint["team_preview_prior"])
            if checkpoint.get("team_preview_prior") and self.use_team_preview_prior
            else None
        )
        if use_usage_priors is None:
            self.use_usage_priors = bool(trainer_config.get("use_usage_priors", True))
        else:
            self.use_usage_priors = bool(use_usage_priors)
        resolved_usage_data_dir = usage_data_dir if usage_data_dir is not None else trainer_config.get("usage_data_dir")
        self.usage_prior_catalog = load_usage_prior_catalog(resolved_usage_data_dir) if self.use_usage_priors else None

        model_config_payload = dict(checkpoint["model_config"])
        if "use_static_rule_features" not in model_config_payload:
            model_config_payload["use_static_rule_features"] = False
        if "candidate_prior_feature_dim" not in model_config_payload:
            prior_weight = checkpoint["model_state"].get("prior_feature_proj.0.weight")
            model_config_payload["candidate_prior_feature_dim"] = int(prior_weight.shape[1]) if prior_weight is not None else 2
        if "preview_prior_feature_dim" not in model_config_payload:
            preview_weight = checkpoint["model_state"].get("preview_prior_mlp.0.weight")
            model_config_payload["preview_prior_feature_dim"] = int(preview_weight.shape[1]) if preview_weight is not None else TEAM_PREVIEW_FEATURE_DIM
        self.model_config = PokeStrategistDecisionConfig(**model_config_payload)
        self.model = build_model_for_checkpoint(self.model_config, checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def describe(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "hidden_candidate_topk": self.hidden_candidate_topk,
            "use_team_preview_prior": self.use_team_preview_prior,
            "use_usage_priors": self.use_usage_priors,
            "hidden_size": self.model_config.hidden_size,
            "max_legal_actions": self.model_config.max_legal_actions,
        }

    @torch.inference_mode()
    def predict_sample(self, sample: DecisionSample, *, topk: int = 3) -> dict[str, Any]:
        item = tensorize_decision_sample(
            sample,
            hidden_move_prior=self.hidden_move_prior,
            hidden_candidate_topk=self.hidden_candidate_topk,
            usage_prior_catalog=self.usage_prior_catalog,
            team_preview_prior=self.team_preview_prior,
            include_action_metadata=True,
        )
        action_entries = list(item.get("legal_action_entries", []))
        batch = _to_device(collate_decision_batch([item]), self.device)
        outputs = self.model(batch)  # type: ignore[arg-type]

        legal_mask = batch["legal_action_mask"][0].bool()  # type: ignore[index]
        scores = outputs["legal_action_scores"][0]
        ranking_probs = torch.softmax(scores.masked_fill(~legal_mask, -1e9), dim=-1)
        valid_count = int(legal_mask.sum().item())
        suggestion_count = max(1, min(topk, valid_count))
        top_indices = torch.topk(scores.masked_fill(~legal_mask, -1e9), k=suggestion_count).indices.tolist()

        plan_index = int(outputs["plan_posterior"][0].argmax().item())
        phase_index = int(outputs["phase_posterior"][0].argmax().item())
        line_index = int(outputs["line_posterior"][0].argmax().item())
        response_index = int(outputs["response_logits"][0].argmax().item())
        head_probs = torch.softmax(outputs["head_logits"][0], dim=-1)

        # Detect team preview: all actions are switches, no moves available, turn 0/1 or team preview
        all_switches = all(e.get("head") == "switch" for e in action_entries if e)
        is_lead_phase = (all_switches and len(action_entries) >= 3 and sample.turn <= 1) or (sample.turn <= 0)

        metadata = {
            "plan_label": PLAN_LABELS[plan_index],
            "phase_label": "team_preview" if is_lead_phase else PHASE_LABELS[phase_index],
            "line_label": LINE_LABELS[line_index],
            "response_label": RESPONSE_LABELS[response_index],
            "frontier_size": int(outputs["frontier_mask"][0].sum().item()),
            "legal_action_count": valid_count,
            "move_head_prob": float(head_probs[0].item()),
            "switch_head_prob": float(head_probs[1].item()),
            "turn": sample.turn,
            "battle_format": sample.battle_format,
            "is_lead_phase": is_lead_phase,
        }

        # Extract opponent team for lead analysis
        opp_team = list(sample.observation.opp_side.team_order) if sample.observation.opp_side.team_order else []

        suggestions = []
        for index in top_indices:
            entry = action_entries[index]
            chain = _structured_reason_chain(
                entry, metadata,
                action_index=index, outputs=outputs,
                is_lead=is_lead_phase,
                opp_team=opp_team,
                team_preview_prior=self.team_preview_prior,
            )
            suggestions.append(
                {
                    "label": _action_display(entry),
                    "confidence": float(ranking_probs[index].item()),
                    "reason": _reason_text(entry, metadata),
                    "reason_chain": chain,
                    "action": entry,
                }
            )

        return {
            "source": "pokestrategist-local",
            "metadata": metadata,
            "suggestions": suggestions,
        }

    def predict_snapshot(self, snapshot: ShowdownBattleSnapshot, *, topk: int = 3) -> dict[str, Any]:
        return self.predict_sample(snapshot_to_decision_sample(snapshot), topk=topk)

    def predict_snapshot_payload(self, payload: dict[str, Any], *, topk: int = 3) -> dict[str, Any]:
        snapshot_payload = payload.get("snapshot", payload)
        snapshot = ShowdownBattleSnapshot.model_validate(snapshot_payload)
        return self.predict_snapshot(snapshot, topk=topk)


def health_payload(predictor: PokeStrategistLocalPredictor) -> dict[str, Any]:
    return {"ok": True, "service": "pokestrategist-showdown", "model": predictor.describe()}


def suggestion_payload_from_json(predictor: PokeStrategistLocalPredictor, body: bytes, *, topk: int) -> dict[str, Any]:
    payload = json.loads(body.decode("utf-8") or "{}")
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    request_topk = int(payload.get("topk", topk)) if payload.get("topk") is not None else topk
    return predictor.predict_snapshot_payload(payload, topk=request_topk)