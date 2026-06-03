"""Replay protocol parsing for the clean v1 pipeline."""

from __future__ import annotations

from typing import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field


class ReplayMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replay_id: str
    battle_format: str
    players: tuple[str, ...] = ()
    rating: int | None = None
    upload_time: int | None = None
    winner: str | None = None


class TurnWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn: int
    messages: list[str] = Field(default_factory=list)


class ParsedReplay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta: ReplayMeta
    preamble: list[str] = Field(default_factory=list)
    turns: list[TurnWindow] = Field(default_factory=list)


def protocol_lines(log_text: str) -> list[str]:
    return [line.strip() for line in log_text.splitlines() if line.strip().startswith("|")]


def _winner_from_lines(lines: Iterable[str]) -> str | None:
    for line in lines:
        parts = line.split("|")
        if len(parts) > 2 and parts[1] == "win":
            return parts[2]
    return None


def split_turn_windows(log_text: str) -> tuple[list[str], list[TurnWindow]]:
    preamble: list[str] = []
    turns: list[TurnWindow] = []
    current_turn: int | None = None
    current_messages: list[str] = []

    def flush_current() -> None:
        nonlocal current_turn, current_messages
        if current_turn is not None:
            turns.append(TurnWindow(turn=current_turn, messages=list(current_messages)))
        current_turn = None
        current_messages = []

    for line in protocol_lines(log_text):
        parts = line.split("|")
        cmd = parts[1] if len(parts) > 1 else ""
        if cmd == "turn" and len(parts) > 2 and parts[2].isdigit():
            flush_current()
            current_turn = int(parts[2])
            current_messages = [line]
            continue
        if current_turn is None:
            preamble.append(line)
        else:
            current_messages.append(line)

    flush_current()
    return preamble, turns


def parse_replay_payload(payload: Mapping[str, object]) -> ParsedReplay:
    log_text = str(payload.get("log") or "")
    preamble, turns = split_turn_windows(log_text)
    lines = protocol_lines(log_text)
    rating = payload.get("rating")
    meta = ReplayMeta(
        replay_id=str(payload.get("id") or payload.get("replay_id") or "unknown-replay"),
        battle_format=str(payload.get("format") or "gen9ou"),
        players=tuple(str(player) for player in (payload.get("players") or [])),
        rating=int(rating) if isinstance(rating, int) else None,
        upload_time=int(payload["uploadtime"]) if payload.get("uploadtime") is not None else None,
        winner=_winner_from_lines(lines),
    )
    return ParsedReplay(meta=meta, preamble=preamble, turns=turns)