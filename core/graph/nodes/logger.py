"""LOGGER node — persists each graph transition as a MissionStep in the DB
and writes a structured log file under logs/{mission_id}.jsonl.

The LOGGER is called transparently by the graph builder after every node
transition (via a wrapper), so individual nodes don't need to call it directly.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.graph.state import GraphState

LOGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize(value: Any) -> str | None:
    """Safely JSON-serialize a value; return None on failure."""
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def persist_mission_step(
    *,
    db,
    mission_id: str,
    node_name: str,
    status: str,
    input_payload: Any = None,
    output_payload: Any = None,
    error_payload: Any = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> str:
    """Write a MissionStep row to the database and return the step id."""
    from core.models import MissionStep

    step = MissionStep(
        id=str(uuid.uuid4()),
        mission_id=mission_id,
        node_name=node_name,
        status=status,
        input_payload=_serialize(input_payload),
        output_payload=_serialize(output_payload),
        error_payload=_serialize(error_payload),
        started_at=started_at or _now(),
        ended_at=ended_at or _now(),
    )
    db.add(step)
    db.commit()
    return step.id


def write_log_file(
    *,
    mission_id: str,
    node_name: str,
    status: str,
    input_payload: Any = None,
    output_payload: Any = None,
    error_payload: Any = None,
) -> None:
    """Append a JSON log line to logs/{mission_id}.jsonl."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{mission_id}.jsonl"
    entry = {
        "ts": _now().isoformat(),
        "node": node_name,
        "status": status,
        "input": input_payload,
        "output": output_payload,
        "error": error_payload,
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def logger_node(state: GraphState, *, db=None) -> GraphState:
    """Standalone LOGGER node — logs whatever is in the current state.

    When used as a pass-through node in the graph, it records the full
    state snapshot and returns the state unchanged.
    """
    mission_id = state.get("mission_id", "unknown")
    node_name = "LOGGER"
    status = "error" if state.get("error") else "success"

    output_payload = {
        "decision": state.get("decision"),
        "skill_id": state.get("skill_id"),
        "rationale": state.get("rationale"),
        "risk_level": state.get("risk_level"),
        "result_text": state.get("result_text"),
        "error": state.get("error"),
    }

    write_log_file(
        mission_id=mission_id,
        node_name=node_name,
        status=status,
        input_payload={"prompt": state.get("prompt", "")},
        output_payload=output_payload,
        error_payload={"error": state.get("error")} if state.get("error") else None,
    )

    if db is not None:
        try:
            persist_mission_step(
                db=db,
                mission_id=mission_id,
                node_name=node_name,
                status=status,
                input_payload={"prompt": state.get("prompt", "")},
                output_payload=output_payload,
                error_payload={"error": state.get("error")} if state.get("error") else None,
            )
        except Exception:
            pass  # never let LOGGER crash the pipeline

    return state
