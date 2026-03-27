"""The Living Kernel — LangGraph execution engine.

Graph topology (Sprint 2):

  START → PLANNER → LOGGER → [EXECUTOR | error] → LOGGER → END

Routing:
  - PLANNER → EXECUTOR   if decision == "use_existing_skill"
  - PLANNER → END(error) if decision == "forge_new_tool"  (Sprint 3 adds FORGE)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from core.graph.nodes.executor import executor_node
from core.graph.nodes.logger import persist_mission_step, write_log_file
from core.graph.nodes.planner import planner_node
from core.graph.state import GraphState


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Node wrappers — each wrapper calls persist_mission_step + write_log_file
# ---------------------------------------------------------------------------


def _wrap_node(node_fn, node_name: str, db=None):
    """Return a wrapped version of node_fn that auto-logs its step to DB + file."""

    def _wrapped(state: GraphState) -> GraphState:
        started_at = _now()
        input_snapshot = {
            "prompt": state.get("prompt", ""),
            "decision": state.get("decision"),
            "skill_id": state.get("skill_id"),
        }

        new_state = node_fn(state)

        status = "error" if new_state.get("error") else "success"
        output_snapshot = {
            "decision": new_state.get("decision"),
            "skill_id": new_state.get("skill_id"),
            "rationale": new_state.get("rationale"),
            "risk_level": new_state.get("risk_level"),
            "result_text": new_state.get("result_text"),
            "error": new_state.get("error"),
        }
        error_snapshot = (
            {"error": new_state.get("error")} if new_state.get("error") else None
        )

        mission_id = new_state.get("mission_id", "unknown")
        write_log_file(
            mission_id=mission_id,
            node_name=node_name,
            status=status,
            input_payload=input_snapshot,
            output_payload=output_snapshot,
            error_payload=error_snapshot,
        )

        if db is not None:
            try:
                persist_mission_step(
                    db=db,
                    mission_id=mission_id,
                    node_name=node_name,
                    status=status,
                    input_payload=input_snapshot,
                    output_payload=output_snapshot,
                    error_payload=error_snapshot,
                    started_at=started_at,
                    ended_at=_now(),
                )
            except Exception:
                pass

        return new_state

    return _wrapped


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def _route_after_planner(state: GraphState) -> str:
    """Route based on PLANNER decision."""
    if state.get("error"):
        return "end_with_error"
    decision = state.get("decision")
    if decision == "use_existing_skill":
        return "executor"
    # forge_new_tool → Sprint 3 will add FORGE node; for now mark as error
    return "end_with_error"


def _end_with_error_node(state: GraphState) -> GraphState:
    """Terminal node when no skill is found and FORGE is not yet available."""
    if not state.get("error"):
        return {
            **state,
            "error": (
                "Aucune skill existante trouvée pour cette mission. "
                "La forge automatique sera disponible au Sprint 3."
            ),
            "result_text": None,
        }
    return state


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(db=None) -> StateGraph:
    """Build and compile the LangGraph execution graph.

    Args:
        db: Optional SQLAlchemy session. When provided, each node persists
            its MissionStep to the database.
    """
    planner = _wrap_node(planner_node, "PLANNER", db=db)
    executor = _wrap_node(executor_node, "EXECUTOR", db=db)
    end_error = _wrap_node(_end_with_error_node, "END_ERROR", db=db)

    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("end_with_error", end_error)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "executor": "executor",
            "end_with_error": "end_with_error",
        },
    )
    builder.add_edge("executor", END)
    builder.add_edge("end_with_error", END)

    return builder.compile()


def run_mission(
    *,
    mission_id: str,
    prompt: str,
    autonomy_level: str = "supervised",
    db=None,
) -> dict:
    """Execute the full mission graph and return a result dict.

    Returns:
        {
            "mission_id": str,
            "decision": str,
            "skill_id": str | None,
            "rationale": str,
            "risk_level": str,
            "result_text": str | None,
            "error": str | None,
        }
    """
    graph = build_graph(db=db)

    initial_state: GraphState = {
        "mission_id": mission_id,
        "prompt": prompt,
        "autonomy_level": autonomy_level,
    }

    final_state: GraphState = graph.invoke(initial_state)

    return {
        "mission_id": mission_id,
        "decision": final_state.get("decision"),
        "skill_id": final_state.get("skill_id"),
        "rationale": final_state.get("rationale"),
        "risk_level": final_state.get("risk_level"),
        "result_text": final_state.get("result_text"),
        "error": final_state.get("error"),
    }
