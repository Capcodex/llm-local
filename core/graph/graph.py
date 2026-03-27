"""The Living Kernel — LangGraph execution engine.

Graph topology (Sprint 5):

  START
    └─► PLANNER
          ├─ use_existing_skill ──► EXECUTOR ──► END
          └─ forge_new_tool ──► GOVERNOR_FORGE
                                    ├─ blocked ──► AWAITING_APPROVAL ──► END
                                    └─ pass    ──► FORGE ──► TESTER
                                                               ├─ retry   ──► FORGE  (correction loop)
                                                               ├─ give_up ──► END_ERROR ──► END
                                                               └─ success ──► REGISTRY ──► GOVERNOR_REGISTRY
                                                                                               ├─ blocked ──► AWAITING_APPROVAL ──► END
                                                                                               └─ pass    ──► EXECUTOR ──► END

Correction loop (T15):
  - TESTER failure  → back to FORGE with traceback injected in state
  - After MAX_FORGE_ATTEMPTS failures → END_ERROR

REGISTRY (T17):
  - Creates RegistryEntry (pending for supervised, approved for extended)
  - Increments version, activates Tool + Skill for extended
  - Indexes skill in ChromaDB (T19), updates brain/agent.md (T21)

GOVERNOR (T22):
  - governor_forge:    blocks restricted missions before FORGE
  - governor_registry: blocks restricted/supervised missions before EXECUTOR
  - extended autonomy passes both gates
"""
from __future__ import annotations

from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph

from core.graph.nodes.executor import executor_node
from core.graph.nodes.forge import forge_node
from core.graph.nodes.governor import governor_forge_node, governor_registry_node
from core.graph.nodes.logger import persist_mission_step, write_log_file
from core.graph.nodes.planner import planner_node
from core.graph.nodes.registry import registry_node
from core.graph.nodes.tester import tester_node
from core.graph.state import MAX_FORGE_ATTEMPTS, GraphState


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Mission status helper
# ---------------------------------------------------------------------------


def _set_mission_status(db, mission_id: str, status: str) -> None:
    if db is None or not mission_id or mission_id == "unknown":
        return
    try:
        from core.models import Mission as MissionModel
        mission = db.query(MissionModel).filter(MissionModel.id == mission_id).first()
        if mission:
            mission.status = status
            db.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Node wrapper — auto-logs every node transition
# ---------------------------------------------------------------------------


def _wrap_node(node_fn, node_name: str, db=None, mission_status: str | None = None):
    """Wrap a node function with logging, DB persistence and status update."""

    def _wrapped(state: GraphState) -> GraphState:
        mission_id = state.get("mission_id", "unknown")

        # Optionally advance the mission status before running
        if mission_status:
            _set_mission_status(db, mission_id, mission_status)

        started_at = _now()
        input_snapshot = {
            "prompt": state.get("prompt", ""),
            "decision": state.get("decision"),
            "skill_id": state.get("skill_id"),
            "tool_slug": state.get("tool_slug"),
            "forge_attempt": state.get("forge_attempt"),
            "autonomy_level": state.get("autonomy_level"),
        }

        # Inject db into state so node functions can use it directly
        new_state = node_fn({**state, "_db": db})

        # Remove the private _db key before returning
        new_state = {k: v for k, v in new_state.items() if k != "_db"}

        status = "error" if new_state.get("error") else "success"
        output_snapshot = {
            "decision": new_state.get("decision"),
            "skill_id": new_state.get("skill_id"),
            "tool_slug": new_state.get("tool_slug"),
            "forge_attempt": new_state.get("forge_attempt"),
            "test_status": new_state.get("test_status"),
            "rationale": new_state.get("rationale"),
            "risk_level": new_state.get("risk_level"),
            "result_text": new_state.get("result_text"),
            "blocked": new_state.get("blocked"),
            "requires_approval_for": new_state.get("requires_approval_for"),
            "error": new_state.get("error"),
        }
        error_snapshot = (
            {"error": new_state.get("error")} if new_state.get("error") else None
        )

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
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_planner(state: GraphState) -> str:
    if state.get("error"):
        return "end_with_error"
    if state.get("decision") == "use_existing_skill":
        return "executor"
    return "governor_forge"   # all forge paths go through GOVERNOR first


def _route_after_governor(state: GraphState) -> str:
    """Shared routing for both GOVERNOR nodes."""
    if state.get("blocked"):
        return "awaiting_approval"
    return "pass"


def _route_after_tester(state: GraphState) -> str:
    """T15 — decide whether to retry FORGE, proceed to REGISTRY, or give up."""
    if state.get("test_status") == "success":
        return "executor"   # mapped to "registry" in add_conditional_edges

    attempt = state.get("forge_attempt", 1)
    max_attempts = state.get("max_forge_attempts", MAX_FORGE_ATTEMPTS)

    if attempt < max_attempts:
        return "forge"   # retry with traceback in state

    return "end_with_error"


# ---------------------------------------------------------------------------
# Terminal nodes
# ---------------------------------------------------------------------------


def _end_with_error_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state
    attempt = state.get("forge_attempt", 0)
    max_attempts = state.get("max_forge_attempts", MAX_FORGE_ATTEMPTS)
    if attempt >= max_attempts:
        msg = (
            f"Forge échouée après {attempt} tentative(s) sur {max_attempts}. "
            f"Dernier traceback : {state.get('test_traceback') or 'inconnu'}"
        )
    else:
        msg = (
            "Aucune skill existante trouvée et aucun outil forgé avec succès. "
            f"Décision PLANNER : {state.get('decision')} — "
            f"Rationale : {state.get('rationale')}"
        )
    return {**state, "error": msg, "result_text": None}


def _awaiting_approval_node(state: GraphState) -> GraphState:
    """Terminal node when GOVERNOR blocks — mission waits for human approval."""
    phase = state.get("requires_approval_for", "unknown")
    return {
        **state,
        "result_text": (
            state.get("result_text")
            or f"Mission bloquée en attente d'approbation humaine (phase : {phase})."
        ),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(db=None) -> StateGraph:
    """Compile the LangGraph execution graph for Sprint 5."""
    planner      = _wrap_node(planner_node,          "PLANNER",          db=db, mission_status="planning")
    gov_forge    = _wrap_node(governor_forge_node,   "GOVERNOR_FORGE",   db=db)
    forge        = _wrap_node(forge_node,            "FORGE",            db=db, mission_status="forging")
    tester       = _wrap_node(tester_node,           "TESTER",           db=db, mission_status="testing")
    registry     = _wrap_node(registry_node,         "REGISTRY",         db=db)
    gov_registry = _wrap_node(governor_registry_node,"GOVERNOR_REGISTRY",db=db)
    executor     = _wrap_node(executor_node,         "EXECUTOR",         db=db, mission_status="executing")
    end_err      = _wrap_node(_end_with_error_node,  "END_ERROR",        db=db)
    awaiting     = _wrap_node(_awaiting_approval_node,"AWAITING_APPROVAL",db=db, mission_status="awaiting_approval")

    builder: StateGraph = StateGraph(GraphState)
    builder.add_node("planner",           planner)
    builder.add_node("governor_forge",    gov_forge)
    builder.add_node("forge",             forge)
    builder.add_node("tester",            tester)
    builder.add_node("registry",          registry)
    builder.add_node("governor_registry", gov_registry)
    builder.add_node("executor",          executor)
    builder.add_node("end_with_error",    end_err)
    builder.add_node("awaiting_approval", awaiting)

    builder.add_edge(START, "planner")

    builder.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "executor":      "executor",
            "governor_forge":"governor_forge",
            "end_with_error":"end_with_error",
        },
    )

    builder.add_conditional_edges(
        "governor_forge",
        _route_after_governor,
        {"pass": "forge", "awaiting_approval": "awaiting_approval"},
    )

    builder.add_edge("forge", "tester")

    builder.add_conditional_edges(
        "tester",
        _route_after_tester,
        # "executor" key → "registry" node (T17)
        {"executor": "registry", "forge": "forge", "end_with_error": "end_with_error"},
    )

    builder.add_edge("registry", "governor_registry")

    builder.add_conditional_edges(
        "governor_registry",
        _route_after_governor,
        {"pass": "executor", "awaiting_approval": "awaiting_approval"},
    )

    builder.add_edge("executor",          END)
    builder.add_edge("end_with_error",    END)
    builder.add_edge("awaiting_approval", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_mission(
    *,
    mission_id: str,
    prompt: str,
    autonomy_level: str = "supervised",
    db=None,
) -> dict:
    """Execute the full mission graph and return a result dict."""
    graph = build_graph(db=db)

    initial_state: GraphState = {
        "mission_id": mission_id,
        "prompt": prompt,
        "autonomy_level": autonomy_level,
        "max_forge_attempts": MAX_FORGE_ATTEMPTS,
    }

    final_state: GraphState = graph.invoke(initial_state)

    return {
        "mission_id": mission_id,
        "decision": final_state.get("decision"),
        "skill_id": final_state.get("skill_id"),
        "tool_slug": final_state.get("tool_slug"),
        "forge_attempt": final_state.get("forge_attempt"),
        "test_status": final_state.get("test_status"),
        "registry_entry_id": final_state.get("registry_entry_id"),
        "registered_version": final_state.get("registered_version"),
        "rationale": final_state.get("rationale"),
        "risk_level": final_state.get("risk_level"),
        "result_text": final_state.get("result_text"),
        "blocked": final_state.get("blocked", False),
        "requires_approval_for": final_state.get("requires_approval_for"),
        "error": final_state.get("error"),
    }
