"""GOVERNOR node — enforces autonomy_level governance gates (T22 — Sprint 5).

Two gate variants:
  - governor_forge_node    : blocks BEFORE forge when autonomy_level == "restricted"
  - governor_registry_node : blocks BEFORE executor when autonomy_level in
                             ("restricted", "supervised") — fires after REGISTRY has
                             already created a pending RegistryEntry.

Decision table:
  autonomy_level | forge gate | registry gate
  ---------------|------------|---------------
  restricted     | BLOCK      | BLOCK
  supervised     | pass       | BLOCK
  extended       | pass       | pass
"""
from __future__ import annotations

from core.graph.state import GraphState


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------


def _block(state: GraphState, phase: str, msg: str) -> GraphState:
    return {
        **state,
        "blocked": True,
        "requires_approval_for": phase,
        "result_text": msg,
        "error": None,
    }


def _pass(state: GraphState) -> GraphState:
    return {**state, "blocked": False}


# ---------------------------------------------------------------------------
# Public node functions
# ---------------------------------------------------------------------------


def governor_forge_node(state: GraphState) -> GraphState:
    """Gate before FORGE — blocks 'restricted' missions."""
    autonomy = state.get("autonomy_level", "supervised")
    if autonomy == "restricted":
        return _block(
            state,
            "forge",
            (
                "Mission en attente de validation humaine avant la forge de l'outil. "
                "Approuvez via POST /missions/{id}/approve pour continuer."
            ),
        )
    return _pass(state)


def governor_registry_node(state: GraphState) -> GraphState:
    """Gate before EXECUTOR (after REGISTRY) — blocks 'restricted' and 'supervised'."""
    autonomy = state.get("autonomy_level", "supervised")
    if autonomy in ("restricted", "supervised"):
        return _block(
            state,
            "registry",
            (
                "Mission en attente de validation humaine avant l'activation de l'outil. "
                "Approuvez via POST /missions/{id}/approve pour continuer."
            ),
        )
    return _pass(state)
