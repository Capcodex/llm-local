from __future__ import annotations

from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """Shared state passed between all LangGraph nodes."""

    # ── Mission context ────────────────────────────────────────────────────
    mission_id: str
    prompt: str
    autonomy_level: Literal["restricted", "supervised", "extended"]

    # ── PLANNER output ────────────────────────────────────────────────────
    decision: Literal["use_existing_skill", "forge_new_tool"]
    skill_id: Optional[str]
    skill_path: Optional[str]       # filesystem path to the .md skill file
    rationale: str
    risk_level: str

    # ── EXECUTOR output ───────────────────────────────────────────────────
    result: Optional[Any]           # final answer returned to the user
    result_text: Optional[str]      # formatted string version

    # ── Error tracking ────────────────────────────────────────────────────
    error: Optional[str]

    # ── Internal routing ─────────────────────────────────────────────────
    # Next node override used by LOGGER / GOVERNOR
    next_node: Optional[str]
