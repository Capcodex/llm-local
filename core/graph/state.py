from __future__ import annotations

from typing import Any, Literal, Optional
from typing_extensions import TypedDict

MAX_FORGE_ATTEMPTS = 3


class GraphState(TypedDict, total=False):
    """Shared state passed between all LangGraph nodes."""

    # ── Mission context ────────────────────────────────────────────────────
    mission_id: str
    prompt: str
    autonomy_level: Literal["restricted", "supervised", "extended"]

    # ── PLANNER output ────────────────────────────────────────────────────
    decision: Literal["use_existing_skill", "forge_new_tool"]
    skill_id: Optional[str]          # slug of an *existing* skill
    skill_path: Optional[str]        # filesystem path to the existing .md file
    rationale: str
    risk_level: str

    # ── FORGE state ───────────────────────────────────────────────────────
    tool_id_db: Optional[str]        # DB UUID of the Tool record
    tool_slug: Optional[str]         # filesystem slug (tool_{slug}.py)
    forged_skill_id_db: Optional[str]  # DB UUID of the forged Skill record
    forged_skill_path: Optional[str]   # path written to brain/skills/
    forge_attempt: int               # current attempt counter (1-based)
    max_forge_attempts: int          # ceiling before giving up

    # ── TESTER output ─────────────────────────────────────────────────────
    test_status: Optional[Literal["success", "failure"]]
    test_run_id: Optional[str]       # DB UUID of the ToolTestRun
    test_stdout: Optional[str]
    test_stderr: Optional[str]
    test_traceback: Optional[str]

    # ── EXECUTOR output ───────────────────────────────────────────────────
    result: Optional[Any]            # raw return value of run()
    result_text: Optional[str]       # human-readable version

    # ── Error tracking ────────────────────────────────────────────────────
    error: Optional[str]

    # ── Internal routing ─────────────────────────────────────────────────
    next_node: Optional[str]
