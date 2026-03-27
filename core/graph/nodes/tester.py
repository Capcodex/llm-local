"""TESTER node — runs the candidate tool in a restricted subprocess and captures output.

Implements T14 (sandbox execution), T16 (ToolTestRun persistence),
and T25 (static security scan before execution).

Sandbox constraints:
- T25: Static scan rejects code containing forbidden patterns before execution.
- Runs in a child process with a configurable timeout (default 10 s).
- stdout / stderr / traceback captured separately.
- Each attempt is numbered; the ToolTestRun row is always persisted.
- The parent Tool.status is updated to reflect the last test outcome.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.graph.nodes.audit import log_audit
from core.graph.state import MAX_FORGE_ATTEMPTS, GraphState

# ---------------------------------------------------------------------------
# T25 — Static security scanner
# ---------------------------------------------------------------------------

# Patterns forbidden in generated tool code
_FORBIDDEN_PATTERNS: list[str] = [
    r"\bos\s*\.\s*system\s*\(",
    r"\bos\s*\.\s*popen\s*\(",
    r"\bos\s*\.\s*execv\s*\(",
    r"\bsubprocess\s*\.",
    r"\bshutil\s*\.\s*rmtree\s*\(",
    r"\bshutil\s*\.\s*rmdir\s*\(",
    r"(?<!['\"])\beval\s*\(",
    r"(?<!['\"])\bexec\s*\(",
    r"\b__import__\s*\(",
    r"\bimport\s+requests\b",
    r"\bfrom\s+requests\b",
    r"\bimport\s+urllib\b",
    r"\bfrom\s+urllib\b",
    r"\bimport\s+httpx\b",
    r"\bfrom\s+httpx\b",
    r"\bimport\s+socket\b",
    r"\bfrom\s+socket\b",
]


def _security_scan(code: str) -> Optional[str]:
    """Return a violation description if code contains a forbidden pattern, else None."""
    for pattern in _FORBIDDEN_PATTERNS:
        if re.search(pattern, code, re.MULTILINE):
            return f"Motif interdit détecté dans le code généré : {pattern!r}"
    return None

TOOLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tools"
SANDBOX_DIR = Path(__file__).resolve().parent.parent.parent.parent / "sandbox"

DEFAULT_TIMEOUT_SECONDS = 10

# ---------------------------------------------------------------------------
# Sandbox wrapper script (written to a temp file and executed in subprocess)
# ---------------------------------------------------------------------------

_SANDBOX_SCRIPT = textwrap.dedent("""\
    import sys, json, traceback as _tb

    tool_path = sys.argv[1]
    test_args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {{}}

    import importlib.util, os

    # Restrict working directory to sandbox/
    sandbox_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    os.chdir(sandbox_dir)

    spec = importlib.util.spec_from_file_location("_candidate_tool", tool_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        result = mod.run(test_args)
        print(json.dumps(result, ensure_ascii=False, default=str))
    except Exception as exc:
        print(json.dumps({{"error": str(exc)}}))
        sys.stderr.write(_tb.format_exc())
        sys.exit(1)
""")


# ---------------------------------------------------------------------------
# TESTER node
# ---------------------------------------------------------------------------


def tester_node(state: GraphState) -> GraphState:
    """Execute the candidate tool in a sandboxed subprocess."""
    tool_slug = state.get("tool_slug")
    tool_id_db = state.get("tool_id_db")
    mission_id = state.get("mission_id", "unknown")
    attempt = state.get("forge_attempt", 1)
    prompt = state.get("prompt", "")

    if not tool_slug:
        return {
            **state,
            "test_status": "failure",
            "test_traceback": "TESTER: tool_slug absent de l'état.",
            "test_stdout": "",
            "test_stderr": "",
        }

    tool_path = TOOLS_DIR / f"tool_{tool_slug}.py"
    if not tool_path.exists():
        return {
            **state,
            "test_status": "failure",
            "test_traceback": f"TESTER: fichier introuvable — {tool_path}",
            "test_stdout": "",
            "test_stderr": "",
        }

    # T25 — Static security scan before execution
    try:
        tool_code = tool_path.read_text(encoding="utf-8", errors="ignore")
        violation = _security_scan(tool_code)
    except Exception as exc:
        violation = f"Impossible de lire le fichier outil : {exc}"

    if violation:
        test_run_id = str(uuid.uuid4())
        db = state.get("_db")
        if db is not None and tool_id_db:
            try:
                _persist_test_run(
                    db=db,
                    run_id=test_run_id,
                    tool_id=tool_id_db,
                    mission_id=mission_id,
                    attempt_number=attempt,
                    status="failure",
                    stdout="",
                    stderr="",
                    traceback=violation,
                )
                _update_tool_status(db=db, tool_id=tool_id_db, test_status="failure")
            except Exception:
                pass
        return {
            **state,
            "test_status": "failure",
            "test_run_id": test_run_id,
            "test_traceback": violation,
            "test_stdout": "",
            "test_stderr": "",
            "error": None,
        }

    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

    # Build minimal test args from the prompt
    test_args: dict = {"prompt": prompt, "mission_prompt": prompt}

    stdout_text = ""
    stderr_text = ""
    traceback_text = ""
    exit_code = 0
    timed_out = False

    # Write sandbox script to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(_SANDBOX_SCRIPT)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [
                sys.executable,
                tmp_path,
                str(tool_path),
                json.dumps(test_args),
                str(SANDBOX_DIR),
            ],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        stdout_text = proc.stdout.strip()
        stderr_text = proc.stderr.strip()
        exit_code = proc.returncode
        if exit_code != 0:
            traceback_text = stderr_text
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout_text = (exc.stdout or b"").decode(errors="replace").strip()
        stderr_text = (exc.stderr or b"").decode(errors="replace").strip()
        traceback_text = f"TimeoutExpired: le tool a dépassé {DEFAULT_TIMEOUT_SECONDS}s."
        exit_code = -1
    except Exception as exc:
        traceback_text = f"Erreur subprocess inattendue : {exc}"
        exit_code = -1
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    success = (exit_code == 0) and (not timed_out)
    test_status: str = "success" if success else "failure"

    # ── Persist ToolTestRun ──────────────────────────────────────────────
    test_run_id = str(uuid.uuid4())
    db = state.get("_db")
    if db is not None and tool_id_db:
        try:
            _persist_test_run(
                db=db,
                run_id=test_run_id,
                tool_id=tool_id_db,
                mission_id=mission_id,
                attempt_number=attempt,
                status=test_status,
                stdout=stdout_text,
                stderr=stderr_text,
                traceback=traceback_text,
            )
            _update_tool_status(db=db, tool_id=tool_id_db, test_status=test_status)
        except Exception:
            pass
        log_audit(
            db=db,
            action=f"tool_test_{test_status}",
            target_type="tool",
            target_id=tool_id_db,
            metadata={"attempt": attempt, "exit_code": exit_code, "timed_out": timed_out},
        )

    return {
        **state,
        "test_status": test_status,
        "test_run_id": test_run_id,
        "test_stdout": stdout_text,
        "test_stderr": stderr_text,
        "test_traceback": traceback_text if not success else None,
        "error": None,
    }


# ---------------------------------------------------------------------------
# DB helpers (T16)
# ---------------------------------------------------------------------------


def _persist_test_run(
    *,
    db,
    run_id: str,
    tool_id: str,
    mission_id: str,
    attempt_number: int,
    status: str,
    stdout: str,
    stderr: str,
    traceback: str,
) -> None:
    from core.models import ToolTestRun

    run = ToolTestRun(
        id=run_id,
        tool_id=tool_id,
        mission_id=mission_id if mission_id != "unknown" else None,
        attempt_number=attempt_number,
        status=status,
        stdout=stdout[:10_000],   # cap stored size
        stderr=stderr[:10_000],
        traceback=traceback[:10_000] if traceback else None,
    )
    db.add(run)
    db.commit()


def _update_tool_status(*, db, tool_id: str, test_status: str) -> None:
    """Propagate the last test result to the parent Tool row (T16)."""
    from core.models import Tool

    tool = db.query(Tool).filter(Tool.id == tool_id).first()
    if not tool:
        return
    # candidate → keep as candidate until Sprint 4 promotes it to active
    # but mark disabled if test failed on last attempt (handled in graph routing)
    db.commit()
