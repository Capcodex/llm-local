"""Tests Sprint 3 — Forge & Test Engine.

Tests are isolated: no LLM calls, no real subprocess, no DB.
All external dependencies are monkeypatched.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VALID_TOOL_CODE = """\
# tool_test_slug.py
# id: fake-id
# version: 1.0.0
# created_by_mission: fake-mission

\"\"\"
Description : Test tool
Usage       : run({})
Limites     : none
\"\"\"

def run(args: dict) -> dict:
    return {"output": "hello from test tool"}
"""

# ── FORGE node ─────────────────────────────────────────────────────────────

class TestForgeNode:
    def test_generates_tool_file_on_first_attempt(self, tmp_path, monkeypatch):
        from core.graph.nodes import forge as forge_mod

        monkeypatch.setattr(forge_mod, "TOOLS_DIR", tmp_path / "tools")
        monkeypatch.setattr(forge_mod, "SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr(forge_mod, "_call_llm", lambda prompt, **kw: VALID_TOOL_CODE)

        state = {
            "mission_id": "m1",
            "prompt": "calcule la somme de deux nombres",
            "autonomy_level": "supervised",
        }
        from core.graph.nodes.forge import forge_node
        new_state = forge_node(state)

        assert new_state.get("error") is None
        assert new_state.get("forge_attempt") == 1
        slug = new_state["tool_slug"]
        assert (tmp_path / "tools" / f"tool_{slug}.py").exists()

    def test_generates_skill_markdown_on_first_attempt(self, tmp_path, monkeypatch):
        from core.graph.nodes import forge as forge_mod

        monkeypatch.setattr(forge_mod, "TOOLS_DIR", tmp_path / "tools")
        monkeypatch.setattr(forge_mod, "SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr(forge_mod, "_call_llm", lambda prompt, **kw: VALID_TOOL_CODE)

        state = {"mission_id": "m1", "prompt": "affiche la date du jour"}
        from core.graph.nodes.forge import forge_node
        new_state = forge_node(state)

        skill_path = new_state.get("forged_skill_path")
        assert skill_path and Path(skill_path).exists()

    def test_uses_traceback_on_retry(self, tmp_path, monkeypatch):
        from core.graph.nodes import forge as forge_mod

        monkeypatch.setattr(forge_mod, "TOOLS_DIR", tmp_path / "tools")
        monkeypatch.setattr(forge_mod, "SKILLS_DIR", tmp_path / "skills")

        prompts_seen = []
        def _fake_llm(prompt, **kw):
            prompts_seen.append(prompt)
            return VALID_TOOL_CODE

        monkeypatch.setattr(forge_mod, "_call_llm", _fake_llm)

        slug = "retry_slug"
        tool_path = tmp_path / "tools" / f"tool_{slug}.py"
        tool_path.parent.mkdir(parents=True, exist_ok=True)
        tool_path.write_text(VALID_TOOL_CODE)

        state = {
            "mission_id": "m1",
            "prompt": "some task",
            "tool_slug": slug,
            "tool_id_db": str(uuid.uuid4()),
            "forge_attempt": 1,
            "test_status": "failure",
            "test_traceback": "NameError: name 'x' is not defined",
            "test_stdout": "",
            "test_stderr": "",
        }
        from core.graph.nodes.forge import forge_node
        new_state = forge_node(state)

        assert new_state["forge_attempt"] == 2
        # The retry prompt must include the traceback
        assert any("NameError" in p for p in prompts_seen)

    def test_invalid_llm_output_marks_test_failure(self, tmp_path, monkeypatch):
        from core.graph.nodes import forge as forge_mod

        monkeypatch.setattr(forge_mod, "TOOLS_DIR", tmp_path / "tools")
        monkeypatch.setattr(forge_mod, "SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr(forge_mod, "_call_llm", lambda *a, **kw: "pas du code Python valide")

        state = {"mission_id": "m1", "prompt": "test"}
        from core.graph.nodes.forge import forge_node
        new_state = forge_node(state)

        assert new_state.get("test_status") == "failure"
        assert new_state.get("error") is None  # not fatal — loop will retry


# ── TESTER node ────────────────────────────────────────────────────────────

class TestTesterNode:
    def _make_tool(self, tools_dir: Path, slug: str, code: str) -> Path:
        p = tools_dir / f"tool_{slug}.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
        return p

    def test_success_on_valid_tool(self, tmp_path, monkeypatch):
        from core.graph.nodes import tester as tester_mod

        monkeypatch.setattr(tester_mod, "TOOLS_DIR", tmp_path / "tools")
        monkeypatch.setattr(tester_mod, "SANDBOX_DIR", tmp_path / "sandbox")

        slug = "good_tool"
        self._make_tool(tmp_path / "tools", slug, VALID_TOOL_CODE)

        state = {
            "mission_id": "m1",
            "prompt": "test",
            "tool_slug": slug,
            "tool_id_db": str(uuid.uuid4()),
            "forge_attempt": 1,
        }
        from core.graph.nodes.tester import tester_node
        new_state = tester_node(state)

        assert new_state["test_status"] == "success"
        assert new_state.get("error") is None

    def test_failure_on_broken_tool(self, tmp_path, monkeypatch):
        from core.graph.nodes import tester as tester_mod

        monkeypatch.setattr(tester_mod, "TOOLS_DIR", tmp_path / "tools")
        monkeypatch.setattr(tester_mod, "SANDBOX_DIR", tmp_path / "sandbox")

        slug = "bad_tool"
        broken_code = "def run(args):\n    raise RuntimeError('intentional error')\n"
        self._make_tool(tmp_path / "tools", slug, broken_code)

        state = {
            "mission_id": "m1",
            "prompt": "test",
            "tool_slug": slug,
            "tool_id_db": str(uuid.uuid4()),
            "forge_attempt": 1,
        }
        from core.graph.nodes.tester import tester_node
        new_state = tester_node(state)

        assert new_state["test_status"] == "failure"
        assert new_state.get("test_traceback")

    def test_missing_tool_file_is_failure(self, tmp_path, monkeypatch):
        from core.graph.nodes import tester as tester_mod

        monkeypatch.setattr(tester_mod, "TOOLS_DIR", tmp_path / "tools")
        monkeypatch.setattr(tester_mod, "SANDBOX_DIR", tmp_path / "sandbox")

        state = {
            "mission_id": "m1",
            "prompt": "test",
            "tool_slug": "nonexistent",
            "tool_id_db": str(uuid.uuid4()),
            "forge_attempt": 1,
        }
        from core.graph.nodes.tester import tester_node
        new_state = tester_node(state)

        assert new_state["test_status"] == "failure"


# ── Graph routing (T15) ────────────────────────────────────────────────────

class TestGraphRouting:
    def test_route_after_tester_success(self):
        from core.graph.graph import _route_after_tester
        state = {"test_status": "success", "forge_attempt": 1, "max_forge_attempts": 3}
        assert _route_after_tester(state) == "executor"

    def test_route_after_tester_retry(self):
        from core.graph.graph import _route_after_tester
        state = {"test_status": "failure", "forge_attempt": 1, "max_forge_attempts": 3}
        assert _route_after_tester(state) == "forge"

    def test_route_after_tester_give_up(self):
        from core.graph.graph import _route_after_tester
        state = {"test_status": "failure", "forge_attempt": 3, "max_forge_attempts": 3}
        assert _route_after_tester(state) == "end_with_error"

    def test_route_after_planner_forge(self):
        from core.graph.graph import _route_after_planner
        state = {"decision": "forge_new_tool"}
        assert _route_after_planner(state) == "forge"

    def test_route_after_planner_executor(self):
        from core.graph.graph import _route_after_planner
        state = {"decision": "use_existing_skill"}
        assert _route_after_planner(state) == "executor"

    def test_graph_compiles_with_all_nodes(self):
        from core.graph.graph import build_graph
        g = build_graph()
        assert "forge" in g.nodes
        assert "tester" in g.nodes
        assert "executor" in g.nodes
        assert "planner" in g.nodes
