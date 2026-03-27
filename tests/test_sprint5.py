"""Sprint 5 tests — GOVERNOR, sandbox security, AuditLog, approve/reject, rate limiting."""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.models import (
    AuditLog,
    Base,
    Mission,
    RegistryEntry,
    Skill,
    Tool,
    ToolTestRun,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_engine():
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture
def db():
    engine = _make_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture
def api_db():
    engine = _make_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture
def api_client(server_module, api_db):
    from fastapi.testclient import TestClient
    from core.database import get_db

    def _override():
        yield api_db

    server_module.app.dependency_overrides[get_db] = _override
    with TestClient(server_module.app) as c:
        yield c, api_db
    server_module.app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# T22 — GOVERNOR node
# ---------------------------------------------------------------------------


class TestGovernorForge:
    def test_restricted_is_blocked(self):
        from core.graph.nodes.governor import governor_forge_node

        state = {"autonomy_level": "restricted", "prompt": "test"}
        result = governor_forge_node(state)
        assert result["blocked"] is True
        assert result["requires_approval_for"] == "forge"

    def test_supervised_passes(self):
        from core.graph.nodes.governor import governor_forge_node

        state = {"autonomy_level": "supervised", "prompt": "test"}
        result = governor_forge_node(state)
        assert result["blocked"] is False

    def test_extended_passes(self):
        from core.graph.nodes.governor import governor_forge_node

        state = {"autonomy_level": "extended", "prompt": "test"}
        result = governor_forge_node(state)
        assert result["blocked"] is False

    def test_default_passes(self):
        """Missing autonomy_level defaults to supervised → passes forge gate."""
        from core.graph.nodes.governor import governor_forge_node

        result = governor_forge_node({"prompt": "test"})
        assert result["blocked"] is False


class TestGovernorRegistry:
    def test_restricted_is_blocked(self):
        from core.graph.nodes.governor import governor_registry_node

        state = {"autonomy_level": "restricted"}
        result = governor_registry_node(state)
        assert result["blocked"] is True
        assert result["requires_approval_for"] == "registry"

    def test_supervised_is_blocked(self):
        from core.graph.nodes.governor import governor_registry_node

        state = {"autonomy_level": "supervised"}
        result = governor_registry_node(state)
        assert result["blocked"] is True

    def test_extended_passes(self):
        from core.graph.nodes.governor import governor_registry_node

        state = {"autonomy_level": "extended"}
        result = governor_registry_node(state)
        assert result["blocked"] is False


class TestGraphGovernorRouting:
    def test_route_after_governor_blocked(self):
        from core.graph.graph import _route_after_governor

        assert _route_after_governor({"blocked": True}) == "awaiting_approval"

    def test_route_after_governor_not_blocked(self):
        from core.graph.graph import _route_after_governor

        assert _route_after_governor({"blocked": False}) == "pass"
        assert _route_after_governor({}) == "pass"

    def test_graph_has_governor_nodes(self):
        from core.graph.graph import build_graph

        g = build_graph()
        assert "governor_forge" in g.nodes
        assert "governor_registry" in g.nodes
        assert "awaiting_approval" in g.nodes

    def test_full_graph_restricted_blocked_at_forge(self):
        """restricted mission should result in blocked=True without hitting forge."""
        from core.graph.graph import run_mission

        with patch("core.graph.nodes.planner.planner_node") as mock_planner:
            mock_planner.return_value = {
                "decision": "forge_new_tool",
                "rationale": "test",
                "risk_level": "low",
            }
            result = run_mission(
                mission_id="m1",
                prompt="test",
                autonomy_level="restricted",
            )
        assert result["blocked"] is True
        assert result["requires_approval_for"] == "forge"
        assert result["error"] is None


# ---------------------------------------------------------------------------
# T25 — Sandbox security scan
# ---------------------------------------------------------------------------


class TestSecurityScan:
    def test_os_system_forbidden(self):
        from core.graph.nodes.tester import _security_scan

        code = "import os\ndef run(args):\n    os.system('rm -rf /')\n    return {}"
        assert _security_scan(code) is not None

    def test_subprocess_forbidden(self):
        from core.graph.nodes.tester import _security_scan

        code = "import subprocess\ndef run(args):\n    subprocess.run(['ls'])\n    return {}"
        assert _security_scan(code) is not None

    def test_shutil_rmtree_forbidden(self):
        from core.graph.nodes.tester import _security_scan

        code = "import shutil\ndef run(args):\n    shutil.rmtree('/tmp')\n    return {}"
        assert _security_scan(code) is not None

    def test_requests_import_forbidden(self):
        from core.graph.nodes.tester import _security_scan

        code = "import requests\ndef run(args):\n    return {}"
        assert _security_scan(code) is not None

    def test_urllib_import_forbidden(self):
        from core.graph.nodes.tester import _security_scan

        code = "import urllib.request\ndef run(args):\n    return {}"
        assert _security_scan(code) is not None

    def test_clean_code_passes(self):
        from core.graph.nodes.tester import _security_scan

        code = (
            "import json\n"
            "def run(args):\n"
            "    return {'output': args.get('prompt', '')}\n"
        )
        assert _security_scan(code) is None

    def test_tester_node_rejects_forbidden_code(self, tmp_path):
        """tester_node returns failure without executing forbidden code."""
        from core.graph.nodes.tester import tester_node
        import core.graph.nodes.tester as tester_mod

        tool_file = tmp_path / "tool_evil.py"
        tool_file.write_text(
            "import os\ndef run(args):\n    os.system('whoami')\n    return {}\n",
            encoding="utf-8",
        )

        original_tools_dir = tester_mod.TOOLS_DIR
        tester_mod.TOOLS_DIR = tmp_path
        try:
            state = {
                "tool_slug": "evil",
                "tool_id_db": None,
                "mission_id": "m-test",
                "forge_attempt": 1,
                "prompt": "test",
            }
            result = tester_node(state)
        finally:
            tester_mod.TOOLS_DIR = original_tools_dir

        assert result["test_status"] == "failure"
        assert "interdit" in (result.get("test_traceback") or "").lower()


# ---------------------------------------------------------------------------
# T24 — AuditLog helper
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_log_audit_creates_row(self, db):
        from core.graph.nodes.audit import log_audit

        log_audit(
            db=db,
            action="tool_forged",
            target_type="tool",
            target_id="tid-1",
            metadata={"slug": "my_tool"},
        )
        entries = db.query(AuditLog).all()
        assert len(entries) == 1
        assert entries[0].action == "tool_forged"
        assert entries[0].actor_type == "agent"

    def test_log_audit_no_db_is_noop(self):
        from core.graph.nodes.audit import log_audit

        # Must not raise
        log_audit(db=None, action="tool_forged")

    def test_log_audit_metadata_serialized(self, db):
        from core.graph.nodes.audit import log_audit
        import json

        log_audit(db=db, action="test_ok", metadata={"attempt": 1})
        entry = db.query(AuditLog).first()
        assert json.loads(entry.metadata_json)["attempt"] == 1


# ---------------------------------------------------------------------------
# T23 — Approve / Reject endpoints
# ---------------------------------------------------------------------------


class TestApproveEndpoint:
    def _create_awaiting_mission(self, db, phase: str = "registry"):
        mission = Mission(
            id=str(uuid.uuid4()),
            prompt="approve me",
            autonomy_level="supervised",
            status="awaiting_approval",
        )
        db.add(mission)
        db.commit()

        if phase == "registry":
            tool = Tool(
                id=str(uuid.uuid4()),
                name="T",
                slug=f"t_{uuid.uuid4().hex[:6]}",
                status="candidate",
                created_by_mission_id=mission.id,
            )
            db.add(tool)
            db.commit()
            entry = RegistryEntry(
                id=str(uuid.uuid4()),
                tool_id=tool.id,
                published_version="1.0.0",
                validation_status="pending",
            )
            db.add(entry)
            db.commit()
            return mission, tool, entry

        return mission, None, None

    def test_approve_registry_phase(self, api_client):
        client, db = api_client
        mission, tool, entry = self._create_awaiting_mission(db, "registry")

        resp = client.post(f"/missions/{mission.id}/approve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["registry_entry_id"] == entry.id

        db.refresh(mission)
        assert mission.status == "done"
        db.refresh(tool)
        assert tool.status == "active"
        db.refresh(entry)
        assert entry.validation_status == "approved"
        assert entry.published_at is not None

    def test_approve_forge_phase(self, api_client):
        """Mission awaiting approval without a tool → forge-blocked path."""
        client, db = api_client
        mission, _, _ = self._create_awaiting_mission(db, "forge")

        resp = client.post(f"/missions/{mission.id}/approve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"

        db.refresh(mission)
        assert mission.status == "pending"
        assert mission.autonomy_level == "extended"

    def test_approve_wrong_status_409(self, api_client):
        client, db = api_client
        mission = Mission(
            id=str(uuid.uuid4()), prompt="p", status="done"
        )
        db.add(mission)
        db.commit()

        resp = client.post(f"/missions/{mission.id}/approve")
        assert resp.status_code == 409

    def test_approve_not_found_404(self, api_client):
        client, db = api_client
        resp = client.post("/missions/nonexistent/approve")
        assert resp.status_code == 404

    def test_approve_logs_audit(self, api_client):
        client, db = api_client
        mission, tool, entry = self._create_awaiting_mission(db, "registry")

        client.post(f"/missions/{mission.id}/approve")

        entries = db.query(AuditLog).filter(
            AuditLog.action == "mission_approved_registry"
        ).all()
        assert len(entries) == 1


class TestRejectEndpoint:
    def _create_awaiting_mission_with_tool(self, db):
        mission = Mission(
            id=str(uuid.uuid4()),
            prompt="reject me",
            autonomy_level="supervised",
            status="awaiting_approval",
        )
        db.add(mission)
        db.commit()
        tool = Tool(
            id=str(uuid.uuid4()),
            name="T",
            slug=f"tr_{uuid.uuid4().hex[:6]}",
            status="candidate",
            created_by_mission_id=mission.id,
        )
        db.add(tool)
        db.commit()
        entry = RegistryEntry(
            id=str(uuid.uuid4()),
            tool_id=tool.id,
            published_version="1.0.0",
            validation_status="pending",
        )
        db.add(entry)
        db.commit()
        return mission, tool, entry

    def test_reject_sets_error_and_archives_tool(self, api_client):
        client, db = api_client
        mission, tool, entry = self._create_awaiting_mission_with_tool(db)

        resp = client.post(
            f"/missions/{mission.id}/reject",
            json={"motif": "Code dangereux"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"

        db.refresh(mission)
        assert mission.status == "error"
        db.refresh(tool)
        assert tool.status == "archived"
        db.refresh(entry)
        assert entry.validation_status == "rejected"

    def test_reject_wrong_status_409(self, api_client):
        client, db = api_client
        mission = Mission(id=str(uuid.uuid4()), prompt="p", status="done")
        db.add(mission)
        db.commit()
        resp = client.post(f"/missions/{mission.id}/reject", json={})
        assert resp.status_code == 409

    def test_reject_logs_audit(self, api_client):
        client, db = api_client
        mission, _, _ = self._create_awaiting_mission_with_tool(db)

        client.post(f"/missions/{mission.id}/reject", json={"motif": "test"})

        entries = db.query(AuditLog).filter(AuditLog.action == "mission_rejected").all()
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# T24 — GET /audit-logs endpoint
# ---------------------------------------------------------------------------


class TestAuditLogsEndpoint:
    def test_list_empty(self, api_client):
        client, db = api_client
        resp = client.get("/audit-logs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_list_with_filter(self, api_client):
        client, db = api_client
        db.add(AuditLog(id=str(uuid.uuid4()), action="tool_forged", target_type="tool"))
        db.add(AuditLog(id=str(uuid.uuid4()), action="tool_registered", target_type="tool"))
        db.commit()

        resp = client.get("/audit-logs?action=tool_forged")
        data = resp.json()
        assert data["total"] == 1
        assert data["audit_logs"][0]["action"] == "tool_forged"


# ---------------------------------------------------------------------------
# T26 — Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_create_mission_accepts_request(self, api_client):
        """Verify POST /missions still works with the rate limiter active."""
        client, db = api_client
        resp = client.post(
            "/missions",
            json={"prompt": "rate limit test", "autonomy_level": "supervised"},
        )
        assert resp.status_code == 201
