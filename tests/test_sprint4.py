"""Sprint 4 tests — REGISTRY node, /tools and /skills endpoints, T20 filter."""
from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.models import Base, Mission, RegistryEntry, Skill, Tool


# ---------------------------------------------------------------------------
# Shared in-memory DB fixture
# ---------------------------------------------------------------------------


def _make_engine():
    """In-memory SQLite engine with StaticPool so all connections share the same DB."""
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
    session = Session()
    yield session
    session.close()


def _make_mission(db, autonomy_level: str = "supervised") -> str:
    mission = Mission(
        id=str(uuid.uuid4()),
        prompt="test mission",
        autonomy_level=autonomy_level,
        status="testing",
    )
    db.add(mission)
    db.commit()
    return mission.id


def _make_tool(db, mission_id: str, slug: str = "test_tool", status: str = "candidate") -> Tool:
    tool = Tool(
        id=str(uuid.uuid4()),
        name="Test Tool",
        slug=slug,
        status=status,
        created_by_mission_id=mission_id,
    )
    db.add(tool)
    db.commit()
    return tool


def _make_skill(db, tool_id: str, slug: str = "test_skill", status: str = "candidate") -> Skill:
    skill = Skill(
        id=str(uuid.uuid4()),
        tool_id=tool_id,
        title="Test Skill",
        slug=slug,
        summary="A test skill",
        status=status,
    )
    db.add(skill)
    db.commit()
    return skill


# ---------------------------------------------------------------------------
# T17 — REGISTRY node: version, registration, activation
# ---------------------------------------------------------------------------


class TestRegistryNode:
    def test_next_version_from_no_entries(self):
        from core.graph.nodes.registry import _compute_new_version

        engine = _make_engine()
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        tool = Tool(id=str(uuid.uuid4()), name="T", slug="t", status="candidate")
        db.add(tool)
        db.commit()
        assert _compute_new_version(db, tool.id) == "1.0.0"
        db.close()

    def test_next_version_increments_patch(self):
        from core.graph.nodes.registry import _compute_new_version

        engine = _make_engine()
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        tool = Tool(id=str(uuid.uuid4()), name="T", slug="t2", status="candidate")
        db.add(tool)
        db.commit()
        entry = RegistryEntry(
            id=str(uuid.uuid4()),
            tool_id=tool.id,
            published_version="1.0.2",
            validation_status="approved",
        )
        db.add(entry)
        db.commit()
        assert _compute_new_version(db, tool.id) == "1.0.3"
        db.close()

    def test_registry_node_no_tool_id(self, db):
        """If no tool was forged, registry_node is a no-op."""
        from core.graph.nodes.registry import registry_node

        state = {"_db": db, "prompt": "test", "autonomy_level": "supervised"}
        result = registry_node(state)
        assert result is state  # unchanged

    def test_registry_node_creates_entry_and_activates(self, db, tmp_path):
        from core.graph.nodes.registry import registry_node

        mission_id = _make_mission(db, "extended")
        tool = _make_tool(db, mission_id)
        skill = _make_skill(db, tool.id)

        # Create a fake skill markdown file
        md_file = tmp_path / "skill_test_tool.md"
        md_file.write_text("# Test Skill\n\nA test skill.", encoding="utf-8")

        state = {
            "_db": db,
            "prompt": "do a test",
            "tool_id_db": tool.id,
            "tool_slug": "test_tool",
            "forged_skill_id_db": skill.id,
            "forged_skill_path": str(md_file),
            "autonomy_level": "extended",
        }

        with patch("core.graph.nodes.registry._index_skill_in_chromadb"), \
             patch("core.graph.nodes.registry._update_agent_md"):
            result = registry_node(state)

        assert "registry_entry_id" in result
        assert result["registered_version"] == "1.0.0"
        assert result.get("error") is None

        # Tool should now be active
        db.refresh(tool)
        assert tool.status == "active"
        assert tool.version == "1.0.0"

        # Skill should now be active
        db.refresh(skill)
        assert skill.status == "active"

        # RegistryEntry should be approved (extended autonomy)
        entry = db.query(RegistryEntry).filter(RegistryEntry.tool_id == tool.id).first()
        assert entry is not None
        assert entry.validation_status == "approved"
        assert entry.published_at is not None

    def test_registry_node_supervised_sets_pending(self, db, tmp_path):
        from core.graph.nodes.registry import registry_node

        mission_id = _make_mission(db, "supervised")
        tool = _make_tool(db, mission_id)

        state = {
            "_db": db,
            "prompt": "supervised test",
            "tool_id_db": tool.id,
            "tool_slug": "sv_tool",
            "forged_skill_id_db": None,
            "forged_skill_path": None,
            "autonomy_level": "supervised",
        }

        with patch("core.graph.nodes.registry._index_skill_in_chromadb"), \
             patch("core.graph.nodes.registry._update_agent_md"):
            result = registry_node(state)

        entry = db.query(RegistryEntry).filter(RegistryEntry.tool_id == tool.id).first()
        assert entry is not None
        assert entry.validation_status == "pending"
        assert entry.published_at is None


# ---------------------------------------------------------------------------
# T19 — ChromaDB indexing
# ---------------------------------------------------------------------------


class TestChromaDBIndexing:
    def test_index_skill_in_chromadb(self, tmp_path):
        from core.graph.nodes.registry import _index_skill_in_chromadb

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.PersistentClient", return_value=mock_client):
            _index_skill_in_chromadb(
                skill_id="skill-abc",
                tool_slug="my_tool",
                title="My Tool",
                summary="Does something useful",
                skill_path=None,
            )

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        assert call_kwargs.kwargs["ids"] == ["skill-abc"]
        assert call_kwargs.kwargs["metadatas"][0]["tool_slug"] == "my_tool"

    def test_index_uses_markdown_file_content(self, tmp_path):
        from core.graph.nodes.registry import _index_skill_in_chromadb

        skill_md = tmp_path / "skill.md"
        skill_md.write_text("# Full Markdown Content", encoding="utf-8")

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.PersistentClient", return_value=mock_client):
            _index_skill_in_chromadb(
                skill_id="s1",
                tool_slug="t1",
                title="T1",
                summary="fallback",
                skill_path=str(skill_md),
            )

        docs = mock_collection.upsert.call_args.kwargs["documents"]
        assert "Full Markdown Content" in docs[0]


# ---------------------------------------------------------------------------
# T20 — Planner: filter disabled/archived tools
# ---------------------------------------------------------------------------


class TestPlannerFilterDisabledTools:
    def test_disabled_tool_excluded(self, db):
        from core.graph.nodes.planner import _filter_disabled_tools

        mission_id = _make_mission(db)
        _make_tool(db, mission_id, slug="disabled_t", status="disabled")

        skills = [{"slug": "s1", "tool_slug": "disabled_t", "title": "T", "summary": "S"}]
        result = _filter_disabled_tools(skills, db)
        assert result == []

    def test_archived_tool_excluded(self, db):
        from core.graph.nodes.planner import _filter_disabled_tools

        mission_id = _make_mission(db)
        _make_tool(db, mission_id, slug="archived_t", status="archived")

        skills = [{"slug": "s2", "tool_slug": "archived_t", "title": "T", "summary": "S"}]
        result = _filter_disabled_tools(skills, db)
        assert result == []

    def test_active_tool_kept(self, db):
        from core.graph.nodes.planner import _filter_disabled_tools

        mission_id = _make_mission(db)
        _make_tool(db, mission_id, slug="active_t", status="active")

        skills = [{"slug": "s3", "tool_slug": "active_t", "title": "T", "summary": "S"}]
        result = _filter_disabled_tools(skills, db)
        assert len(result) == 1

    def test_unknown_tool_slug_kept(self, db):
        """Skills with no matching DB row are kept (don't block unknown tools)."""
        from core.graph.nodes.planner import _filter_disabled_tools

        skills = [{"slug": "s4", "tool_slug": "unknown_slug", "title": "T", "summary": "S"}]
        result = _filter_disabled_tools(skills, db)
        assert len(result) == 1

    def test_no_db_returns_all_skills(self):
        from core.graph.nodes.planner import _filter_disabled_tools

        skills = [{"slug": "s5", "tool_slug": "t5", "title": "T", "summary": "S"}]
        result = _filter_disabled_tools(skills, None)
        assert result == skills


# ---------------------------------------------------------------------------
# T21 — brain/agent.md update
# ---------------------------------------------------------------------------


class TestAgentMdUpdate:
    def test_creates_table_from_placeholder(self, tmp_path):
        from core.graph.nodes.registry import _update_agent_md
        import core.graph.nodes.registry as reg_module

        agent_md = tmp_path / "agent.md"
        agent_md.write_text(
            "## Compétences\n\n_(aucune skill enregistrée pour le moment)_\n",
            encoding="utf-8",
        )

        original = reg_module.AGENT_MD
        reg_module.AGENT_MD = agent_md
        try:
            _update_agent_md(title="My Tool", tool_slug="my_tool", version="1.0.0")
        finally:
            reg_module.AGENT_MD = original

        content = agent_md.read_text(encoding="utf-8")
        assert "my_tool" in content
        assert "1.0.0" in content

    def test_appends_row_to_existing_table(self, tmp_path):
        from core.graph.nodes.registry import _update_agent_md
        import core.graph.nodes.registry as reg_module

        agent_md = tmp_path / "agent.md"
        agent_md.write_text(
            "## Skills\n\n| Date | Skill | Outil | Version |\n|------|-------|-------|---------|\n| 2025-01-01 | Old | `old_tool` | 1.0.0 |\n",
            encoding="utf-8",
        )

        original = reg_module.AGENT_MD
        reg_module.AGENT_MD = agent_md
        try:
            _update_agent_md(title="New Tool", tool_slug="new_tool", version="1.0.1")
        finally:
            reg_module.AGENT_MD = original

        content = agent_md.read_text(encoding="utf-8")
        assert "new_tool" in content
        assert "old_tool" in content  # existing row preserved


# ---------------------------------------------------------------------------
# T18 — REST endpoints /tools and /skills
# ---------------------------------------------------------------------------


@pytest.fixture
def api_db():
    """In-memory DB used for API endpoint tests (StaticPool keeps one connection)."""
    engine = _make_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def api_client(server_module, api_db):
    from fastapi.testclient import TestClient
    from core.database import get_db

    def _override_db():
        yield api_db

    # Override the FastAPI dependency with the in-memory DB
    server_module.app.dependency_overrides[get_db] = _override_db

    with TestClient(server_module.app) as c:
        yield c, api_db

    server_module.app.dependency_overrides.clear()


class TestToolsEndpoints:
    def test_list_tools_empty(self, api_client):
        client, db = api_client
        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["tools"] == []

    def test_list_tools_with_filter(self, api_client):
        client, db = api_client
        mission = Mission(id=str(uuid.uuid4()), prompt="p", status="done")
        db.add(mission)
        active = Tool(id=str(uuid.uuid4()), name="A", slug="active_x", status="active",
                      created_by_mission_id=mission.id)
        candidate = Tool(id=str(uuid.uuid4()), name="B", slug="candidate_x", status="candidate",
                         created_by_mission_id=mission.id)
        db.add_all([active, candidate])
        db.commit()

        resp = client.get("/tools?status=active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["tools"][0]["slug"] == "active_x"

    def test_get_tool_not_found(self, api_client):
        client, db = api_client
        resp = client.get("/tools/nonexistent-id")
        assert resp.status_code == 404

    def test_get_tool_detail(self, api_client):
        client, db = api_client
        mission = Mission(id=str(uuid.uuid4()), prompt="p", status="done")
        db.add(mission)
        tool = Tool(id=str(uuid.uuid4()), name="MyTool", slug="my_t", status="candidate",
                    created_by_mission_id=mission.id)
        db.add(tool)
        db.commit()

        resp = client.get(f"/tools/{tool.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "my_t"
        assert "test_runs" in data
        assert "registry_entries" in data

    def test_patch_tool_status_valid_transition(self, api_client):
        client, db = api_client
        mission = Mission(id=str(uuid.uuid4()), prompt="p", status="done")
        db.add(mission)
        tool = Tool(id=str(uuid.uuid4()), name="T", slug="patch_t", status="active",
                    created_by_mission_id=mission.id)
        db.add(tool)
        db.commit()

        resp = client.patch(f"/tools/{tool.id}", json={"status": "disabled"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "disabled"

    def test_patch_tool_status_invalid_transition(self, api_client):
        client, db = api_client
        mission = Mission(id=str(uuid.uuid4()), prompt="p", status="done")
        db.add(mission)
        tool = Tool(id=str(uuid.uuid4()), name="T", slug="arch_t", status="archived",
                    created_by_mission_id=mission.id)
        db.add(tool)
        db.commit()

        resp = client.patch(f"/tools/{tool.id}", json={"status": "active"})
        assert resp.status_code == 409

    def test_patch_tool_not_found(self, api_client):
        client, db = api_client
        resp = client.patch("/tools/nope", json={"status": "disabled"})
        assert resp.status_code == 404


class TestSkillsEndpoints:
    def test_list_skills_empty(self, api_client):
        client, db = api_client
        resp = client.get("/skills")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["skills"] == []

    def test_list_skills_with_filter(self, api_client):
        client, db = api_client
        s1 = Skill(id=str(uuid.uuid4()), title="S1", slug="s1_x", status="active")
        s2 = Skill(id=str(uuid.uuid4()), title="S2", slug="s2_x", status="candidate")
        db.add_all([s1, s2])
        db.commit()

        resp = client.get("/skills?status=active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["skills"][0]["slug"] == "s1_x"

    def test_get_skill_not_found(self, api_client):
        client, db = api_client
        resp = client.get("/skills/nonexistent")
        assert resp.status_code == 404

    def test_get_skill_detail(self, api_client, tmp_path):
        client, db = api_client
        md = tmp_path / "skill.md"
        md.write_text("# My Skill\n\nContent here.", encoding="utf-8")
        skill = Skill(
            id=str(uuid.uuid4()),
            title="My Skill",
            slug="my_skill_x",
            summary="Does stuff",
            status="active",
            markdown_path=str(md),
        )
        db.add(skill)
        db.commit()

        resp = client.get(f"/skills/{skill.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "my_skill_x"
        assert "My Skill" in data.get("markdown_content", "")
