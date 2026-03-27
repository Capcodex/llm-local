"""REGISTRY node — registers a validated tool + skill after TESTER success.

Implements:
  T17 — create RegistryEntry, version the tool, set Tool.status = active
  T19 — upsert the skill into the ChromaDB "skills" collection
  T21 — append a summary line to brain/agent.md
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.graph.state import GraphState

AGENT_MD = Path(__file__).resolve().parent.parent.parent.parent / "brain" / "agent.md"
VECTORSTORE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "vectorstore"


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def _next_version(current: str) -> str:
    """Increment the patch segment: '1.0.0' → '1.0.1'."""
    parts = current.split(".")
    if len(parts) == 3:
        try:
            return f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
        except ValueError:
            pass
    return "1.0.0"


def _compute_new_version(db, tool_id: str) -> str:
    """Return the next published_version for this tool."""
    from core.models import RegistryEntry

    latest = (
        db.query(RegistryEntry)
        .filter(RegistryEntry.tool_id == tool_id)
        .order_by(RegistryEntry.published_at.desc())
        .first()
    )
    if latest and latest.published_version:
        return _next_version(latest.published_version)
    return "1.0.0"


# ---------------------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------------------


def _create_registry_entry(
    *,
    db,
    tool_id: str,
    skill_id: Optional[str],
    version: str,
    autonomy_level: str,
) -> str:
    from core.models import RegistryEntry

    # For 'extended' missions auto-approve; otherwise leave as pending (Sprint 5)
    validation_status = "approved" if autonomy_level == "extended" else "pending"
    published_at = datetime.now(timezone.utc) if validation_status == "approved" else None

    entry = RegistryEntry(
        id=str(uuid.uuid4()),
        tool_id=tool_id,
        skill_id=skill_id,
        published_version=version,
        validation_status=validation_status,
        published_at=published_at,
    )
    db.add(entry)
    db.commit()
    return entry.id


def _activate_tool(*, db, tool_id: str, version: str) -> None:
    from core.models import Tool

    tool = db.query(Tool).filter(Tool.id == tool_id).first()
    if tool:
        tool.status = "active"
        tool.version = version
        db.commit()


def _activate_skill(*, db, skill_id: str) -> None:
    from core.models import Skill

    skill = db.query(Skill).filter(Skill.id == skill_id).first()
    if skill:
        skill.status = "active"
        db.commit()


# ---------------------------------------------------------------------------
# T19 — ChromaDB indexing
# ---------------------------------------------------------------------------


def _index_skill_in_chromadb(
    *,
    skill_id: str,
    tool_slug: str,
    title: str,
    summary: str,
    skill_path: Optional[str],
) -> None:
    """Upsert the skill document into the ChromaDB 'skills' collection."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        collection = client.get_or_create_collection("skills")

        # Use the full Markdown as the document when available
        document = summary
        if skill_path:
            try:
                document = Path(skill_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass

        collection.upsert(
            documents=[document],
            metadatas=[
                {
                    "slug": tool_slug,
                    "tool_slug": tool_slug,
                    "title": title,
                    "skill_id": skill_id,
                }
            ],
            ids=[skill_id],
        )
    except Exception:
        pass  # ChromaDB unavailable — PLANNER will fall back to filesystem scan


# ---------------------------------------------------------------------------
# T21 — brain/agent.md update
# ---------------------------------------------------------------------------


def _update_agent_md(*, title: str, tool_slug: str, version: str) -> None:
    """Append one row to the Skills table in brain/agent.md."""
    try:
        AGENT_MD.parent.mkdir(parents=True, exist_ok=True)
        content = AGENT_MD.read_text(encoding="utf-8") if AGENT_MD.exists() else ""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        new_row = f"| {date_str} | {title} | `{tool_slug}` | {version} |"

        placeholder = "_(aucune skill enregistrée"
        if placeholder in content:
            table_header = (
                "\n| Date | Skill | Outil | Version |\n"
                "|------|-------|-------|---------|"
            )
            content = content.replace(
                next(
                    line
                    for line in content.splitlines()
                    if placeholder in line
                ),
                table_header + "\n" + new_row,
            )
        else:
            # Table already exists — append a row after the last | line
            lines = content.splitlines()
            last_table_idx = max(
                (i for i, l in enumerate(lines) if l.startswith("|")), default=None
            )
            if last_table_idx is not None:
                lines.insert(last_table_idx + 1, new_row)
                content = "\n".join(lines) + "\n"
            else:
                content += f"\n{new_row}\n"

        AGENT_MD.write_text(content, encoding="utf-8")
    except Exception:
        pass  # never let agent.md update crash the pipeline


# ---------------------------------------------------------------------------
# REGISTRY node
# ---------------------------------------------------------------------------


def registry_node(state: GraphState) -> GraphState:
    """Register the validated tool and skill, update ChromaDB and agent.md."""
    db = state.get("_db")
    tool_id_db = state.get("tool_id_db")
    forged_skill_id_db = state.get("forged_skill_id_db")
    tool_slug = state.get("tool_slug")
    autonomy_level = state.get("autonomy_level", "supervised")
    forged_skill_path = state.get("forged_skill_path")

    if not tool_id_db:
        # No tool was forged — nothing to register (existing skill path)
        return state

    registry_entry_id: Optional[str] = None

    if db is not None:
        try:
            version = _compute_new_version(db, tool_id_db)
            registry_entry_id = _create_registry_entry(
                db=db,
                tool_id=tool_id_db,
                skill_id=forged_skill_id_db,
                version=version,
                autonomy_level=autonomy_level,
            )
            _activate_tool(db=db, tool_id=tool_id_db, version=version)
            if forged_skill_id_db:
                _activate_skill(db=db, skill_id=forged_skill_id_db)
        except Exception as exc:
            return {**state, "error": f"REGISTRY: erreur DB — {exc}"}
    else:
        # No DB — compute version locally
        version = "1.0.0"

    # T19 — index skill in ChromaDB
    if tool_slug:
        from core.models import Skill as SkillModel

        title = tool_slug.replace("_", " ").title()
        summary = state.get("prompt", "")[:500]

        if db is not None and forged_skill_id_db:
            try:
                skill_row = (
                    db.query(SkillModel)
                    .filter(SkillModel.id == forged_skill_id_db)
                    .first()
                )
                if skill_row:
                    title = skill_row.title or title
                    summary = skill_row.summary or summary
            except Exception:
                pass

        _index_skill_in_chromadb(
            skill_id=forged_skill_id_db or tool_slug,
            tool_slug=tool_slug,
            title=title,
            summary=summary,
            skill_path=forged_skill_path,
        )

        # T21 — update brain/agent.md
        _update_agent_md(title=title, tool_slug=tool_slug, version=version)

    return {
        **state,
        "registry_entry_id": registry_entry_id,
        "registered_version": version,
        "error": None,
    }
