from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Mission(Base):
    __tablename__ = "missions"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    title = Column(String(255), nullable=True)
    prompt = Column(Text, nullable=False)
    context_path = Column(String(500), nullable=True)
    autonomy_level = Column(String(20), nullable=False, default="supervised")
    # pending | planning | forging | testing | executing | done | error
    status = Column(String(20), nullable=False, default="pending")
    risk_level = Column(String(50), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    steps = relationship("MissionStep", back_populates="mission", cascade="all, delete-orphan")
    tools = relationship("Tool", back_populates="created_by_mission")
    test_runs = relationship("ToolTestRun", back_populates="mission")


class MissionStep(Base):
    __tablename__ = "mission_steps"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    mission_id = Column(String(36), ForeignKey("missions.id"), nullable=False)
    # PLANNER | FORGE | TESTER | EXECUTOR | LOGGER | GOVERNOR | REGISTRY
    node_name = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    input_payload = Column(Text, nullable=True)   # JSON serialized
    output_payload = Column(Text, nullable=True)  # JSON serialized
    error_payload = Column(Text, nullable=True)   # JSON serialized
    started_at = Column(DateTime(timezone=True), nullable=False, default=_now)
    ended_at = Column(DateTime(timezone=True), nullable=True)

    mission = relationship("Mission", back_populates="steps")


class Tool(Base):
    __tablename__ = "tools"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, nullable=False)
    version = Column(String(20), nullable=False, default="1.0.0")
    description = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=True)
    # candidate | active | disabled | archived
    status = Column(String(20), nullable=False, default="candidate")
    created_by_mission_id = Column(String(36), ForeignKey("missions.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    created_by_mission = relationship("Mission", back_populates="tools")
    test_runs = relationship("ToolTestRun", back_populates="tool")
    registry_entries = relationship("RegistryEntry", back_populates="tool")


class Skill(Base):
    __tablename__ = "skills"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    tool_id = Column(String(36), ForeignKey("tools.id"), nullable=True)
    title = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, nullable=False)
    summary = Column(Text, nullable=True)
    markdown_path = Column(String(500), nullable=True)
    # candidate | active | disabled
    status = Column(String(20), nullable=False, default="candidate")
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    registry_entries = relationship("RegistryEntry", back_populates="skill")


class ToolTestRun(Base):
    __tablename__ = "tool_test_runs"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    tool_id = Column(String(36), ForeignKey("tools.id"), nullable=False)
    mission_id = Column(String(36), ForeignKey("missions.id"), nullable=True)
    attempt_number = Column(Integer, nullable=False, default=1)
    # success | failure | pending
    status = Column(String(20), nullable=False, default="pending")
    stdout = Column(Text, nullable=True)
    stderr = Column(Text, nullable=True)
    traceback = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)

    tool = relationship("Tool", back_populates="test_runs")
    mission = relationship("Mission", back_populates="test_runs")


class RegistryEntry(Base):
    __tablename__ = "registry_entries"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    tool_id = Column(String(36), ForeignKey("tools.id"), nullable=False)
    skill_id = Column(String(36), ForeignKey("skills.id"), nullable=True)
    published_version = Column(String(20), nullable=False, default="1.0.0")
    # pending | approved | rejected
    validation_status = Column(String(20), nullable=False, default="pending")
    approved_by = Column(String(255), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)

    tool = relationship("Tool", back_populates="registry_entries")
    skill = relationship("Skill", back_populates="registry_entries")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=_new_uuid)
    # system | human | agent
    actor_type = Column(String(50), nullable=False, default="system")
    action = Column(String(100), nullable=False)
    target_type = Column(String(50), nullable=True)
    target_id = Column(String(36), nullable=True)
    metadata_json = Column(Text, nullable=True)  # JSON serialized metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now)
