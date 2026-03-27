"""Shared audit-logging helper (T24 — Sprint 5).

All critical agent actions (forge, test, register, approve, reject) call
_log_audit() to create an AuditLog row.  Failures are silently ignored so
they never crash the pipeline.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Optional


def log_audit(
    *,
    db,
    action: str,
    actor_type: str = "agent",
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Persist one AuditLog row.  Always best-effort — never raises."""
    if db is None:
        return
    try:
        from core.models import AuditLog

        entry = AuditLog(
            id=str(uuid.uuid4()),
            actor_type=actor_type,
            action=action,
            target_type=target_type,
            target_id=target_id,
            metadata_json=json.dumps(metadata, default=str) if metadata else None,
        )
        db.add(entry)
        db.commit()
    except Exception:
        pass
