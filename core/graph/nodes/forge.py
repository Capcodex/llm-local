"""FORGE node — generates a Python tool and its Markdown skill card via LLM.

On first call (forge_attempt == 1):
  - Generate code + skill from the mission prompt.
  - Persist Tool (candidate) + Skill (candidate) in DB.

On retry (forge_attempt > 1):
  - Include the previous traceback in the LLM prompt for self-correction.
  - Overwrite the tool file on disk; update the DB record.

Both T12 (tool generation) and T13 (skill generation) live here.
"""
from __future__ import annotations

import re
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.graph.state import MAX_FORGE_ATTEMPTS, GraphState

TOOLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tools"
SKILLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "brain" / "skills"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_TOOL_PROMPT_FIRST = """\
Tu dois générer un module Python autonome pour réaliser la tâche suivante :

Mission : {prompt}
Slug de l'outil : {slug}
ID de l'outil : {tool_id}
Mission ID : {mission_id}

CONTRAINTES STRICTES :
1. La seule fonction exportée est `run(args: dict) -> dict`.
2. Pas d'accès réseau (n'importe quels imports `requests`, `urllib`, `httpx` sont interdits).
3. Pas d'écriture de fichiers en dehors du dossier `sandbox/`.
4. Gère toutes les erreurs en interne — retourne toujours un dict.
5. Le dict retourné contient au minimum la clé `"output"` ou `"error"`.
6. N'utilise que la bibliothèque standard Python (pas de pip install).

TEMPLATE OBLIGATOIRE (respecte exactement l'en-tête) :
# tool_{slug}.py
# id: {tool_id}
# version: 1.0.0
# created_by_mission: {mission_id}

\"\"\"
Description : <une ligne>
Usage       : run({{"arg1": "valeur"}})
Limites     : <liste des limitations>
\"\"\"

def run(args: dict) -> dict:
    \"\"\"Point d'entrée standard.\"\"\"
    ...

Réponds UNIQUEMENT avec le code Python brut — pas de markdown, pas de balises.
"""

_TOOL_PROMPT_RETRY = """\
Tu dois corriger le module Python ci-dessous qui a échoué lors des tests.

Mission : {prompt}
Slug de l'outil : {slug}
Tentative : {attempt} / {max_attempts}

ERREUR DE LA TENTATIVE PRÉCÉDENTE :
--- traceback ---
{traceback}
--- stdout ---
{stdout}
--- stderr ---
{stderr}

CODE ACTUEL (à corriger) :
{current_code}

Applique les mêmes contraintes que la version initiale.
Réponds UNIQUEMENT avec le code Python corrigé — pas de markdown, pas de balises.
"""

_SKILL_PROMPT = """\
Génère un fichier Markdown de documentation pour l'outil suivant.

Mission : {prompt}
Slug de l'outil : {slug}
ID de la skill : {skill_id}

Le fichier DOIT commencer par ce frontmatter YAML exact :
---
id: {skill_id}
title: {title}
tool_slug: {slug}
status: candidate
created_at: {created_at}
---

Puis inclure ces sections (en français) :
# {title}
## Contexte d'usage
## Prérequis
## Mode d'appel
## Limites connues

Réponds UNIQUEMENT avec le contenu Markdown brut.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Convert a prompt to a filesystem-safe slug (max 40 chars)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = text.strip("_")
    return text[:40] or "tool"


def _extract_code(raw: str) -> str:
    """Strip markdown fences and return only the Python code."""
    raw = raw.strip()
    # Remove ```python ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:python)?\s*\n?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()


def _validate_tool_code(code: str) -> bool:
    """Minimal check: the code must define a `run` function."""
    return bool(re.search(r"^\s*def\s+run\s*\(", code, re.MULTILINE))


def _call_llm(prompt_text: str, model: str = "deepseek-r1:8b") -> str:
    from langchain_community.llms import Ollama
    llm = Ollama(model=model)
    return str(llm.invoke(prompt_text)).strip()


def _title_from_slug(slug: str) -> str:
    return slug.replace("_", " ").title()


# ---------------------------------------------------------------------------
# FORGE node
# ---------------------------------------------------------------------------


def forge_node(state: GraphState) -> GraphState:
    """FORGE node — generates / corrects a Python tool + Markdown skill."""
    mission_id = state.get("mission_id", "unknown")
    prompt = state.get("prompt", "")
    attempt = state.get("forge_attempt", 0) + 1
    max_attempts = state.get("max_forge_attempts", MAX_FORGE_ATTEMPTS)

    # ── Determine slug (stable across retries) ──────────────────────────
    tool_slug = state.get("tool_slug") or _slugify(prompt)
    tool_id_db = state.get("tool_id_db") or str(uuid.uuid4())

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    tool_path = TOOLS_DIR / f"tool_{tool_slug}.py"

    # ── Generate Python tool code ────────────────────────────────────────
    if attempt == 1:
        llm_prompt = _TOOL_PROMPT_FIRST.format(
            prompt=prompt,
            slug=tool_slug,
            tool_id=tool_id_db,
            mission_id=mission_id,
        )
    else:
        current_code = tool_path.read_text(encoding="utf-8") if tool_path.exists() else ""
        llm_prompt = _TOOL_PROMPT_RETRY.format(
            prompt=prompt,
            slug=tool_slug,
            attempt=attempt,
            max_attempts=max_attempts,
            traceback=state.get("test_traceback") or "(aucun)",
            stdout=state.get("test_stdout") or "(aucun)",
            stderr=state.get("test_stderr") or "(aucun)",
            current_code=current_code,
        )

    try:
        raw_code = _call_llm(llm_prompt)
        tool_code = _extract_code(raw_code)
    except Exception as exc:
        return {
            **state,
            "forge_attempt": attempt,
            "tool_slug": tool_slug,
            "tool_id_db": tool_id_db,
            "error": f"FORGE: LLM inaccessible lors de la génération du code — {exc}",
        }

    if not _validate_tool_code(tool_code):
        # LLM produced something that doesn't look like valid tool code —
        # count as an attempt so the loop can retry or give up.
        return {
            **state,
            "forge_attempt": attempt,
            "tool_slug": tool_slug,
            "tool_id_db": tool_id_db,
            "test_status": "failure",
            "test_traceback": "Le LLM n'a pas retourné un code valide (pas de def run(...)).",
            "test_stdout": "",
            "test_stderr": "",
            "error": None,  # not fatal yet — TESTER will decide
        }

    # Write tool file
    tool_path.write_text(tool_code, encoding="utf-8")

    # ── Persist Tool in DB ───────────────────────────────────────────────
    db = state.get("_db")  # injected by the graph wrapper when available
    if db is not None:
        try:
            _upsert_tool(
                db=db,
                tool_id=tool_id_db,
                slug=tool_slug,
                mission_id=mission_id,
                file_path=str(tool_path),
                prompt=prompt,
            )
        except Exception:
            pass

    # ── Generate Markdown skill (only on first attempt) ─────────────────
    forged_skill_id_db = state.get("forged_skill_id_db") or str(uuid.uuid4())
    forged_skill_path = state.get("forged_skill_path")

    if attempt == 1:
        title = _title_from_slug(tool_slug)
        created_at = datetime.now(timezone.utc).isoformat()
        skill_prompt = _SKILL_PROMPT.format(
            prompt=prompt,
            slug=tool_slug,
            skill_id=forged_skill_id_db,
            title=title,
            created_at=created_at,
        )
        try:
            skill_md = _call_llm(skill_prompt)
        except Exception:
            skill_md = _minimal_skill_md(
                skill_id=forged_skill_id_db,
                slug=tool_slug,
                title=title,
                prompt=prompt,
                created_at=created_at,
            )

        skill_path = SKILLS_DIR / f"skill_{tool_slug}.md"
        skill_path.write_text(skill_md, encoding="utf-8")
        forged_skill_path = str(skill_path)

        if db is not None:
            try:
                _upsert_skill(
                    db=db,
                    skill_id=forged_skill_id_db,
                    tool_id=tool_id_db,
                    slug=tool_slug,
                    title=title,
                    file_path=forged_skill_path,
                    prompt=prompt,
                )
            except Exception:
                pass

    return {
        **state,
        "forge_attempt": attempt,
        "tool_slug": tool_slug,
        "tool_id_db": tool_id_db,
        "forged_skill_id_db": forged_skill_id_db,
        "forged_skill_path": forged_skill_path,
        # skill_path used by EXECUTOR — point it to the forged skill
        "skill_path": forged_skill_path,
        "error": None,
        # Clear stale test results so TESTER starts fresh
        "test_status": None,
        "test_stdout": None,
        "test_stderr": None,
        "test_traceback": None,
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _upsert_tool(*, db, tool_id: str, slug: str, mission_id: str, file_path: str, prompt: str) -> None:
    from core.models import Tool

    existing = db.query(Tool).filter(Tool.id == tool_id).first()
    if existing:
        existing.file_path = file_path
        db.commit()
        return

    name = slug.replace("_", " ").title()
    tool = Tool(
        id=tool_id,
        name=name,
        slug=slug,
        version="1.0.0",
        description=prompt[:500],
        file_path=file_path,
        status="candidate",
        created_by_mission_id=mission_id if mission_id != "unknown" else None,
    )
    db.add(tool)
    db.commit()


def _upsert_skill(
    *, db, skill_id: str, tool_id: str, slug: str, title: str, file_path: str, prompt: str
) -> None:
    from core.models import Skill

    existing = db.query(Skill).filter(Skill.id == skill_id).first()
    if existing:
        return

    skill = Skill(
        id=skill_id,
        tool_id=tool_id,
        title=title,
        slug=slug,
        summary=prompt[:300],
        markdown_path=file_path,
        status="candidate",
    )
    db.add(skill)
    db.commit()


def _minimal_skill_md(
    *, skill_id: str, slug: str, title: str, prompt: str, created_at: str
) -> str:
    """Fallback skill Markdown when LLM is unavailable."""
    return textwrap.dedent(f"""\
        ---
        id: {skill_id}
        title: {title}
        tool_slug: {slug}
        status: candidate
        created_at: {created_at}
        ---
        # {title}

        ## Contexte d'usage
        {prompt}

        ## Prérequis
        Aucun prérequis spécifique.

        ## Mode d'appel
        ```python
        from tools.tool_{slug} import run
        result = run({{}})
        ```

        ## Limites connues
        Outil généré automatiquement — à valider.
    """)
