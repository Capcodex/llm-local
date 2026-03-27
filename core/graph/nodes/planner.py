"""PLANNER node — analyses the mission prompt and decides routing.

Decision logic:
1. Search brain/skills/ directory for a matching skill (filename + frontmatter keywords).
2. If a matching skill is found  → decision = "use_existing_skill"
3. Otherwise                     → decision = "forge_new_tool"

The node also tries a lightweight vector-similarity search in ChromaDB when the
skills collection is available, but falls back gracefully if it is not.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.llms import Ollama

from core.graph.state import GraphState

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "brain" / "skills"

# Minimum relevance score to consider a ChromaDB result a hit
_SKILL_VECTOR_MIN_SCORE = 0.45

_PLANNER_PROMPT = """Tu es un agent d'analyse de missions.

Mission utilisateur :
"{prompt}"

Compétences disponibles (extraits) :
{skills_summary}

Réponds UNIQUEMENT avec un objet JSON valide (pas de commentaire, pas de markdown) :
{{
  "decision": "use_existing_skill" | "forge_new_tool",
  "skill_id": "<slug de la skill choisie ou null>",
  "rationale": "<explication courte>",
  "risk_level": "low" | "medium" | "high"
}}"""


def _load_skills_from_fs() -> list[dict]:
    """Return list of {slug, tool_slug, title, summary, path} from brain/skills/*.md."""
    skills: list[dict] = []
    if not SKILLS_DIR.exists():
        return skills
    for md_file in sorted(SKILLS_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        slug = md_file.stem
        title = slug
        summary = ""
        status = "active"
        tool_slug = slug  # fallback: same as file stem
        # Parse YAML frontmatter
        fm_match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        if fm_match:
            for line in fm_match.group(1).splitlines():
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip()
                elif line.startswith("summary:"):
                    summary = line.split(":", 1)[1].strip()
                elif line.startswith("status:"):
                    status = line.split(":", 1)[1].strip()
                elif line.startswith("tool_slug:"):
                    tool_slug = line.split(":", 1)[1].strip()
        if status not in ("active",):
            continue
        if not summary:
            # Grab first non-empty non-heading line as summary
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("---"):
                    summary = line[:200]
                    break
        skills.append(
            {
                "slug": slug,
                "tool_slug": tool_slug,
                "title": title,
                "summary": summary,
                "path": str(md_file),
            }
        )
    return skills


def _filter_disabled_tools(skills: list[dict], db) -> list[dict]:
    """T20 — remove skills whose associated tool is disabled or archived in DB."""
    if db is None:
        return skills
    try:
        from core.models import Tool

        active = []
        for s in skills:
            tool = db.query(Tool).filter(Tool.slug == s["tool_slug"]).first()
            if tool and tool.status in ("disabled", "archived"):
                continue
            active.append(s)
        return active
    except Exception:
        return skills


def _search_skills_vector(prompt: str) -> dict | None:
    """Try to find a matching skill via ChromaDB similarity search."""
    try:
        import chromadb

        client = chromadb.PersistentClient(
            path=str(
                Path(__file__).resolve().parent.parent.parent.parent
                / "data"
                / "vectorstore"
            )
        )
        collection = client.get_collection("skills")
        results = collection.query(query_texts=[prompt], n_results=1)
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        if distances and metadatas:
            # ChromaDB returns L2 distance; convert to a 0-1 similarity proxy
            score = 1.0 / (1.0 + float(distances[0]))
            if score >= _SKILL_VECTOR_MIN_SCORE:
                return metadatas[0]
    except Exception:
        pass
    return None


def _build_skills_summary(skills: list[dict]) -> str:
    if not skills:
        return "(aucune compétence disponible)"
    lines = []
    for s in skills[:10]:  # cap at 10 to stay within LLM context
        lines.append(f"- [{s['slug']}] {s['title']}: {s['summary']}")
    return "\n".join(lines)


def _parse_llm_json(raw: str) -> dict:
    """Extract the first JSON object from an LLM response."""
    raw = raw.strip()
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def planner_node(state: GraphState) -> GraphState:
    """PLANNER node implementation."""
    prompt = state.get("prompt", "")
    mission_id = state.get("mission_id", "")
    db = state.get("_db")

    # 1. Load skills from filesystem, then filter disabled tools (T20)
    skills = _load_skills_from_fs()
    skills = _filter_disabled_tools(skills, db)

    # 2. Quick vector search (best-effort)
    vector_hit = _search_skills_vector(prompt)
    vector_skill_slug = None
    if vector_hit:
        vector_skill_slug = vector_hit.get("slug") or vector_hit.get("tool_slug")

    # 3. Ask LLM to decide
    skills_summary = _build_skills_summary(skills)
    llm_prompt = _PLANNER_PROMPT.format(
        prompt=prompt, skills_summary=skills_summary
    )

    decision = "forge_new_tool"
    skill_id = None
    skill_path = None
    rationale = "Aucune skill correspondante trouvée."
    risk_level = "low"

    try:
        from langchain_community.llms import Ollama as _Ollama

        llm = _Ollama(model="deepseek-r1:8b")
        raw_response = llm.invoke(llm_prompt)
        parsed = _parse_llm_json(str(raw_response))

        decision = parsed.get("decision", decision)
        skill_id = parsed.get("skill_id") or None
        rationale = parsed.get("rationale", rationale)
        risk_level = parsed.get("risk_level", risk_level)
    except Exception as exc:
        # LLM unavailable — fall back to simple keyword matching
        rationale = f"LLM unavailable ({exc}); falling back to keyword match."
        if skills:
            prompt_lower = prompt.lower()
            for s in skills:
                keywords = (s["title"] + " " + s["summary"]).lower().split()
                if any(kw in prompt_lower for kw in keywords if len(kw) > 3):
                    decision = "use_existing_skill"
                    skill_id = s["slug"]
                    rationale = f"Keyword match on skill '{s['slug']}'."
                    break

    # Override with vector hit if LLM said forge but vector found something
    if decision == "forge_new_tool" and vector_skill_slug:
        decision = "use_existing_skill"
        skill_id = vector_skill_slug
        rationale = f"Vector search found skill '{vector_skill_slug}'."

    # Resolve skill path
    if decision == "use_existing_skill" and skill_id:
        candidate = SKILLS_DIR / f"{skill_id}.md"
        if candidate.exists():
            skill_path = str(candidate)
        else:
            # Skill referenced but not found on disk — must forge
            decision = "forge_new_tool"
            skill_path = None
            rationale = f"Skill '{skill_id}' not found on disk; will forge."
            skill_id = None

    return {
        **state,
        "decision": decision,
        "skill_id": skill_id,
        "skill_path": skill_path,
        "rationale": rationale,
        "risk_level": risk_level,
    }
