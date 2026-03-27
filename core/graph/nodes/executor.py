"""EXECUTOR node — loads and runs an existing skill, returns a formatted result.

The skill file is a Markdown document with a frontmatter header that specifies
the associated tool slug.  The executor:
1. Reads the skill's frontmatter to find the tool slug.
2. Imports the tool Python module from tools/tool_{slug}.py.
3. Calls its run(args) function with the mission prompt as the sole argument.
4. Formats and stores the result in the shared state.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

from core.graph.state import GraphState

TOOLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tools"


def _get_tool_slug_from_skill(skill_path: str) -> str | None:
    """Extract tool_slug from skill Markdown frontmatter."""
    try:
        text = Path(skill_path).read_text(encoding="utf-8", errors="ignore")
        fm_match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        if fm_match:
            for line in fm_match.group(1).splitlines():
                if line.startswith("tool_slug:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _load_tool_module(slug: str):
    """Dynamically import tools/tool_{slug}.py and return the module."""
    tool_file = TOOLS_DIR / f"tool_{slug}.py"
    if not tool_file.exists():
        raise FileNotFoundError(f"Tool file not found: {tool_file}")

    module_name = f"tools.tool_{slug}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, tool_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {tool_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _format_result(result: Any) -> str:
    """Convert a tool result to a human-readable string."""
    if isinstance(result, dict):
        if "output" in result:
            return str(result["output"])
        if "result" in result:
            return str(result["result"])
        if "error" in result:
            return f"Erreur lors de l'exécution : {result['error']}"
        return str(result)
    return str(result)


def executor_node(state: GraphState) -> GraphState:
    """EXECUTOR node implementation."""
    skill_path = state.get("skill_path")
    skill_id = state.get("skill_id")
    prompt = state.get("prompt", "")

    if not skill_path:
        return {
            **state,
            "error": "EXECUTOR: aucun skill_path dans l'état — impossible d'exécuter.",
            "result": None,
            "result_text": None,
        }

    # 1. Get tool slug from skill frontmatter
    tool_slug = _get_tool_slug_from_skill(skill_path)
    if not tool_slug:
        return {
            **state,
            "error": f"EXECUTOR: impossible de lire tool_slug depuis {skill_path}.",
            "result": None,
            "result_text": None,
        }

    # 2. Load the tool module
    try:
        tool_module = _load_tool_module(tool_slug)
    except Exception as exc:
        return {
            **state,
            "error": f"EXECUTOR: chargement du tool '{tool_slug}' échoué — {exc}",
            "result": None,
            "result_text": None,
        }

    # 3. Execute run(args)
    run_fn = getattr(tool_module, "run", None)
    if not callable(run_fn):
        return {
            **state,
            "error": f"EXECUTOR: le tool '{tool_slug}' n'expose pas de fonction run().",
            "result": None,
            "result_text": None,
        }

    try:
        result = run_fn({"prompt": prompt, "mission_prompt": prompt})
    except Exception as exc:
        return {
            **state,
            "error": f"EXECUTOR: erreur durant run() du tool '{tool_slug}' — {exc}",
            "result": None,
            "result_text": None,
        }

    result_text = _format_result(result)
    return {
        **state,
        "result": result,
        "result_text": result_text,
        "error": None,
    }
