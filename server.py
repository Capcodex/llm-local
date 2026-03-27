from __future__ import annotations

import json
import re
import traceback
import unicodedata
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Literal, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.database import create_tables, get_db
from core.models import Mission as MissionModel
from core.models import MissionStep as MissionStepModel

# Import the tools we created
from langchain_tools import TOOLS_REQUIRING_CONFIRMATION, create_note, get_current_time

# ---------------------------------------------------------------------------
# Context vars — replaces Flask's `g` for per-request state
# ---------------------------------------------------------------------------

_require_tool_confirmation: ContextVar[bool] = ContextVar(
    "require_tool_confirmation", default=False
)
_approved_tool_calls: ContextVar[list] = ContextVar(
    "approved_tool_calls", default=[]
)

# ---------------------------------------------------------------------------
# secure_filename replacement (no werkzeug dependency)
# ---------------------------------------------------------------------------


def secure_filename(filename: str) -> str:
    filename = str(filename).strip()
    filename = filename.replace("/", "_").replace("\\", "_")
    filename = re.sub(r"[^\w.\-]", "_", filename)
    filename = filename.lstrip(".")
    return filename or "file"


# ---------------------------------------------------------------------------
# Tool confirmation middleware
# ---------------------------------------------------------------------------


class ToolConfirmationRequired(Exception):
    def __init__(self, tool_name: str, tool_args: dict):
        super().__init__(f"Confirmation required for tool '{tool_name}'")
        self.tool_name = tool_name
        self.tool_args = tool_args


def is_tool_call_approved(tool_name: str, tool_args: dict) -> bool:
    approvals = _approved_tool_calls.get()
    for approval in approvals:
        if approval.get("name") == tool_name and approval.get("args") == tool_args:
            return True
    return False


def wrap_tool_with_confirmation(tool_obj):
    if tool_obj.name not in TOOLS_REQUIRING_CONFIRMATION:
        return tool_obj

    def _confirmed_tool(**kwargs):
        if _require_tool_confirmation.get() and not is_tool_call_approved(
            tool_obj.name, kwargs
        ):
            raise ToolConfirmationRequired(tool_obj.name, kwargs)
        return tool_obj.invoke(kwargs)

    return StructuredTool.from_function(
        func=_confirmed_tool,
        name=tool_obj.name,
        description=tool_obj.description,
        args_schema=tool_obj.args_schema,
        return_direct=tool_obj.return_direct,
    )


raw_tools = [get_current_time, create_note]
tools = [wrap_tool_with_confirmation(t) for t in raw_tools]
tool_by_name = {t.name: t for t in tools}

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

base_prompt = hub.pull("hwchase17/react")
system_instruction_template = (
    "Tu es un assistant IA conversationnel de type ChatGPT. "
    "Tu dois toujours repondre en francais avec un ton clair, naturel et professionnel. "
    "Outils disponibles: {tool_list}. "
    "Utilise les outils si cela permet d'etre plus precis ou si l'utilisateur le demande. "
    "Si une action modifie des fichiers, confirme d'abord (si exige) puis execute proprement. "
    "Ne fabrique pas de resultats d'outil. "
    "Quand la demande est ambigue, pose une question breve. "
    "Sinon, fournis directement une reponse structuree et utile."
)
DEFAULT_MODEL_NAME = "deepseek-r1:8b"
MAX_CONTEXT_MESSAGES = 12
PLAIN_CHAT_SYSTEM_INSTRUCTION = (
    "Tu es un assistant IA utile, clair et concis. "
    "Reponds toujours en francais. "
    "N'affiche jamais de balises techniques du type Thought/Action/Observation. "
    "Donne directement la reponse finale."
)

# ---------------------------------------------------------------------------
# RAG configuration
# ---------------------------------------------------------------------------

RAG_ENABLED = True
RAG_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
RAG_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
RAG_CHUNK_SIZE = 1200
RAG_CHUNK_OVERLAP = 200
RAG_COLLECTION_NAME = "local_docs"
RAG_BASE_DIR = Path(__file__).resolve().parent / "data"
RAG_UPLOAD_DIR = RAG_BASE_DIR / "uploads"
RAG_VECTORSTORE_DIR = RAG_BASE_DIR / "vectorstore"
RAG_REGISTRY_PATH = RAG_BASE_DIR / "documents_registry.json"
RAG_EMBEDDING_MODEL = "nomic-embed-text:latest"
RAG_DEFAULT_TOP_K = 4
RAG_MAX_TOP_K = 8
RAG_MIN_RELEVANCE_SCORE = 0.08
RAG_FALLBACK_CHUNKS_IF_NONE = 2
RAG_MAX_SOURCE_PREVIEW_CHARS = 280
RAG_MAX_CONTEXT_CHARS = 8000

_rag_vectorstore = None
_rag_embedding_model = None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="The Living Kernel", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    create_tables()


# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    enabled_tools: Optional[List[str]] = None
    messages: Optional[List[dict]] = None
    require_tool_confirmation: bool = False
    tool_approvals: Optional[List[dict]] = None
    model: Optional[str] = None
    use_rag: bool = False
    rag_top_k: Optional[int] = None
    top_k: Optional[int] = None


class RagChatRequest(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None
    model: Optional[str] = None
    top_k: Optional[int] = None


class MissionCreate(BaseModel):
    prompt: str
    title: Optional[str] = None
    context_path: Optional[str] = None
    autonomy_level: Literal["restricted", "supervised", "extended"] = "supervised"


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def build_prompt(selected_tools):
    tool_list = ", ".join([t.name for t in selected_tools]) or "aucun"
    return PromptTemplate(
        input_variables=base_prompt.input_variables,
        template=system_instruction_template + "\n\n" + base_prompt.template,
        partial_variables={
            **getattr(base_prompt, "partial_variables", {}),
            "tool_list": tool_list,
        },
    )


def get_agent_executor(selected_tools, model_name: str):
    llm = Ollama(model=model_name)
    prompt = build_prompt(selected_tools)
    agent = create_react_agent(llm, selected_tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=selected_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------


def ensure_rag_directories() -> None:
    RAG_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RAG_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_allowed_upload(filename: str) -> bool:
    return get_file_extension(filename) in RAG_ALLOWED_EXTENSIONS


def build_saved_filename(doc_id: str, original_filename: str) -> str:
    safe_name = secure_filename(original_filename) or f"document-{doc_id}.txt"
    return f"{doc_id}_{safe_name}"


def extract_text_from_pdf(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("Le package 'pypdf' est requis pour importer des PDF.") from exc

    reader = PdfReader(str(file_path))
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_text.append(page_text.strip())
    return "\n\n".join(pages_text).strip()


def extract_text_from_file(file_path: Path, extension: str) -> str:
    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    if extension in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore").strip()
    raise ValueError(f"Extension non supportee: {extension}")


def split_text_into_chunks(text: str) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:
        raise RuntimeError(
            "Le package 'langchain-text-splitters' est requis pour le decoupage des documents."
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
    )
    chunks = [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]
    return chunks


def get_rag_embedding_model():
    global _rag_embedding_model
    if _rag_embedding_model is None:
        try:
            from langchain_community.embeddings import OllamaEmbeddings
        except Exception as exc:
            raise RuntimeError(
                "Le package LangChain community embeddings est requis pour le mode RAG."
            ) from exc
        _rag_embedding_model = OllamaEmbeddings(model=RAG_EMBEDDING_MODEL)
    return _rag_embedding_model


def get_rag_vectorstore():
    global _rag_vectorstore
    if _rag_vectorstore is None:
        try:
            from langchain_community.vectorstores import Chroma
        except Exception as exc:
            raise RuntimeError(
                "Le package Chroma est requis pour le stockage vectoriel local."
            ) from exc

        ensure_rag_directories()
        _rag_vectorstore = Chroma(
            collection_name=RAG_COLLECTION_NAME,
            embedding_function=get_rag_embedding_model(),
            persist_directory=str(RAG_VECTORSTORE_DIR),
        )
    return _rag_vectorstore


def persist_vectorstore(vectorstore) -> None:
    persist_fn = getattr(vectorstore, "persist", None)
    if callable(persist_fn):
        persist_fn()


def build_upload_error(filename: str, message: str) -> dict:
    return {"filename": filename, "error": message}


def load_rag_registry() -> dict[str, dict]:
    ensure_rag_directories()
    if not RAG_REGISTRY_PATH.exists():
        return {}
    try:
        payload = json.loads(RAG_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    clean_registry = {}
    for doc_id, metadata in payload.items():
        if isinstance(doc_id, str) and isinstance(metadata, dict):
            clean_registry[doc_id] = metadata
    return clean_registry


def save_rag_registry(registry: dict[str, dict]) -> None:
    ensure_rag_directories()
    RAG_REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_rag_registry_entry(entry: dict) -> None:
    doc_id = str(entry.get("doc_id") or "").strip()
    if not doc_id:
        return
    registry = load_rag_registry()
    registry[doc_id] = entry
    save_rag_registry(registry)


def remove_rag_registry_entry(doc_id: str) -> dict | None:
    registry = load_rag_registry()
    removed = registry.pop(doc_id, None)
    save_rag_registry(registry)
    return removed


def infer_documents_from_vectorstore() -> dict[str, dict]:
    inferred: dict[str, dict] = {}
    try:
        vectorstore = get_rag_vectorstore()
        collection = getattr(vectorstore, "_collection", None)
        if collection is None:
            return inferred
        result = collection.get(include=["metadatas"])
    except Exception:
        return inferred

    metadatas = result.get("metadatas") or []
    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue
        doc_id = str(metadata.get("doc_id") or "").strip()
        if not doc_id:
            continue
        entry = inferred.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "filename": metadata.get("filename") or "document",
                "file_type": metadata.get("file_type") or "",
                "created_at": metadata.get("created_at") or "",
                "chunks": 0,
            },
        )
        entry["chunks"] = int(entry.get("chunks", 0)) + 1

    for doc_id, entry in inferred.items():
        matches = sorted(RAG_UPLOAD_DIR.glob(f"{doc_id}_*"))
        if matches:
            first_path = matches[0]
            entry["stored_filename"] = first_path.name
            try:
                entry["size_bytes"] = first_path.stat().st_size
            except Exception:
                pass
    return inferred


def list_rag_documents() -> list[dict]:
    registry = load_rag_registry()
    inferred = infer_documents_from_vectorstore()

    merged = dict(registry)
    for doc_id, inferred_entry in inferred.items():
        if doc_id not in merged:
            merged[doc_id] = inferred_entry
            continue
        existing = merged[doc_id]
        if not existing.get("chunks"):
            existing["chunks"] = inferred_entry.get("chunks", 0)
        if not existing.get("size_bytes") and inferred_entry.get("size_bytes"):
            existing["size_bytes"] = inferred_entry.get("size_bytes")
        if not existing.get("stored_filename") and inferred_entry.get("stored_filename"):
            existing["stored_filename"] = inferred_entry.get("stored_filename")

    if merged != registry:
        save_rag_registry(merged)

    documents = list(merged.values())
    documents.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return documents


def clamp_top_k(value: Any) -> int:
    try:
        parsed = int(value)
    except Exception:
        return RAG_DEFAULT_TOP_K
    if parsed < 1:
        return 1
    if parsed > RAG_MAX_TOP_K:
        return RAG_MAX_TOP_K
    return parsed


def build_rag_sources(chunks: list[dict]) -> list[dict]:
    sources = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        content = str(chunk.get("content") or "").strip()
        preview = content[: RAG_MAX_SOURCE_PREVIEW_CHARS]
        if len(content) > RAG_MAX_SOURCE_PREVIEW_CHARS:
            preview = f"{preview}..."
        sources.append(
            {
                "source_id": f"S{idx}",
                "doc_id": metadata.get("doc_id"),
                "filename": metadata.get("filename") or "document",
                "chunk_index": metadata.get("chunk_index"),
                "score": round(float(chunk.get("score", 0.0)), 4),
                "preview": preview,
            }
        )
    return sources


def retrieve_rag_chunks(query: str, top_k: int) -> list[dict]:
    vectorstore = get_rag_vectorstore()
    chunks: list[dict] = []
    low_score_chunks: list[dict] = []

    def _build_chunk(document, score_value: float) -> dict:
        return {
            "content": document.page_content or "",
            "metadata": document.metadata or {},
            "score": score_value,
        }

    def _split_by_threshold(rows: list[tuple]) -> None:
        for document, score_value in rows:
            chunk = _build_chunk(document, float(score_value))
            if not str(chunk["content"]).strip():
                continue
            if chunk["score"] >= RAG_MIN_RELEVANCE_SCORE:
                chunks.append(chunk)
            else:
                low_score_chunks.append(chunk)

    try:
        results = vectorstore.similarity_search_with_relevance_scores(query=query, k=top_k)
        _split_by_threshold([(document, score) for document, score in results])
    except Exception:
        results = vectorstore.similarity_search_with_score(query=query, k=top_k)
        proxy_rows = [
            (document, 1.0 / (1.0 + max(float(distance), 0.0)))
            for document, distance in results
        ]
        _split_by_threshold(proxy_rows)

    if chunks:
        return chunks

    if low_score_chunks:
        low_score_chunks.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return low_score_chunks[: min(top_k, RAG_FALLBACK_CHUNKS_IF_NONE)]

    return chunks


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context_parts = []
    total_chars = 0
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        filename = metadata.get("filename") or "document"
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue
        snippet = f"[S{idx}] Fichier: {filename}\n{content}"
        total_chars += len(snippet)
        if total_chars > RAG_MAX_CONTEXT_CHARS:
            break
        context_parts.append(snippet)

    context_block = "\n\n".join(context_parts)
    return (
        "Tu es un assistant RAG. Reponds uniquement a partir des sources fournies. "
        "Si l'information n'est pas dans les sources, dis-le explicitement. "
        "Reponds en francais, de facon concise et factuelle.\n\n"
        "Sources:\n"
        f"{context_block}\n\n"
        "Consigne de citation: quand tu affirmes un fait, ajoute la reference [Sx] correspondante.\n\n"
        "Question utilisateur:\n"
        f"{question}\n\n"
        "Reponse finale:"
    )


def generate_rag_answer(question: str, chunks: list[dict], model_name: str) -> str:
    prompt = build_rag_prompt(question, chunks)
    llm = Ollama(model=model_name)
    response = llm.invoke(prompt)
    if isinstance(response, str):
        return response.strip()
    return str(response).strip()


def run_rag_chat_pipeline(
    question: str, model_name: str, top_k: int
) -> tuple[dict, int]:
    try:
        documents = list_rag_documents()
    except Exception as exc:
        return {"error": f"Impossible de charger les documents indexes: {exc}"}, 500

    if not documents:
        return (
            {
                "reply": "Aucun document n'est indexe pour le moment. Importe d'abord des fichiers.",
                "sources": [],
                "used_rag": False,
            },
            200,
        )

    try:
        chunks = retrieve_rag_chunks(query=question, top_k=top_k)
    except Exception as exc:
        return {"error": f"Erreur de retrieval RAG: {exc}"}, 500

    if not chunks:
        return (
            {
                "reply": (
                    "Je n'ai pas trouve de passage pertinent dans les documents indexes pour repondre "
                    "de facon fiable a cette question."
                ),
                "sources": [],
                "used_rag": True,
            },
            200,
        )

    try:
        reply = generate_rag_answer(question=question, chunks=chunks, model_name=model_name)
    except Exception as exc:
        return {"error": f"Generation RAG impossible: {exc}"}, 500

    sources = build_rag_sources(chunks)
    return (
        {
            "reply": reply,
            "sources": sources,
            "used_rag": True,
            "retrieved_chunks": len(chunks),
            "top_k": top_k,
        },
        200,
    )


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------


def sanitize_message_content(content: Any) -> str:
    if not isinstance(content, str):
        return ""
    return content.strip()


def build_agent_input(user_message: str, conversation_history: list[dict]) -> str:
    if not conversation_history:
        return user_message

    normalized_history = []
    for item in conversation_history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = sanitize_message_content(item.get("content"))
        if role not in ("user", "assistant") or not content:
            continue
        normalized_history.append({"role": role, "content": content})

    if not normalized_history:
        return user_message

    trimmed_history = normalized_history[-MAX_CONTEXT_MESSAGES:]
    if (
        trimmed_history
        and trimmed_history[-1]["role"] == "user"
        and trimmed_history[-1]["content"] == user_message
    ):
        trimmed_history = trimmed_history[:-1]

    if not trimmed_history:
        return user_message

    history_lines = []
    for message in trimmed_history:
        speaker = "Utilisateur" if message["role"] == "user" else "Assistant"
        history_lines.append(f"{speaker}: {message['content']}")

    history_block = "\n".join(history_lines)
    return (
        "Contexte conversationnel (messages precedents):\n"
        f"{history_block}\n\n"
        "Nouvelle demande utilisateur:\n"
        f"{user_message}\n\n"
        "Donne une reponse concise, utile et coherente avec ce contexte."
    )


def build_plain_chat_input(user_message: str, conversation_history: list[dict]) -> str:
    normalized_history = []
    for item in conversation_history[-MAX_CONTEXT_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = sanitize_message_content(item.get("content"))
        if role in ("user", "assistant") and content:
            normalized_history.append((role, content))

    lines = [PLAIN_CHAT_SYSTEM_INSTRUCTION]
    if normalized_history:
        lines.append("\nContexte recent:")
        for role, content in normalized_history:
            speaker = "Utilisateur" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")
    lines.append("\nQuestion actuelle:")
    lines.append(user_message)
    lines.append("\nReponse finale:")
    return "\n".join(lines)


def get_plain_chat_reply(
    user_message: str, conversation_history: list[dict], model_name: str
) -> str | None:
    prompt = build_plain_chat_input(user_message, conversation_history)
    try:
        plain_llm = Ollama(model=model_name)
        response = plain_llm.invoke(prompt)
        if isinstance(response, str):
            cleaned = response.strip()
            return cleaned if cleaned else None
        cleaned = str(response).strip()
        return cleaned if cleaned else None
    except Exception:
        return None


def extract_final_answer(raw_output: str) -> str | None:
    if not raw_output:
        return None
    match = re.search(r"Final Answer:\s*(.*)", raw_output, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().strip("`").strip()


def extract_raw_output(raw_output: str) -> str | None:
    if not raw_output:
        return None
    match = re.search(
        r"Could not parse LLM output:\s*`([\s\S]*?)`",
        raw_output,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    match = re.search(
        r"Could not parse LLM output:\s*([\s\S]*?)(?:\nFor troubleshooting|\Z)",
        raw_output,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().strip("`").strip()
    return None


def fallback_parsing_reply(raw_output: str) -> str:
    generic_message = (
        "Je n'ai pas pu analyser proprement la reponse du modele. "
        "Reessaie en reformulant la demande en une phrase simple."
    )
    extracted = extract_raw_output(raw_output)
    if extracted:
        if "hypothetical scenario" in extracted.lower():
            return generic_message
        filtered_lines = [
            line
            for line in extracted.splitlines()
            if not re.match(
                r"^\s*(Thought|Action|Action Input|Observation|Question)\s*:",
                line,
                re.IGNORECASE,
            )
            and not re.search(
                r"output_parsing_failure|troubleshooting", line, re.IGNORECASE
            )
        ]
        cleaned = "\n".join(filtered_lines).strip()
        if cleaned:
            return cleaned
    final_answer = extract_final_answer(raw_output)
    if final_answer:
        return final_answer
    return generic_message


def extract_action_call(raw_output: str) -> tuple[str, str] | None:
    if not raw_output:
        return None
    pattern_with_input = (
        r"Action:\s*([A-Za-z0-9_-]+)\s*Action Input:\s*(.*?)(?:\n[A-Z][A-Za-z ]+:|$)"
    )
    matches = re.findall(pattern_with_input, raw_output, re.DOTALL)
    if matches:
        tool_name, action_input = matches[-1]
        return tool_name.strip(), action_input.strip().strip("`")

    pattern_no_input = r"Action\s*:\s*([A-Za-z0-9_-]+)\s*(?:\((.*?)\))?"
    matches = re.findall(pattern_no_input, raw_output, re.DOTALL)
    if matches:
        tool_name, action_input = matches[-1]
        return tool_name.strip(), (action_input or "").strip().strip("`")
    return None


def is_parsing_error_message(raw_output: str) -> bool:
    if not raw_output:
        return False
    lowered = raw_output.lower()
    return (
        "output parsing error occurred" in lowered
        or "output_parsing_failure" in lowered
        or "parsing llm output" in lowered
        or "could not parse llm output" in lowered
    )


def has_action_none(raw_output: str) -> bool:
    if not raw_output:
        return False
    return re.search(r"Action:\s*None\b", raw_output, re.IGNORECASE) is not None


def try_extract_reply_from_error(raw_output: str) -> str | None:
    if not raw_output:
        return None
    if is_parsing_error_message(raw_output):
        return fallback_parsing_reply(raw_output)
    raw_answer = extract_raw_output(raw_output)
    if raw_answer:
        return raw_answer
    final_answer = extract_final_answer(raw_output)
    if final_answer:
        return final_answer
    if is_parsing_error_message(raw_output):
        return fallback_parsing_reply(raw_output)
    return None


def normalize_intent_text(text: str) -> str:
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_quotes(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00ab", '"')
        .replace("\u00bb", '"')
        .replace("\u2019", "'")
    )


def get_approved_tool_args(tool_name: str) -> dict | None:
    approvals = _approved_tool_calls.get()
    for approval in approvals:
        if not isinstance(approval, dict):
            continue
        if approval.get("name") != tool_name:
            continue
        args = approval.get("args")
        if isinstance(args, dict):
            return args
    return None


def is_time_request(user_message: str) -> bool:
    text = normalize_intent_text(user_message)
    patterns = [
        r"\bquelle heure\b",
        r"\bil est quelle heure\b",
        r"\bheure\b",
        r"\btime\b",
        r"\bwhat time\b",
        r"\bcurrent time\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def is_create_note_request(user_message: str) -> bool:
    normalized_message = normalize_quotes(user_message)
    text = normalize_intent_text(normalized_message)
    has_target = bool(
        re.search(r"\b(note|fichier|file|txt)\b", text)
        or re.search(r"\b\w+\.txt\b", text)
    )
    has_action = bool(
        re.search(
            r"\b(cree|creer|ecrire|sauvegarder|enregistrer|write|create|met|mettre)\b", text
        )
    )
    has_txt_context = bool(re.search(r"\bdans\s+un?\s+\.?txt\b", text))
    return has_target and (has_action or has_txt_context)


def extract_note_args_from_message(user_message: str) -> tuple[dict | None, str | None]:
    normalized_message = normalize_quotes(user_message)

    filename_match = re.search(r"\b([a-zA-Z0-9._-]+\.txt)\b", normalized_message, re.IGNORECASE)
    filename = filename_match.group(1) if filename_match else "note.txt"

    content = None
    content_source = "none"
    quoted_match = re.search(r"\"([^\"]+)\"|'([^']+)'", normalized_message)
    if quoted_match:
        content = quoted_match.group(1) or quoted_match.group(2)
        content_source = "quoted"

    if not content:
        between_match = re.search(
            r"(?:ecrire|écrire|mettre)\s+(.+?)\s+(?:dans|en)\s+un?\s+\.?txt\b",
            normalized_message,
            re.IGNORECASE,
        )
        if between_match:
            content = between_match.group(1)
            content_source = "inferred"

    if not content:
        trailing_match = re.search(
            r"(?:avec|contenant|contenu|texte|ecrit|écrit)\s*[:\-]?\s*(.+)$",
            normalized_message,
            re.IGNORECASE,
        )
        if trailing_match:
            content = trailing_match.group(1)
            content_source = "inferred"

    if not content:
        return None, (
            "Je peux creer le .txt, mais j'ai besoin du contenu exact. "
            "Exemple: cree `note.txt` avec \"Taiwan est un pays libre\"."
        )

    cleaned_content = content.strip().strip(" .")
    if not cleaned_content:
        return None, (
            "Je peux creer le .txt, mais le contenu est vide. "
            "Donne-moi une phrase a ecrire entre guillemets."
        )

    if content_source != "quoted":
        normalized_content = normalize_intent_text(cleaned_content)
        asks_for_datetime = (
            ("date du jour" in normalized_content)
            or ("date" in normalized_content and "heure" in normalized_content)
            or ("heure actuelle" in normalized_content)
            or ("date actuelle" in normalized_content)
        )
        if asks_for_datetime:
            cleaned_content = current_datetime_text()

    return {"filename": filename, "content": cleaned_content}, None


def format_time_reply(raw_value: str) -> str:
    cleaned = raw_value.strip()
    try:
        parsed = datetime.fromisoformat(cleaned)
        return f"Il est actuellement {parsed.strftime('%d/%m/%Y %H:%M:%S')}."
    except Exception:
        return f"Heure actuelle: {cleaned}"


def current_datetime_text() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def confirmation_message(tool_name: str, tool_args: dict) -> str:
    if tool_name == "create_note":
        filename = tool_args.get("filename") or "le fichier"
        return f"Je peux creer le fichier '{filename}'. Souhaite-tu confirmer ?"
    return f"Je peux appeler l'outil '{tool_name}'. Souhaite-tu confirmer ?"


def try_direct_tool_reply(user_message: str, selected_tools) -> dict | None:
    selected_tool_names = {t.name for t in selected_tools}
    create_note_intent = is_create_note_request(user_message)
    if create_note_intent:
        if "create_note" not in selected_tool_names:
            return {"reply": "L'outil `Creer une note` est desactive. Active-le puis renvoie la demande."}
        approved_args = get_approved_tool_args("create_note")
        if approved_args:
            tool_args = approved_args
            clarification_message = None
        else:
            tool_args, clarification_message = extract_note_args_from_message(user_message)
        if clarification_message:
            return {"reply": clarification_message}
        try:
            result = tool_by_name["create_note"].invoke(tool_args)
            return {"reply": result}
        except ToolConfirmationRequired as confirmation_error:
            return {
                "reply": confirmation_message(
                    confirmation_error.tool_name, confirmation_error.tool_args
                ),
                "requires_confirmation": True,
                "tool_call": {
                    "name": confirmation_error.tool_name,
                    "args": confirmation_error.tool_args,
                },
            }
        except Exception:
            return None
    if is_time_request(user_message) and "get_current_time" in selected_tool_names:
        try:
            tool_result = tool_by_name["get_current_time"].invoke({})
            return {"reply": format_time_reply(str(tool_result))}
        except Exception:
            return None
    return None


def should_use_agent(user_message: str, selected_tools) -> bool:
    selected_tool_names = {t.name for t in selected_tools}
    text = normalize_intent_text(user_message)
    if "create_note" in selected_tool_names:
        note_patterns = [
            r"\bcree\b.*\b(note|fichier|txt)\b",
            r"\bcreer\b.*\b(note|fichier|txt)\b",
            r"\becrire\b.*\b(note|fichier|txt)\b",
            r"\bsauvegarder\b.*\b(note|fichier|txt)\b",
            r"\bwrite\b.*\bfile\b",
            r"\bcreate\b.*\b(note|file)\b",
        ]
        if any(re.search(pattern, text) for pattern in note_patterns):
            return True
    return False


def get_tool_fields(tool_obj) -> list[str]:
    schema = getattr(tool_obj, "args_schema", None)
    if not schema:
        return []
    fields = getattr(schema, "model_fields", None)
    if fields:
        return list(fields.keys())
    fields = getattr(schema, "__fields__", None)
    if fields:
        return list(fields.keys())
    return []


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def strip_wrapping_chars(value: str) -> str:
    cleaned = value.strip()
    for opening, closing in (("(", ")"), ("[", "]")):
        if cleaned.startswith(opening) and cleaned.endswith(closing):
            cleaned = cleaned[1:-1].strip()
    return cleaned


def parse_action_input(tool_obj, raw_input: str) -> dict:
    cleaned = strip_wrapping_chars(raw_input.strip().strip("`"))
    if cleaned.lower() in ("none", "null", ""):
        return {}
    if cleaned.startswith("{"):
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    fields = get_tool_fields(tool_obj)
    if fields:
        if len(fields) == 1:
            return {fields[0]: strip_quotes(strip_wrapping_chars(cleaned))}
        if "," in cleaned:
            parts = [
                strip_quotes(strip_wrapping_chars(part.strip()))
                for part in cleaned.split(",", maxsplit=len(fields) - 1)
            ]
            if len(parts) >= len(fields):
                return {field: parts[idx] for idx, field in enumerate(fields)}
    return {}


def resolve_placeholders(tool_args: dict) -> dict:
    resolved = {}
    for key, value in tool_args.items():
        if isinstance(value, str) and "get_current_time()" in value:
            current_time = get_current_time.invoke({})
            resolved[key] = value.replace("get_current_time()", current_time).strip()
        elif isinstance(value, str):
            resolved[key] = strip_wrapping_chars(value.strip())
        else:
            resolved[key] = value
    return resolved


def try_handle_action_from_text(raw_text: str) -> dict | None:
    action_call = extract_action_call(raw_text)
    if not action_call:
        return None
    tool_name, action_input = action_call
    tool_obj = tool_by_name.get(tool_name)
    if not tool_obj:
        return None
    tool_args = resolve_placeholders(parse_action_input(tool_obj, action_input))
    try:
        result = tool_obj.invoke(tool_args)
        return {"reply": result}
    except ToolConfirmationRequired as confirmation_error:
        return {
            "reply": confirmation_message(
                confirmation_error.tool_name, confirmation_error.tool_args
            ),
            "requires_confirmation": True,
            "tool_call": {
                "name": confirmation_error.tool_name,
                "args": confirmation_error.tool_args,
            },
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RAG endpoints
# ---------------------------------------------------------------------------


@app.post("/rag/documents", status_code=201)
async def rag_upload_documents(
    files: List[UploadFile] = File(default=[]),
    file: Optional[UploadFile] = File(default=None),
):
    """Upload and index documents for local RAG (PDF, TXT, MD)."""
    if not RAG_ENABLED:
        return JSONResponse({"error": "Le mode RAG est desactive."}, status_code=503)

    # Collect all uploaded files
    all_files: list[UploadFile] = [f for f in files if f.filename and f.filename.strip()]
    if file and file.filename and file.filename.strip():
        all_files.append(file)

    if not all_files:
        return JSONResponse(
            {"error": "Aucun fichier recu. Utilise le champ `files` (multipart/form-data)."},
            status_code=400,
        )

    try:
        vectorstore = get_rag_vectorstore()
    except Exception as exc:
        return JSONResponse(
            {"error": f"Initialisation RAG impossible: {exc}"}, status_code=500
        )

    added_documents: list[dict] = []
    errors: list[dict] = []
    ensure_rag_directories()

    for upload_file in all_files:
        original_filename = (upload_file.filename or "").strip()
        extension = get_file_extension(original_filename)

        if not is_allowed_upload(original_filename):
            errors.append(
                build_upload_error(
                    original_filename,
                    f"Extension non supportee ({extension or 'inconnue'}). Types autorises: .pdf, .txt, .md",
                )
            )
            continue

        try:
            content_bytes = await upload_file.read()
        except Exception:
            errors.append(build_upload_error(original_filename, "Impossible de lire le fichier."))
            continue

        size_bytes = len(content_bytes)

        if size_bytes <= 0:
            errors.append(build_upload_error(original_filename, "Fichier vide."))
            continue

        if size_bytes > RAG_MAX_FILE_SIZE_BYTES:
            errors.append(
                build_upload_error(
                    original_filename,
                    f"Fichier trop volumineux ({size_bytes} octets). Limite: {RAG_MAX_FILE_SIZE_BYTES} octets.",
                )
            )
            continue

        doc_id = str(uuid.uuid4())
        saved_filename = build_saved_filename(doc_id, original_filename)
        saved_path = RAG_UPLOAD_DIR / saved_filename

        try:
            saved_path.write_bytes(content_bytes)
            extracted_text = extract_text_from_file(saved_path, extension)
            if not extracted_text.strip():
                raise ValueError("Le fichier ne contient pas de texte exploitable.")

            chunks = split_text_into_chunks(extracted_text)
            if not chunks:
                raise ValueError("Le contenu du fichier est trop court ou vide apres decoupage.")

            created_at = datetime.now(timezone.utc).isoformat()
            chunk_ids = [f"{doc_id}:{idx}" for idx in range(len(chunks))]
            metadatas = [
                {
                    "doc_id": doc_id,
                    "filename": original_filename,
                    "file_type": extension.lstrip("."),
                    "chunk_index": idx,
                    "created_at": created_at,
                }
                for idx in range(len(chunks))
            ]

            vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=chunk_ids)
            persist_vectorstore(vectorstore)

            document_entry = {
                "doc_id": doc_id,
                "filename": original_filename,
                "stored_filename": saved_filename,
                "file_type": extension.lstrip("."),
                "size_bytes": size_bytes,
                "chunks": len(chunks),
                "created_at": created_at,
            }
            upsert_rag_registry_entry(document_entry)
            added_documents.append(document_entry)
        except Exception as exc:
            if saved_path.exists():
                try:
                    saved_path.unlink()
                except Exception:
                    pass
            errors.append(build_upload_error(original_filename, str(exc)))

    status_code = 201 if added_documents else 400
    payload: dict = {
        "added_documents": added_documents,
        "errors": errors,
        "total_uploaded": len(all_files),
        "total_added": len(added_documents),
        "total_failed": len(errors),
    }
    if status_code == 400:
        payload["error"] = "Aucun document n'a pu etre indexe."
    return JSONResponse(payload, status_code=status_code)


@app.get("/rag/documents")
def rag_list_documents_endpoint():
    """List indexed RAG documents."""
    if not RAG_ENABLED:
        return JSONResponse({"error": "Le mode RAG est desactive."}, status_code=503)

    try:
        documents = list_rag_documents()
    except Exception as exc:
        return JSONResponse(
            {"error": f"Impossible de lister les documents: {exc}"}, status_code=500
        )

    return {"documents": documents, "total": len(documents)}


@app.delete("/rag/documents/{doc_id}")
def rag_delete_document(doc_id: str):
    """Delete one document from index and local storage."""
    if not RAG_ENABLED:
        return JSONResponse({"error": "Le mode RAG est desactive."}, status_code=503)

    cleaned_doc_id = sanitize_message_content(doc_id)
    if not cleaned_doc_id:
        return JSONResponse({"error": "doc_id invalide."}, status_code=400)

    deleted_chunk_count = 0
    chunk_delete_error = None
    try:
        vectorstore = get_rag_vectorstore()
        collection = getattr(vectorstore, "_collection", None)
        if collection is not None:
            get_result = collection.get(where={"doc_id": cleaned_doc_id}, include=["metadatas"])
            chunk_ids = get_result.get("ids") or []
            if chunk_ids:
                collection.delete(ids=chunk_ids)
                deleted_chunk_count = len(chunk_ids)
                persist_vectorstore(vectorstore)
    except Exception as exc:
        chunk_delete_error = str(exc)

    removed_registry_entry = remove_rag_registry_entry(cleaned_doc_id)

    removed_files = []
    candidate_paths = []
    if removed_registry_entry and removed_registry_entry.get("stored_filename"):
        candidate_paths.append(RAG_UPLOAD_DIR / removed_registry_entry["stored_filename"])
    candidate_paths.extend(sorted(RAG_UPLOAD_DIR.glob(f"{cleaned_doc_id}_*")))

    seen_paths: set[str] = set()
    for path in candidate_paths:
        unique_key = str(path.resolve())
        if unique_key in seen_paths:
            continue
        seen_paths.add(unique_key)
        if path.exists():
            try:
                path.unlink()
                removed_files.append(path.name)
            except Exception:
                pass

    if not removed_registry_entry and deleted_chunk_count == 0 and not removed_files:
        return JSONResponse({"error": "Document introuvable."}, status_code=404)

    payload: dict = {
        "deleted": True,
        "doc_id": cleaned_doc_id,
        "deleted_chunks": deleted_chunk_count,
        "removed_files": removed_files,
    }
    if removed_registry_entry:
        payload["document"] = removed_registry_entry
    if chunk_delete_error:
        payload["warning"] = (
            "Le fichier a ete supprime, mais la suppression des chunks dans le vector store "
            f"a rencontre une erreur: {chunk_delete_error}"
        )
    return payload


@app.post("/rag/chat")
def rag_chat(body: RagChatRequest):
    """Answer a question using indexed RAG documents."""
    if not RAG_ENABLED:
        return JSONResponse({"error": "Le mode RAG est desactive."}, status_code=503)

    question = sanitize_message_content(body.question or body.message or "")
    if not question:
        return JSONResponse({"error": "Question manquante."}, status_code=400)

    model_name = sanitize_message_content(body.model or "") or DEFAULT_MODEL_NAME
    top_k = clamp_top_k(body.top_k if body.top_k is not None else RAG_DEFAULT_TOP_K)
    payload, status_code = run_rag_chat_pipeline(
        question=question,
        model_name=model_name,
        top_k=top_k,
    )
    return JSONResponse(payload, status_code=status_code)


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


@app.post("/chat")
async def chat(body: ChatRequest):
    """Handles chat messages from the frontend."""
    user_message = sanitize_message_content(body.message)
    if not user_message:
        return JSONResponse({"error": "No message provided"}, status_code=400)

    # Set per-request tool confirmation state via ContextVar
    _require_tool_confirmation.set(body.require_tool_confirmation)
    _approved_tool_calls.set(
        body.tool_approvals if isinstance(body.tool_approvals, list) else []
    )

    model_name = sanitize_message_content(body.model or "") or DEFAULT_MODEL_NAME
    use_rag = body.use_rag
    rag_top_k = clamp_top_k(
        body.rag_top_k
        if body.rag_top_k is not None
        else (body.top_k if body.top_k is not None else RAG_DEFAULT_TOP_K)
    )

    if isinstance(body.enabled_tools, list):
        selected_tools = [tool_by_name[name] for name in body.enabled_tools if name in tool_by_name]
    else:
        selected_tools = tools
    conversation_history = body.messages if isinstance(body.messages, list) else []

    direct_reply = try_direct_tool_reply(user_message, selected_tools)
    if direct_reply:
        return direct_reply

    if use_rag:
        payload, status_code = run_rag_chat_pipeline(
            question=user_message,
            model_name=model_name,
            top_k=rag_top_k,
        )
        return JSONResponse(payload, status_code=status_code)

    if not should_use_agent(user_message, selected_tools):
        plain_reply = get_plain_chat_reply(
            user_message=user_message,
            conversation_history=conversation_history,
            model_name=model_name,
        )
        if plain_reply:
            return {"reply": plain_reply}

    agent_executor = get_agent_executor(selected_tools, model_name=model_name)
    agent_input = build_agent_input(
        user_message=user_message,
        conversation_history=conversation_history,
    )

    try:
        response = agent_executor.invoke({"input": agent_input})
        ai_response = response.get("output")
        return {"reply": ai_response}

    except Exception as e:
        if isinstance(e, ToolConfirmationRequired):
            return {
                "reply": confirmation_message(e.tool_name, e.tool_args),
                "requires_confirmation": True,
                "tool_call": {"name": e.tool_name, "args": e.tool_args},
            }
        raw_error = str(e)
        if is_parsing_error_message(raw_error):
            direct_reply = try_direct_tool_reply(user_message, selected_tools)
            if direct_reply:
                return direct_reply
        action_response = try_handle_action_from_text(raw_error)
        if action_response:
            return action_response
        reply_from_error = try_extract_reply_from_error(raw_error)
        if reply_from_error:
            return {"reply": reply_from_error}

        if isinstance(e, OutputParserException):
            llm_output = getattr(e, "llm_output", None) or str(e)
            if is_parsing_error_message(str(llm_output)):
                direct_reply = try_direct_tool_reply(user_message, selected_tools)
                if direct_reply:
                    return direct_reply
                plain_reply = get_plain_chat_reply(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    model_name=model_name,
                )
                if plain_reply:
                    return {"reply": plain_reply}
            action_response = try_handle_action_from_text(llm_output)
            if action_response:
                return action_response
            reply_from_error = try_extract_reply_from_error(llm_output)
            if reply_from_error:
                return {"reply": reply_from_error}
            final_answer = extract_final_answer(llm_output)
            if final_answer:
                return {"reply": final_answer}
            raw_answer = extract_raw_output(llm_output)
            if raw_answer:
                return {"reply": raw_answer}
            if isinstance(llm_output, str) and llm_output.strip():
                return {"reply": llm_output}
            return {"reply": fallback_parsing_reply(str(e))}

        plain_reply = get_plain_chat_reply(
            user_message=user_message,
            conversation_history=conversation_history,
            model_name=model_name,
        )
        if plain_reply:
            return {"reply": plain_reply}
        print(f"Agent execution error: {e}")
        traceback.print_exc()
        return JSONResponse(
            {"error": "An error occurred while processing your request."},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Missions CRUD endpoints (T4)
# ---------------------------------------------------------------------------


def _mission_to_dict(mission: MissionModel) -> dict:
    return {
        "id": mission.id,
        "title": mission.title,
        "prompt": mission.prompt,
        "context_path": mission.context_path,
        "autonomy_level": mission.autonomy_level,
        "status": mission.status,
        "risk_level": mission.risk_level,
        "started_at": mission.started_at.isoformat() if mission.started_at else None,
        "ended_at": mission.ended_at.isoformat() if mission.ended_at else None,
    }


def _step_to_dict(step: MissionStepModel) -> dict:
    return {
        "id": step.id,
        "mission_id": step.mission_id,
        "node_name": step.node_name,
        "status": step.status,
        "input_payload": json.loads(step.input_payload) if step.input_payload else None,
        "output_payload": json.loads(step.output_payload) if step.output_payload else None,
        "error_payload": json.loads(step.error_payload) if step.error_payload else None,
        "started_at": step.started_at.isoformat() if step.started_at else None,
        "ended_at": step.ended_at.isoformat() if step.ended_at else None,
    }


@app.post("/missions", status_code=201)
def create_mission(body: MissionCreate, db: Session = Depends(get_db)):
    """Create a new mission."""
    mission = MissionModel(
        id=str(uuid.uuid4()),
        prompt=body.prompt,
        title=body.title,
        context_path=body.context_path,
        autonomy_level=body.autonomy_level,
        status="pending",
    )
    db.add(mission)
    db.commit()
    db.refresh(mission)
    return _mission_to_dict(mission)


@app.get("/missions")
def list_missions(status: Optional[str] = None, db: Session = Depends(get_db)):
    """List missions, optionally filtered by status."""
    query = db.query(MissionModel)
    if status:
        query = query.filter(MissionModel.status == status)
    missions = query.order_by(MissionModel.started_at.desc()).all()
    return {"missions": [_mission_to_dict(m) for m in missions], "total": len(missions)}


@app.get("/missions/{mission_id}")
def get_mission(mission_id: str, db: Session = Depends(get_db)):
    """Get mission detail including steps."""
    mission = db.query(MissionModel).filter(MissionModel.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission introuvable.")
    result = _mission_to_dict(mission)
    result["steps"] = [_step_to_dict(s) for s in mission.steps]
    return result


@app.get("/missions/{mission_id}/steps")
def get_mission_steps(mission_id: str, db: Session = Depends(get_db)):
    """Get the decision trail (steps) for a mission."""
    mission = db.query(MissionModel).filter(MissionModel.id == mission_id).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission introuvable.")
    steps = (
        db.query(MissionStepModel)
        .filter(MissionStepModel.mission_id == mission_id)
        .order_by(MissionStepModel.started_at)
        .all()
    )
    return {"mission_id": mission_id, "steps": [_step_to_dict(s) for s in steps]}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("--- Starting The Living Kernel ---")
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=False)
