from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain import hub

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class _DummyPrompt:
    input_variables = ["tools", "tool_names", "agent_scratchpad", "input"]
    template = "Question: {input}\n{agent_scratchpad}"
    partial_variables = {}


class FakeDocument:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class FakeCollection:
    def __init__(self, store: dict[str, dict]):
        self._store = store

    def get(self, where: dict | None = None, include: list[str] | None = None) -> dict:
        rows = list(self._store.values())
        if where and where.get("doc_id"):
            target_doc_id = where["doc_id"]
            rows = [item for item in rows if item["metadata"].get("doc_id") == target_doc_id]
        return {
            "ids": [item["id"] for item in rows],
            "metadatas": [item["metadata"] for item in rows],
        }

    def delete(self, ids: list[str]) -> None:
        for chunk_id in ids:
            self._store.pop(chunk_id, None)


class FakeVectorStore:
    def __init__(self):
        self._store: dict[str, dict] = {}
        self._collection = FakeCollection(self._store)

    def add_texts(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        for text, metadata, chunk_id in zip(texts, metadatas, ids):
            self._store[chunk_id] = {
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
            }

    def similarity_search_with_relevance_scores(self, query: str, k: int):
        del query
        rows = list(self._store.values())[:k]
        results = []
        score = 0.95
        for row in rows:
            results.append((FakeDocument(row["text"], row["metadata"]), score))
            score = max(score - 0.1, 0.2)
        return results

    def similarity_search_with_score(self, query: str, k: int):
        del query
        rows = list(self._store.values())[:k]
        return [(FakeDocument(row["text"], row["metadata"]), 0.1) for row in rows]

    def persist(self) -> None:
        return None


@pytest.fixture(scope="session")
def server_module():
    """Import `server.py` without hitting LangChain Hub network calls."""
    original_pull = hub.pull
    hub.pull = lambda *args, **kwargs: _DummyPrompt()
    try:
        if "server" in sys.modules:
            module = importlib.reload(sys.modules["server"])
        else:
            module = importlib.import_module("server")
    finally:
        hub.pull = original_pull
    return module


@pytest.fixture
def rag_env(server_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    upload_dir = data_dir / "uploads"
    vector_dir = data_dir / "vectorstore"
    registry_path = data_dir / "documents_registry.json"

    monkeypatch.setattr(server_module, "RAG_BASE_DIR", data_dir)
    monkeypatch.setattr(server_module, "RAG_UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(server_module, "RAG_VECTORSTORE_DIR", vector_dir)
    monkeypatch.setattr(server_module, "RAG_REGISTRY_PATH", registry_path)

    fake_vectorstore = FakeVectorStore()
    monkeypatch.setattr(server_module, "get_rag_vectorstore", lambda: fake_vectorstore)
    monkeypatch.setattr(server_module, "persist_vectorstore", lambda _vs: None)

    server_module._rag_vectorstore = fake_vectorstore
    server_module._rag_embedding_model = object()

    server_module.ensure_rag_directories()

    return {
        "server": server_module,
        "vectorstore": fake_vectorstore,
        "data_dir": data_dir,
        "upload_dir": upload_dir,
        "registry_path": registry_path,
    }


@pytest.fixture
def client(rag_env):
    from fastapi.testclient import TestClient

    server = rag_env["server"]
    with TestClient(server.app) as test_client:
        yield test_client
