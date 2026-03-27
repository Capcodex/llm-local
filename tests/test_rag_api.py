from __future__ import annotations

import io


def _upload_txt(client, filename: str = "doc.txt", content: str = "alpha beta gamma"):
    files = [("files", (filename, io.BytesIO(content.encode("utf-8")), "text/plain"))]
    return client.post("/rag/documents", files=files)


def test_upload_list_delete_flow(client, rag_env, monkeypatch):
    server = rag_env["server"]
    vectorstore = rag_env["vectorstore"]

    monkeypatch.setattr(server, "split_text_into_chunks", lambda text: ["chunk-1", "chunk-2"])

    upload_response = _upload_txt(client, filename="notes.txt", content="contenu initial")
    assert upload_response.status_code == 201

    upload_payload = upload_response.json()
    assert upload_payload["total_uploaded"] == 1
    assert upload_payload["total_added"] == 1
    assert upload_payload["total_failed"] == 0

    added_doc = upload_payload["added_documents"][0]
    doc_id = added_doc["doc_id"]
    assert added_doc["filename"] == "notes.txt"
    assert added_doc["chunks"] == 2
    assert len(vectorstore._store) == 2

    first_metadata = next(iter(vectorstore._store.values()))["metadata"]
    assert first_metadata["doc_id"] == doc_id
    assert first_metadata["filename"] == "notes.txt"
    assert first_metadata["file_type"] == "txt"
    assert isinstance(first_metadata["chunk_index"], int)
    assert first_metadata["created_at"]

    list_response = client.get("/rag/documents")
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert list_payload["total"] == 1
    assert list_payload["documents"][0]["doc_id"] == doc_id

    delete_response = client.delete(f"/rag/documents/{doc_id}")
    assert delete_response.status_code == 200
    delete_payload = delete_response.json()
    assert delete_payload["deleted"] is True
    assert delete_payload["deleted_chunks"] == 2
    assert len(delete_payload["removed_files"]) == 1

    list_after_delete = client.get("/rag/documents")
    assert list_after_delete.status_code == 200
    assert list_after_delete.json()["total"] == 0


def test_rag_chat_returns_sources(client, rag_env, monkeypatch):
    server = rag_env["server"]
    monkeypatch.setattr(server, "split_text_into_chunks", lambda text: ["source one", "source two"])
    monkeypatch.setattr(
        server,
        "generate_rag_answer",
        lambda question, chunks, model_name: "Reponse fondee sur [S1].",
    )

    upload_response = _upload_txt(client, filename="knowledge.md", content="Taiwan est ...")
    assert upload_response.status_code == 201

    response = client.post(
        "/rag/chat",
        json={
            "question": "Taiwan est-il libre ?",
            "top_k": 3,
        },
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["used_rag"] is True
    assert payload["reply"] == "Reponse fondee sur [S1]."
    assert payload["retrieved_chunks"] >= 1
    assert payload["top_k"] == 3
    assert len(payload["sources"]) >= 1
    assert payload["sources"][0]["filename"] == "knowledge.md"


def test_chat_route_uses_rag_pipeline_when_enabled(client, rag_env, monkeypatch):
    server = rag_env["server"]

    calls = []

    def _fake_pipeline(question: str, model_name: str, top_k: int):
        calls.append({"question": question, "model_name": model_name, "top_k": top_k})
        return ({"reply": "ok-rag", "sources": [], "used_rag": True}, 200)

    monkeypatch.setattr(server, "run_rag_chat_pipeline", _fake_pipeline)

    response = client.post(
        "/chat",
        json={
            "message": "Question test",
            "messages": [{"role": "user", "content": "Question test"}],
            "use_rag": True,
            "rag_top_k": 6,
            "enabled_tools": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == "ok-rag"
    assert payload["used_rag"] is True
    assert calls == [{"question": "Question test", "model_name": server.DEFAULT_MODEL_NAME, "top_k": 6}]


def test_registry_persists_document_listing_after_vectorstore_reset(
    client, rag_env, monkeypatch
):
    server = rag_env["server"]

    monkeypatch.setattr(server, "split_text_into_chunks", lambda text: ["persisted chunk"])

    upload_response = _upload_txt(client, filename="persist.txt", content="memoire durable")
    assert upload_response.status_code == 201
    doc_id = upload_response.json()["added_documents"][0]["doc_id"]

    rag_env["vectorstore"]._store.clear()

    response = client.get("/rag/documents")
    assert response.status_code == 200
    payload = response.json()

    assert payload["total"] == 1
    assert payload["documents"][0]["doc_id"] == doc_id
