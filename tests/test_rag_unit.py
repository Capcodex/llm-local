from __future__ import annotations

import sys
import types
from pathlib import Path


def test_extract_text_from_txt_and_md(server_module, tmp_path: Path):
    txt_path = tmp_path / "notes.txt"
    md_path = tmp_path / "guide.md"
    txt_path.write_text("  Bonjour\nmonde  ", encoding="utf-8")
    md_path.write_text("# Titre\n\nContenu markdown", encoding="utf-8")

    txt_result = server_module.extract_text_from_file(txt_path, ".txt")
    md_result = server_module.extract_text_from_file(md_path, ".md")

    assert txt_result == "Bonjour\nmonde"
    assert md_result == "# Titre\n\nContenu markdown"


def test_extract_text_from_pdf_with_mocked_reader(server_module, monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    class _FakePage:
        def __init__(self, text: str):
            self._text = text

        def extract_text(self):
            return self._text

    class _FakeReader:
        def __init__(self, file_path: str):
            assert file_path.endswith("fake.pdf")
            self.pages = [_FakePage("Page 1"), _FakePage("  "), _FakePage("Page 2")]

    fake_pypdf = types.SimpleNamespace(PdfReader=_FakeReader)
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    result = server_module.extract_text_from_pdf(pdf_path)

    assert result == "Page 1\n\nPage 2"


def test_split_text_into_chunks_returns_clean_chunks(server_module):
    text = ("A" * 1200) + "\n" + ("B" * 1200)
    chunks = server_module.split_text_into_chunks(text)

    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)


def test_build_rag_sources_includes_metadata_and_preview(server_module):
    chunks = [
        {
            "content": "X" * 350,
            "metadata": {
                "doc_id": "doc-1",
                "filename": "manuel.md",
                "chunk_index": 3,
            },
            "score": 0.87654,
        }
    ]

    sources = server_module.build_rag_sources(chunks)

    assert len(sources) == 1
    source = sources[0]
    assert source["source_id"] == "S1"
    assert source["doc_id"] == "doc-1"
    assert source["filename"] == "manuel.md"
    assert source["chunk_index"] == 3
    assert source["score"] == 0.8765
    assert source["preview"].endswith("...")
    assert len(source["preview"]) == server_module.RAG_MAX_SOURCE_PREVIEW_CHARS + 3
