from pathlib import Path

from app.ingestion.chunking import chunk_document, recursive_split, split_sentences
from app.ingestion.embed import HashEmbedder
from app.ingestion.loaders import LoadedDocument, PageBlock, load_document
from app.ingestion.pipeline import ingest_path

DOCS = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


def test_pdf_loader_keeps_pages_and_sections():
    doc = load_document(DOCS / "product_recall_policy.pdf")
    pages = {b.page for b in doc.blocks}
    sections = {b.section for b in doc.blocks}
    assert len(pages) >= 2
    assert "Recall Classes" in sections
    assert "Class I" in doc.text


def test_docx_loader_uses_headings_as_sections():
    doc = load_document(DOCS / "return_and_refund_policy.docx")
    assert {"General Returns", "Exclusions"} <= {b.section for b in doc.blocks}


def test_markdown_loader():
    doc = load_document(DOCS / "paid_time_off_policy.md")
    assert "Carryover" in {b.section for b in doc.blocks}


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_recursive_split_respects_max():
    text = " ".join(["word"] * 1000)
    parts = recursive_split(text, 300, overlap=0)
    assert all(len(p) <= 300 for p in parts)
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")


def test_semantic_chunking_separates_topics_and_sets_metadata():
    text = ("Refrigerators must stay cold at 41 degrees. Freezers must stay at zero degrees. "
            "Cold cases are checked often. The loyalty program gives points for purchases. "
            "Points can be redeemed for discounts. Members earn rewards on every dollar.")
    doc = LoadedDocument("d1", "x.md", [PageBlock(text, page=3, section="Mixed")])
    chunks = chunk_document(doc, HashEmbedder(), max_chars=400, min_chars=10, percentile=70)
    assert len(chunks) >= 2
    md = chunks[0].metadata
    assert md["page"] == 3 and md["section"] == "Mixed" and md["source"] == "x.md"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_ingestion_is_idempotent(store):
    before = store.count()
    results = ingest_path(DOCS, store=store)
    assert all(r.skipped for r in results)
    assert store.count() == before
    assert len(store.doc_ids()) >= 20
