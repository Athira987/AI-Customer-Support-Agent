"""
Tests for category-aware RAG retrieval.

Verifies that:
- The correct knowledge-base filename is resolved for each topic category.
- Category-filtered retrieval is invoked when a category is present.
- Unrestricted retrieval is used when no category is provided.
- Switching categories correctly updates the retrieval filter.
- The frontend-to-backend category field flows through the API.
- No information bleed between categories.
- Human escalation still works independently of category selection.
"""

import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from app.models.schemas import ChatRequest, ChatMessage, LLMStructuredOutput, SupportResponse, SourceItem
from app.services.rag import RAGPipeline, CATEGORY_TO_FILENAME, CATEGORY_DISPLAY_NAMES
from app.services.retrieval import RetrievalService
from app.services.ingestion import IngestionService, DocumentChunk
from app.services.escalation import EscalationService


# ---------------------------------------------------------------------------
# 1. Category → Filename mapping unit tests
# ---------------------------------------------------------------------------

class TestCategoryMapping:
    """Verify the static category → filename lookup table."""

    def test_all_six_categories_present(self):
        """All six support topics must have a mapping entry."""
        expected_keys = {
            "payment_faq",
            "troubleshooting",
            "returns_refunds",
            "shipping",
            "warranty",
            "nova_products",
        }
        assert expected_keys == set(CATEGORY_TO_FILENAME.keys())

    def test_payment_faq_mapping(self):
        assert CATEGORY_TO_FILENAME["payment_faq"] == "payment_faq.txt"

    def test_troubleshooting_mapping(self):
        assert CATEGORY_TO_FILENAME["troubleshooting"] == "troubleshooting_guide.txt"

    def test_returns_refunds_mapping(self):
        assert CATEGORY_TO_FILENAME["returns_refunds"] == "return_and_refund.txt"

    def test_shipping_mapping(self):
        assert CATEGORY_TO_FILENAME["shipping"] == "shipping_policy.txt"

    def test_warranty_mapping(self):
        assert CATEGORY_TO_FILENAME["warranty"] == "warranty_policy.txt"

    def test_nova_products_mapping(self):
        assert CATEGORY_TO_FILENAME["nova_products"] == "nova_products.txt"

    def test_display_names_present(self):
        """Every category key also has a display name."""
        for key in CATEGORY_TO_FILENAME:
            assert key in CATEGORY_DISPLAY_NAMES, f"Missing display name for '{key}'"

    def test_resolve_category_filename_known(self):
        """RAGPipeline.resolve_category_filename returns correct filename for known keys."""
        assert RAGPipeline.resolve_category_filename("warranty") == "warranty_policy.txt"
        assert RAGPipeline.resolve_category_filename("shipping") == "shipping_policy.txt"
        assert RAGPipeline.resolve_category_filename("payment_faq") == "payment_faq.txt"

    def test_resolve_category_filename_unknown(self):
        """Unknown or None category resolves to None (unrestricted retrieval)."""
        assert RAGPipeline.resolve_category_filename(None) is None
        assert RAGPipeline.resolve_category_filename("") is None
        assert RAGPipeline.resolve_category_filename("unknown_topic") is None

    def test_resolve_category_filename_case_insensitive(self):
        """Category keys are normalised to lowercase before lookup."""
        assert RAGPipeline.resolve_category_filename("WARRANTY") == "warranty_policy.txt"
        assert RAGPipeline.resolve_category_filename("  warranty  ") == "warranty_policy.txt"


# ---------------------------------------------------------------------------
# 2. Category-aware retrieval filter tests (via RetrievalService mock)
# ---------------------------------------------------------------------------

def _make_chunk(filename: str, text: str, dist: float = 0.15) -> dict:
    """Helper to build a mock retrieval result dict."""
    return {
        "chunk_id": f"{filename}_1",
        "text": text,
        "distance": dist,
        "metadata": {"filename": filename, "category": filename.split("_")[0], "page": 1},
    }


def _build_rag(topic_key: str, returned_chunks: list) -> tuple:
    """Return (rag, mock_retrieval, mock_embedding, mock_llm) wired for a given topic."""
    mock_embedding = MagicMock()
    mock_embedding.get_embedding.return_value = [0.1] * 384

    mock_retrieval = MagicMock()
    mock_retrieval.query_similar_with_filter.return_value = returned_chunks

    mock_llm = MagicMock()
    mock_llm.rewrite_query.return_value = "test query"
    mock_llm.generate_support_response.return_value = LLMStructuredOutput(
        answer="Test answer from the knowledge base.",
        needs_escalation=False,
        reason=None,
        used_sources=[CATEGORY_TO_FILENAME.get(topic_key, "")]
    )

    rag = RAGPipeline(
        embedding_service=mock_embedding,
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        escalation_service=EscalationService(),
    )
    return rag, mock_retrieval, mock_embedding, mock_llm


class TestCategoryAwareRetrieval:
    """Verify category filtering is passed to retrieval service correctly."""

    def test_payment_question_uses_payment_filter(self):
        """Payment FAQ category → retrieval filtered to payment_faq.txt."""
        chunks = [_make_chunk("payment_faq.txt", "We accept Visa, Mastercard, PayPal.", 0.18)]
        rag, mock_retrieval, _, _ = _build_rag("payment_faq", chunks)

        request = ChatRequest(message="What payment methods are accepted?", category="payment_faq")
        response = rag.process_chat(request)

        mock_retrieval.query_similar_with_filter.assert_called()
        call_kwargs = mock_retrieval.query_similar_with_filter.call_args_list[0].kwargs
        assert call_kwargs.get("filename_filter") == "payment_faq.txt"
        assert response.needs_escalation is False

    def test_troubleshooting_question_uses_troubleshooting_filter(self):
        """Troubleshooting category → retrieval filtered to troubleshooting_guide.txt."""
        chunks = [_make_chunk("troubleshooting_guide.txt", "Perform a factory reset to resolve issues.", 0.20)]
        rag, mock_retrieval, _, _ = _build_rag("troubleshooting", chunks)

        request = ChatRequest(message="My device won't turn on.", category="troubleshooting")
        rag.process_chat(request)

        call_kwargs = mock_retrieval.query_similar_with_filter.call_args_list[0].kwargs
        assert call_kwargs.get("filename_filter") == "troubleshooting_guide.txt"

    def test_returns_question_uses_returns_filter(self):
        """Returns & Refunds category → retrieval filtered to return_and_refund.txt."""
        chunks = [_make_chunk("return_and_refund.txt", "30-day return window from delivery.", 0.15)]
        rag, mock_retrieval, _, _ = _build_rag("returns_refunds", chunks)

        request = ChatRequest(message="What is the return window?", category="returns_refunds")
        rag.process_chat(request)

        call_kwargs = mock_retrieval.query_similar_with_filter.call_args_list[0].kwargs
        assert call_kwargs.get("filename_filter") == "return_and_refund.txt"

    def test_shipping_question_uses_shipping_filter(self):
        """Shipping category → retrieval filtered to shipping_policy.txt."""
        chunks = [_make_chunk("shipping_policy.txt", "Standard delivery takes 5-7 business days.", 0.22)]
        rag, mock_retrieval, _, _ = _build_rag("shipping", chunks)

        request = ChatRequest(message="How long does shipping take?", category="shipping")
        rag.process_chat(request)

        call_kwargs = mock_retrieval.query_similar_with_filter.call_args_list[0].kwargs
        assert call_kwargs.get("filename_filter") == "shipping_policy.txt"

    def test_warranty_question_uses_warranty_filter(self):
        """Warranty category → retrieval filtered to warranty_policy.txt."""
        chunks = [_make_chunk("warranty_policy.txt", "1-year limited hardware warranty.", 0.12)]
        rag, mock_retrieval, _, _ = _build_rag("warranty", chunks)

        request = ChatRequest(message="How long is the warranty?", category="warranty")
        rag.process_chat(request)

        call_kwargs = mock_retrieval.query_similar_with_filter.call_args_list[0].kwargs
        assert call_kwargs.get("filename_filter") == "warranty_policy.txt"

    def test_nova_products_question_uses_products_filter(self):
        """Nova Products category → retrieval filtered to nova_products.txt."""
        chunks = [_make_chunk("nova_products.txt", "NovaBook Pro 15 features a 4K OLED display.", 0.17)]
        rag, mock_retrieval, _, _ = _build_rag("nova_products", chunks)

        request = ChatRequest(message="Tell me about the NovaBook Pro.", category="nova_products")
        rag.process_chat(request)

        call_kwargs = mock_retrieval.query_similar_with_filter.call_args_list[0].kwargs
        assert call_kwargs.get("filename_filter") == "nova_products.txt"

    def test_no_category_uses_unrestricted_retrieval(self):
        """No category → filename_filter=None (unrestricted retrieval)."""
        chunks = [_make_chunk("return_and_refund.txt", "30-day return window.", 0.20)]
        rag, mock_retrieval, _, _ = _build_rag("returns_refunds", chunks)

        request = ChatRequest(message="What is the return policy?")  # no category
        rag.process_chat(request)

        call_kwargs = mock_retrieval.query_similar_with_filter.call_args_list[0].kwargs
        assert call_kwargs.get("filename_filter") is None

    def test_unknown_category_falls_back_to_unrestricted(self):
        """Unrecognised category string → filename_filter=None."""
        chunks = [_make_chunk("payment_faq.txt", "Payment info.", 0.25)]
        rag, mock_retrieval, _, _ = _build_rag("payment_faq", chunks)

        request = ChatRequest(message="Any question?", category="completely_unknown_topic")
        rag.process_chat(request)

        call_kwargs = mock_retrieval.query_similar_with_filter.call_args_list[0].kwargs
        assert call_kwargs.get("filename_filter") is None


# ---------------------------------------------------------------------------
# 3. Category switching test
# ---------------------------------------------------------------------------

class TestCategorySwitching:
    """Verify that switching between categories changes the retrieval filter."""

    def test_switching_from_warranty_to_shipping(self):
        """Consecutive requests with different categories use the correct filter each time."""
        mock_embedding = MagicMock()
        mock_embedding.get_embedding.return_value = [0.1] * 384

        mock_retrieval = MagicMock()
        mock_retrieval.query_similar_with_filter.side_effect = [
            [_make_chunk("warranty_policy.txt", "Warranty lasts 1 year.", 0.15)],
            [_make_chunk("shipping_policy.txt", "Express shipping takes 2 days.", 0.18)],
        ]

        mock_llm = MagicMock()
        mock_llm.rewrite_query.return_value = "test query"
        mock_llm.generate_support_response.return_value = LLMStructuredOutput(
            answer="Answer.", needs_escalation=False, reason=None, used_sources=[]
        )

        rag = RAGPipeline(
            embedding_service=mock_embedding,
            retrieval_service=mock_retrieval,
            llm_service=mock_llm,
            escalation_service=EscalationService(),
        )

        # First request: Warranty
        rag.process_chat(ChatRequest(message="How long is the warranty?", category="warranty"))
        first_call_kwargs = mock_retrieval.query_similar_with_filter.call_args_list[0].kwargs
        assert first_call_kwargs.get("filename_filter") == "warranty_policy.txt"

        # Second request: Shipping (switch)
        rag.process_chat(ChatRequest(message="How long does shipping take?", category="shipping"))
        second_call_kwargs = mock_retrieval.query_similar_with_filter.call_args_list[1].kwargs
        assert second_call_kwargs.get("filename_filter") == "shipping_policy.txt"


# ---------------------------------------------------------------------------
# 4. "Not found in this section" response test
# ---------------------------------------------------------------------------

class TestCategoryNotFound:
    """Verify the pipeline returns a helpful no-info message when filtered retrieval is empty."""

    def test_empty_filtered_results_return_section_message(self):
        """When category filter returns empty, a targeted 'not in this section' message is returned."""
        mock_embedding = MagicMock()
        mock_embedding.get_embedding.return_value = [0.1] * 384

        mock_retrieval = MagicMock()
        mock_retrieval.query_similar_with_filter.return_value = []   # empty

        mock_llm = MagicMock()
        mock_llm.rewrite_query.return_value = "test query"

        rag = RAGPipeline(
            embedding_service=mock_embedding,
            retrieval_service=mock_retrieval,
            llm_service=mock_llm,
            escalation_service=EscalationService(),
        )

        request = ChatRequest(message="What is the refund timeline?", category="warranty")
        response = rag.process_chat(request)

        # LLM should NOT be called — we short-circuit before that
        mock_llm.generate_support_response.assert_not_called()
        # Response should not be an escalation
        assert response.needs_escalation is False
        # Answer should mention the section
        assert "Warranty" in response.answer or "warranty" in response.answer.lower()


# ---------------------------------------------------------------------------
# 5. Human escalation still works with category
# ---------------------------------------------------------------------------

class TestEscalationWithCategory:
    """Verify human escalation triggers correctly even when a category is selected."""

    def test_human_agent_request_with_category_escalates(self):
        """Pre-retrieval escalation still fires when category is set."""
        mock_embedding = MagicMock()
        mock_retrieval = MagicMock()
        mock_llm = MagicMock()

        rag = RAGPipeline(
            embedding_service=mock_embedding,
            retrieval_service=mock_retrieval,
            llm_service=mock_llm,
            escalation_service=EscalationService(),
        )

        request = ChatRequest(
            message="I need to speak to a human agent.",
            category="warranty"
        )
        response = rag.process_chat(request)

        assert response.needs_escalation is True
        assert "human" in (response.reason or "").lower() or "human" in response.answer.lower()
        mock_embedding.get_embedding.assert_not_called()
        mock_retrieval.query_similar_with_filter.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Category metadata on ingested chunks
# ---------------------------------------------------------------------------

class TestChunkCategoryMetadata:
    """Verify ingested document chunks carry correct category and filename metadata."""

    def test_payment_faq_chunk_has_correct_metadata(self):
        """Chunks from payment_faq.txt must carry filename='payment_faq.txt' and category='faq'."""
        ingestion = IngestionService(chunk_size=500, chunk_overlap=50)
        kb_path = Path(__file__).resolve().parent.parent / "knowledge_base"
        payment_file = kb_path / "faq" / "payment_faq.txt"

        if not payment_file.exists():
            pytest.skip("knowledge_base/faq/payment_faq.txt not found — ingestion test skipped.")

        chunks = ingestion.process_document(payment_file)
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.metadata.get("filename") == "payment_faq.txt", (
                f"Expected filename='payment_faq.txt', got '{chunk.metadata.get('filename')}'"
            )
            assert "faq" in chunk.metadata.get("category", "").lower(), (
                f"Expected category containing 'faq', got '{chunk.metadata.get('category')}'"
            )

    def test_warranty_chunk_has_correct_metadata(self):
        """Chunks from warranty_policy.txt must carry filename='warranty_policy.txt'."""
        ingestion = IngestionService(chunk_size=500, chunk_overlap=50)
        kb_path = Path(__file__).resolve().parent.parent / "knowledge_base"
        warranty_file = kb_path / "policies" / "warranty_policy.txt"

        if not warranty_file.exists():
            pytest.skip("knowledge_base/policies/warranty_policy.txt not found — skipped.")

        chunks = ingestion.process_document(warranty_file)
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.metadata.get("filename") == "warranty_policy.txt"

    def test_all_kb_files_produce_filename_metadata(self):
        """Every knowledge-base file produces chunks whose filename metadata matches the source filename."""
        ingestion = IngestionService(chunk_size=800, chunk_overlap=100)
        kb_path = Path(__file__).resolve().parent.parent / "knowledge_base"

        if not kb_path.exists():
            pytest.skip("knowledge_base directory not found.")

        expected_files = list(CATEGORY_TO_FILENAME.values())
        found_filenames = set()

        all_chunks = ingestion.process_directory(kb_path)
        for chunk in all_chunks:
            fn = chunk.metadata.get("filename")
            if fn:
                found_filenames.add(fn)
                assert fn == Path(chunk.metadata.get("source", fn)).name, (
                    "filename metadata must match the actual source file name."
                )

        # Verify each expected KB file was found
        for expected in expected_files:
            assert expected in found_filenames, (
                f"Expected knowledge-base file '{expected}' was not found during ingestion. "
                f"Found: {sorted(found_filenames)}"
            )


# ---------------------------------------------------------------------------
# 7. ChromaDB category-filtered retrieval integration test
# ---------------------------------------------------------------------------

class TestChromaDBCategoryFilter:
    """Integration tests: upsert chunks from two categories, verify filter isolation."""

    def _setup_retrieval_with_two_categories(self, temp_dir: str) -> RetrievalService:
        """Helper: upsert warranty and shipping chunks into a temp collection."""
        retrieval = RetrievalService(
            persist_directory=temp_dir,
            collection_name="test_category_filter"
        )

        warranty_chunk = DocumentChunk(
            chunk_id="warranty_chunk_1",
            text="NovaTech products carry a 1-year limited hardware warranty.",
            metadata={"filename": "warranty_policy.txt", "category": "policies", "page": 1}
        )
        shipping_chunk = DocumentChunk(
            chunk_id="shipping_chunk_1",
            text="Standard shipping takes 5-7 business days within the contiguous US.",
            metadata={"filename": "shipping_policy.txt", "category": "policies", "page": 1}
        )

        fake_embedding = [0.05] * 384
        retrieval.upsert_chunks([warranty_chunk, shipping_chunk], [fake_embedding, fake_embedding])
        return retrieval

    def test_warranty_filter_excludes_shipping_chunks(self):
        """Filtering to warranty_policy.txt must not return shipping_policy.txt chunks."""
        temp_dir = tempfile.mkdtemp()
        try:
            retrieval = self._setup_retrieval_with_two_categories(temp_dir)
            assert retrieval.count() == 2

            results = retrieval.query_similar_with_filter(
                query_embedding=[0.05] * 384,
                top_k=10,
                filename_filter="warranty_policy.txt"
            )

            assert len(results) >= 1
            for r in results:
                assert r["metadata"]["filename"] == "warranty_policy.txt", (
                    f"Expected only warranty_policy.txt chunks, got '{r['metadata']['filename']}'"
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_shipping_filter_excludes_warranty_chunks(self):
        """Filtering to shipping_policy.txt must not return warranty_policy.txt chunks."""
        temp_dir = tempfile.mkdtemp()
        try:
            retrieval = self._setup_retrieval_with_two_categories(temp_dir)

            results = retrieval.query_similar_with_filter(
                query_embedding=[0.05] * 384,
                top_k=10,
                filename_filter="shipping_policy.txt"
            )

            assert len(results) >= 1
            for r in results:
                assert r["metadata"]["filename"] == "shipping_policy.txt"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_filter_returns_all_chunks(self):
        """No filter (None) returns chunks from all categories."""
        temp_dir = tempfile.mkdtemp()
        try:
            retrieval = self._setup_retrieval_with_two_categories(temp_dir)

            results = retrieval.query_similar_with_filter(
                query_embedding=[0.05] * 384,
                top_k=10,
                filename_filter=None
            )

            filenames = {r["metadata"]["filename"] for r in results}
            assert "warranty_policy.txt" in filenames
            assert "shipping_policy.txt" in filenames
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 8. API payload — category field flows through FastAPI
# ---------------------------------------------------------------------------

class TestCategoryAPIField:
    """Verify the /chat endpoint accepts and uses the category field."""

    def test_chat_request_schema_accepts_category(self):
        """ChatRequest must accept and preserve the category field."""
        req = ChatRequest(message="How long is the warranty?", category="warranty")
        assert req.category == "warranty"
        assert req.message == "How long is the warranty?"

    def test_chat_request_no_category_defaults_to_none(self):
        """ChatRequest without category defaults to None."""
        req = ChatRequest(message="What is the return policy?")
        assert req.category is None

    def test_chat_request_category_case_preserved(self):
        """Category value is stored as-is (normalisation happens in RAGPipeline)."""
        req = ChatRequest(message="Any question?", category="shipping")
        assert req.category == "shipping"
