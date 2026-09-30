import pytest
from unittest.mock import MagicMock
from app.models.schemas import ChatRequest, ChatMessage, LLMStructuredOutput, SourceItem
from app.services.rag import RAGPipeline
from app.services.escalation import EscalationService


def test_standalone_retrieval_query():
    """Verify standalone question with no history uses the exact user message for retrieval."""
    mock_llm = MagicMock()
    mock_llm.rewrite_query.return_value = "What is the standard return window?"
    rag = RAGPipeline(
        embedding_service=MagicMock(),
        retrieval_service=MagicMock(),
        llm_service=mock_llm,
        escalation_service=MagicMock()
    )
    query = rag._build_retrieval_query("What is the standard return window?", conversation_history=None)
    assert query == "What is the standard return window?"
    mock_llm.rewrite_query.assert_called_with("What is the standard return window?", None)

    mock_llm.rewrite_query.return_value = "What is the standard return window?"
    query_empty_list = rag._build_retrieval_query("What is the standard return window?", conversation_history=[])
    assert query_empty_list == "What is the standard return window?"


def test_followup_returns_retrieval_query_with_product():
    """Verify follow-up return question delegates to LLMService rewrite_query."""
    mock_llm = MagicMock()
    mock_llm.rewrite_query.return_value = "NovaSound ANC Elite headphones return refund policy"
    rag = RAGPipeline(
        embedding_service=MagicMock(),
        retrieval_service=MagicMock(),
        llm_service=mock_llm,
        escalation_service=MagicMock()
    )
    history = [
        ChatMessage(role="user", content="Tell me about NovaSound ANC Elite"),
        ChatMessage(role="assistant", content="NovaSound ANC Elite features 40mm titanium drivers and 40h battery.")
    ]
    query = rag._build_retrieval_query("And what about returns?", conversation_history=history)
    assert query == "NovaSound ANC Elite headphones return refund policy"
    mock_llm.rewrite_query.assert_called_with("And what about returns?", history)


def test_followup_warranty_retrieval_query_with_history():
    """Verify follow-up warranty question constructs standalone search query using previous conversation context via LLM."""
    mock_llm = MagicMock()
    mock_llm.rewrite_query.return_value = "What is the warranty period for NovaTech products?"
    rag = RAGPipeline(
        embedding_service=MagicMock(),
        retrieval_service=MagicMock(),
        llm_service=mock_llm,
        escalation_service=MagicMock()
    )
    history = [
        ChatMessage(role="user", content="What is the standard return window?"),
        ChatMessage(role="assistant", content="NovaTech standard return window is 30 days.")
    ]
    query = rag._build_retrieval_query("And what about warranty?", conversation_history=history)
    assert query == "What is the warranty period for NovaTech products?"
    mock_llm.rewrite_query.assert_called_with("And what about warranty?", history)


def test_empty_or_system_only_history():
    """Verify history with only empty content or system roles is passed to llm_service."""
    mock_llm = MagicMock()
    mock_llm.rewrite_query.return_value = "What payment methods are accepted?"
    rag = RAGPipeline(
        embedding_service=MagicMock(),
        retrieval_service=MagicMock(),
        llm_service=mock_llm,
        escalation_service=MagicMock()
    )
    history = [
        ChatMessage(role="system", content="System instruction"),
        ChatMessage(role="user", content="   ")
    ]
    query = rag._build_retrieval_query("What payment methods are accepted?", conversation_history=history)
    assert query == "What payment methods are accepted?"
    mock_llm.rewrite_query.assert_called_with("What payment methods are accepted?", history)


def test_process_chat_multi_turn_embedding_and_llm_preservation():
    """Verify embedding receives rewritten query while LLM receives untouched customer message."""
    mock_embedding = MagicMock()
    mock_embedding.get_embedding.return_value = [0.1] * 384

    mock_retrieval = MagicMock()
    mock_retrieval.query_similar_with_filter.return_value = [
        {
            "chunk_id": "c1",
            "text": "NovaTech standard return window is 30 days for hardware products.",
            "distance": 0.2,
            "metadata": {"filename": "return_and_refund.txt", "category": "policies", "page": 1}
        }
    ]

    mock_llm = MagicMock()
    mock_llm.generate_support_response.return_value = LLMStructuredOutput(
        answer="NovaTech offers a 30-day return window for NovaSound ANC Elite.",
        needs_escalation=False,
        reason=None,
        used_sources=["return_and_refund.txt"]
    )
    mock_llm.rewrite_query.return_value = "Rewritten NovaSound Returns Query"

    rag = RAGPipeline(
        embedding_service=mock_embedding,
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        escalation_service=EscalationService()
    )

    history = [
        ChatMessage(role="user", content="Tell me about NovaSound ANC Elite"),
        ChatMessage(role="assistant", content="NovaSound ANC Elite features hybrid active noise cancellation.")
    ]
    request = ChatRequest(
        message="And what about returns?",
        conversation_history=history
    )

    response = rag.process_chat(request)

    # Verify embedding received enriched retrieval query from LLM
    mock_embedding.get_embedding.assert_called_once()
    actual_query = mock_embedding.get_embedding.call_args[0][0]
    assert actual_query == "Rewritten NovaSound Returns Query"

    # Verify LLM received untouched user question and original conversation history
    mock_llm.generate_support_response.assert_called_once()
    call_kwargs = mock_llm.generate_support_response.call_args.kwargs
    assert call_kwargs["user_question"] == "And what about returns?"
    assert call_kwargs["conversation_history"] == history

    # Verify response
    assert response.needs_escalation is False
    assert "30-day" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].document == "return_and_refund.txt"


def test_process_chat_pre_retrieval_escalation_preserved():
    """Verify human escalation triggers pre-retrieval without calling embedding service."""
    mock_embedding = MagicMock()
    mock_retrieval = MagicMock()
    mock_llm = MagicMock()

    rag = RAGPipeline(
        embedding_service=mock_embedding,
        retrieval_service=mock_retrieval,
        llm_service=mock_llm,
        escalation_service=EscalationService()
    )

    request = ChatRequest(message="I want to speak to a human agent.")
    response = rag.process_chat(request)

    assert response.needs_escalation is True
    assert response.reason == "Customer explicitly requested human intervention"
    mock_embedding.get_embedding.assert_not_called()
    mock_retrieval.query_similar.assert_not_called()
    mock_llm.generate_support_response.assert_not_called()
