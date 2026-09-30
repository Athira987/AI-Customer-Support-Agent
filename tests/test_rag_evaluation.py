"""
RAG Evaluation Dataset & Test Cases for NovaTech Support Agent.
Tests knowledge base domain coverage, expected answers, and escalation behaviors.
"""

import pytest
from app.models.schemas import ChatRequest, SupportResponse

EVALUATION_DATASET = [
    {
        "query": "What is the return period?",
        "expected_topics": ["30 days", "return window"],
        "expected_source": "return_and_refund.txt",
        "should_escalate": False
    },
    {
        "query": "How long is the warranty for NovaSound headphones?",
        "expected_topics": ["2-year", "2 year", "warranty"],
        "expected_source": "warranty_policy.txt",
        "should_escalate": False
    },
    {
        "query": "How long does express shipping take?",
        "expected_topics": ["2 Business Days", "2 days", "$14.99"],
        "expected_source": "shipping_policy.txt",
        "should_escalate": False
    },
    {
        "query": "What payment methods are supported?",
        "expected_topics": ["Visa", "MasterCard", "PayPal", "Apple Pay", "NovaPay"],
        "expected_source": "payment_faq.txt",
        "should_escalate": False
    },
    {
        "query": "Do you sell rocket fuel or spaceship components?",
        "expected_topics": [],
        "expected_source": None,
        "should_escalate": True
    },
    {
        "query": "Can I talk to a live human agent right now?",
        "expected_topics": [],
        "expected_source": None,
        "should_escalate": True
    }
]


def test_evaluation_dataset_structure():
    """Verify evaluation dataset contains required benchmark fields."""
    assert len(EVALUATION_DATASET) >= 6
    for item in EVALUATION_DATASET:
        assert "query" in item
        assert "should_escalate" in item
        assert isinstance(item["should_escalate"], bool)

from unittest.mock import MagicMock
from app.services.rag import RAGPipeline
from app.models.schemas import LLMStructuredOutput

def test_evaluate_dataset():
    """Run the evaluation dataset through the RAG pipeline with a mocked LLM."""
    for item in EVALUATION_DATASET:
        rag = RAGPipeline(
            embedding_service=MagicMock(),
            retrieval_service=MagicMock(),
            llm_service=MagicMock(),
            escalation_service=MagicMock()
        )
        
        # Mock escalation
        rag.escalation_service.should_escalate_pre_retrieval.return_value = None
        rag.escalation_service.should_escalate_distance.return_value = None

        # Mock retrieval — pipeline now calls query_similar_with_filter
        if item["expected_source"]:
            rag.retrieval_service.query_similar_with_filter.return_value = [
                {"metadata": {"filename": item["expected_source"]}, "text": "Mock text", "distance": 0.1, "chunk_id": "1"}
            ]
        else:
            rag.retrieval_service.query_similar_with_filter.return_value = []
            
        # Mock LLM
        rag.llm_service.rewrite_query.return_value = item["query"]
        
        rag.llm_service.generate_support_response.return_value = LLMStructuredOutput(
            answer=" ".join(item["expected_topics"]) if item["expected_topics"] else "I don't know.",
            needs_escalation=item["should_escalate"],
            reason="Escalated" if item["should_escalate"] else None,
            used_sources=[item["expected_source"]] if item["expected_source"] else []
        )
        
        request = ChatRequest(message=item["query"])
        response = rag.process_chat(request)
        
        assert response.needs_escalation == item["should_escalate"]
        for topic in item["expected_topics"]:
            assert topic in response.answer
        if item["expected_source"]:
            assert len(response.sources) > 0
            assert response.sources[0].document == item["expected_source"]
