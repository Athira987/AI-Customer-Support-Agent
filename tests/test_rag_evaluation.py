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
