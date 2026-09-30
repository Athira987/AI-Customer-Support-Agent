import pytest
from app.services.escalation import EscalationService


def test_pre_retrieval_human_trigger():
    """Verify explicit human agent requests are caught before vector search."""
    reason1 = EscalationService.should_escalate_pre_retrieval("I need to speak to human right now")
    assert reason1 is not None
    assert "human" in reason1.lower()

    reason2 = EscalationService.should_escalate_pre_retrieval("Can I speak with a person or live agent?")
    assert reason2 is not None

    reason3 = EscalationService.should_escalate_pre_retrieval("What is the battery life of NovaBook Pro?")
    assert reason3 is None

    reason4 = EscalationService.should_escalate_pre_retrieval("I want to speak to a human agent.")
    assert reason4 == "Customer explicitly requested human intervention"

    response4 = EscalationService.build_escalation_response(reason=reason4)
    assert response4.needs_escalation is True
    assert response4.reason == "Customer explicitly requested human intervention"


def test_distance_threshold_escalation():
    """Verify distance check escalates when retrieved content is dissimilar."""
    # Low distance (relevant)
    relevant_items = [
        {"chunk_id": "c1", "text": "Warranty lasts 1 year.", "distance": 0.25}
    ]
    assert EscalationService.should_escalate_distance(relevant_items, threshold=0.65) is None

    # High distance (irrelevant)
    irrelevant_items = [
        {"chunk_id": "c2", "text": "Something unrelated", "distance": 0.88}
    ]
    reason = EscalationService.should_escalate_distance(irrelevant_items, threshold=0.65)
    assert reason is not None
    assert "not sufficiently relevant" in reason

    # Empty list
    empty_reason = EscalationService.should_escalate_distance([], threshold=0.65)
    assert empty_reason is not None


def test_build_escalation_response():
    """Verify structured escalation response object generation."""
    response = EscalationService.build_escalation_response(
        reason="Information missing from knowledge base."
    )
    assert response.needs_escalation is True
    assert "NovaTech" in response.answer or "information" in response.answer
    assert response.reason == "Information missing from knowledge base."
    assert response.sources == []
