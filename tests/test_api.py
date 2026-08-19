import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.models.schemas import SupportResponse, SourceItem

client = TestClient(app)


def test_health_check():
    """Verify /health returns HTTP 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "NovaTech" in data["service"]


def test_chat_empty_message_validation():
    """Verify empty message triggers 422 Unprocessable Entity."""
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_missing_message_field():
    """Verify missing message triggers 422."""
    response = client.post("/chat", json={})
    assert response.status_code == 422


@patch("app.services.rag.RAGPipeline.process_chat")
def test_chat_successful_response(mock_process_chat):
    """Verify /chat returns structured response when RAG processes normally."""
    mock_process_chat.return_value = SupportResponse(
        answer="NovaTech offers a 30-day standard return window for hardware products.",
        needs_escalation=False,
        reason=None,
        sources=[
            SourceItem(
                document="return_and_refund.txt",
                category="policies",
                page=1,
                snippet="NovaTech Electronics offers a 30-day standard return window..."
            )
        ]
    )

    response = client.post("/chat", json={"message": "What is the return window?"})
    assert response.status_code == 200
    data = response.json()
    assert data["needs_escalation"] is False
    assert "30-day" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document"] == "return_and_refund.txt"


@patch("app.services.rag.RAGPipeline.process_chat")
def test_chat_escalated_response(mock_process_chat):
    """Verify /chat returns escalation flags when question is out of scope."""
    mock_process_chat.return_value = SupportResponse(
        answer="I don't have enough verified information to answer that question.",
        needs_escalation=True,
        reason="Information not found in knowledge base.",
        sources=[]
    )

    response = client.post("/chat", json={"message": "Do you sell spaceship engines?"})
    assert response.status_code == 200
    data = response.json()
    assert data["needs_escalation"] is True
    assert data["reason"] is not None
    assert len(data["sources"]) == 0
