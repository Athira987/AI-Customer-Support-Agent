import pytest
from unittest.mock import patch, MagicMock
from app.services.llm import LLMService
from app.models.schemas import LLMStructuredOutput


def test_llm_parse_normal_rag_response():
    """Verify normal RAG JSON response parses into LLMStructuredOutput cleanly."""
    llm = LLMService()
    valid_json = """
    {
        "answer": "NovaTech offers a 30-day standard return window for all hardware products.",
        "needs_escalation": false,
        "reason": null,
        "used_sources": ["return_and_refund.txt"]
    }
    """
    parsed = llm._parse_structured_output(valid_json)
    assert isinstance(parsed, LLMStructuredOutput)
    assert "30-day" in parsed.answer
    assert parsed.needs_escalation is False
    assert parsed.reason is None
    assert parsed.used_sources == ["return_and_refund.txt"]


def test_llm_parse_escalation_response():
    """Verify escalation JSON response sets needs_escalation=True and preserves reason."""
    llm = LLMService()
    escalation_json = """
    {
        "answer": "I do not have enough verified information about spaceship engines.",
        "needs_escalation": true,
        "reason": "Query out of knowledge base scope.",
        "used_sources": []
    }
    """
    parsed = llm._parse_structured_output(escalation_json)
    assert isinstance(parsed, LLMStructuredOutput)
    assert parsed.needs_escalation is True
    assert parsed.reason == "Query out of knowledge base scope."
    assert parsed.used_sources == []


def test_llm_parse_markdown_wrapped_json():
    """Verify markdown fences (```json ... ```) are stripped and parsed correctly."""
    llm = LLMService()
    markdown_json = """
    ```json
    {
        "answer": "Express shipping takes 2 business days and costs $14.99.",
        "needs_escalation": false,
        "reason": null,
        "used_sources": ["shipping_policy.txt"]
    }
    ```
    """
    parsed = llm._parse_structured_output(markdown_json)
    assert isinstance(parsed, LLMStructuredOutput)
    assert "2 business days" in parsed.answer
    assert parsed.needs_escalation is False
    assert parsed.used_sources == ["shipping_policy.txt"]


def test_llm_parse_malformed_json_graceful_fallback():
    """Verify malformed/invalid JSON does not crash and falls back gracefully."""
    llm = LLMService()
    malformed_json = '{"answer": "Incomplete text without closing brace...'
    parsed = llm._parse_structured_output(malformed_json)
    assert isinstance(parsed, LLMStructuredOutput)
    assert parsed.answer is not None
    assert len(parsed.answer) > 0


def test_llm_parse_plain_text_fallback():
    """Verify raw plain-text responses are handled as valid answers without throwing exceptions."""
    llm = LLMService()
    plain_text = "The standard return period is 30 days from the delivery date."
    parsed = llm._parse_structured_output(plain_text)
    assert isinstance(parsed, LLMStructuredOutput)
    assert "30 days" in parsed.answer
    assert parsed.needs_escalation is False


def test_llm_parse_type_coercion():
    """Verify strings and mixed types for booleans and sources are safely normalized."""
    llm = LLMService()
    messy_json = """
    {
        "answer": "Warranty covers 2 years for NovaSound.",
        "needs_escalation": "false",
        "reason": "",
        "used_sources": "warranty_policy.txt"
    }
    """
    parsed = llm._parse_structured_output(messy_json)
    assert parsed.needs_escalation is False
    assert parsed.reason is None
    assert parsed.used_sources == ["warranty_policy.txt"]


@patch("app.services.llm.LLMService._get_client")
def test_llm_generate_support_response_integration(mock_get_client):
    """Verify end-to-end generate_support_response calls OpenAI client in json_object mode."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"answer": "Standard return is 30 days.", "needs_escalation": false, "reason": null, "used_sources": ["return_and_refund.txt"]}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    llm = LLMService()
    res = llm.generate_support_response(
        system_instruction="You are NovaTech Support AI.",
        retrieved_context="Return policy is 30 days.",
        user_question="What is the return policy?"
    )

    assert isinstance(res, LLMStructuredOutput)
    assert "30 days" in res.answer
    assert res.needs_escalation is False
    assert res.used_sources == ["return_and_refund.txt"]
    mock_client.chat.completions.create.assert_called_once()
