from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = "ok"
    service: str = "NovaTech AI Customer Support Agent"
    version: str = "1.0.0"


class ChatMessage(BaseModel):
    """Single message in a multi-turn conversation."""
    role: str = Field(..., description="Role of the speaker, e.g. 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if v_clean not in {"user", "assistant", "system"}:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return v_clean


class ChatRequest(BaseModel):
    """Inbound customer chat request schema."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer's question or message",
        examples=["What is your return policy for opened items?"]
    )
    conversation_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Recent dialogue history for multi-turn context"
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Message cannot be empty or solely whitespace.")
        return clean


class SourceItem(BaseModel):
    """Source reference returned with RAG answers."""
    document: str = Field(..., description="Filename of the knowledge base document")
    category: Optional[str] = Field(default=None, description="Category/subfolder of document")
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    snippet: Optional[str] = Field(default=None, description="Excerpt of retrieved text chunk")


class SupportResponse(BaseModel):
    """Structured customer support response schema."""
    answer: str = Field(..., description="AI response to the customer")
    needs_escalation: bool = Field(..., description="Whether this request requires human agent intervention")
    reason: Optional[str] = Field(default=None, description="Reason for escalation if applicable")
    sources: List[SourceItem] = Field(default_factory=list, description="List of source document citations")


class LLMStructuredOutput(BaseModel):
    """Internal schema for OpenAI structured generation."""
    answer: str = Field(..., description="The direct, helpful customer answer grounded solely in the provided context")
    needs_escalation: bool = Field(..., description="True if evidence is insufficient, outside knowledge base, or conflicting")
    reason: Optional[str] = Field(default=None, description="Short explanation if escalation is needed, otherwise null")
    used_sources: List[str] = Field(default_factory=list, description="Filenames of documents actually used in formulating the answer")
