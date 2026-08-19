import logging
from typing import List, Optional
from app.models.schemas import SupportResponse, SourceItem
from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_ESCALATION_MESSAGE = (
    "I don't have enough verified company information to answer that accurately. "
    "I have flagged this inquiry for a NovaTech customer support specialist who will assist you shortly."
)


class EscalationService:
    """Service to evaluate and handle support escalation logic."""

    @staticmethod
    def should_escalate_pre_retrieval(user_message: str) -> Optional[str]:
        """
        Check for explicit customer requests to speak with a human agent or supervisor.
        """
        message_lower = user_message.lower().strip()
        human_triggers = [
            "speak to human",
            "talk to human",
            "speak with a person",
            "talk to a person",
            "human representative",
            "live agent",
            "speak to manager",
            "transfer me to support agent",
            "customer representative"
        ]
        for trigger in human_triggers:
            if trigger in message_lower:
                return f"Customer explicitly requested human intervention: '{trigger}'"
        return None

    @staticmethod
    def should_escalate_distance(retrieved_items: list, threshold: Optional[float] = None) -> Optional[str]:
        """
        Check if the nearest retrieved document exceeds the maximum cosine distance threshold.
        """
        thresh = threshold if threshold is not None else settings.similarity_distance_threshold

        if not retrieved_items:
            return "No matching company knowledge base documents found for this question."

        best_distance = min(item.get("distance", 1.0) for item in retrieved_items)
        if best_distance > thresh:
            return f"Retrieved knowledge base content is not sufficiently relevant (cosine distance: {best_distance:.3f} > {thresh:.2f})."

        return None

    @staticmethod
    def build_escalation_response(
        reason: str,
        custom_message: Optional[str] = None,
        sources: Optional[List[SourceItem]] = None
    ) -> SupportResponse:
        """Construct a standardized customer-facing escalation response."""
        return SupportResponse(
            answer=custom_message or DEFAULT_ESCALATION_MESSAGE,
            needs_escalation=True,
            reason=reason,
            sources=sources or []
        )
