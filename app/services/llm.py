import json
import logging
from typing import List, Optional
from openai import OpenAI, OpenAIError, AuthenticationError, RateLimitError, APIConnectionError
from app.core.config import settings
from app.models.schemas import ChatMessage, LLMStructuredOutput

logger = logging.getLogger(__name__)


class LLMService:
    """Service to communicate with OpenAI LLMs using current official SDK."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """Lazy-initialize OpenAI client."""
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is missing or invalid in .env file. "
                "Please add a valid OpenAI API key to proceed."
            )
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate_support_response(
        self,
        system_instruction: str,
        retrieved_context: str,
        user_question: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> LLMStructuredOutput:
        """Generate structured customer support response using OpenAI."""
        client = self._get_client()

        # Build prompt messages
        messages = [
            {"role": "system", "content": system_instruction}
        ]

        # Add recent conversation turns if available
        if conversation_history:
            for msg in conversation_history[-6:]:  # Keep last 6 turns to avoid context overflow
                if msg.role in {"user", "assistant"}:
                    messages.append({"role": msg.role, "content": msg.content})

        # Add current context and customer question
        user_prompt = (
            f"--- RETRIEVED COMPANY KNOWLEDGE BASE CONTEXT ---\n"
            f"{retrieved_context}\n"
            f"--- END CONTEXT ---\n\n"
            f"CUSTOMER QUESTION: {user_question}\n\n"
            f"Formulate your response. If the retrieved context is empty or does not provide enough "
            f"information to answer accurately, set needs_escalation=true and explain the reason."
        )
        messages.append({"role": "user", "content": user_prompt})

        try:
            # Modern OpenAI Structured Output using beta.chat.completions.parse
            completion = client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=LLMStructuredOutput,
                temperature=0.1,  # Low temperature for deterministic, factual grounding
            )
            parsed_output: LLMStructuredOutput = completion.choices[0].message.parsed
            if parsed_output is None:
                raise ValueError("Received empty parsed output from OpenAI model.")
            return parsed_output

        except AuthenticationError as e:
            logger.error(f"OpenAI Authentication error: {e}")
            raise RuntimeError("Invalid OpenAI API Key. Please verify your credentials in .env.") from e
        except RateLimitError as e:
            logger.error(f"OpenAI Rate limit exceeded: {e}")
            raise RuntimeError("OpenAI rate limit or quota exceeded. Please check your account limits.") from e
        except APIConnectionError as e:
            logger.error(f"Failed to connect to OpenAI API: {e}")
            raise RuntimeError("Network connection error reaching OpenAI servers.") from e
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI service error: {str(e)}") from e
        except Exception as e:
            # Fallback in case of parse or structure error
            logger.warning(f"Standard parse failed ({e}). Attempting JSON fallback...")
            return self._fallback_json_completion(client, messages)

    def _fallback_json_completion(self, client: OpenAI, messages: list) -> LLMStructuredOutput:
        """Fallback method using standard JSON response format."""
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            return LLMStructuredOutput(
                answer=data.get("answer", "I am unable to process your request at this time."),
                needs_escalation=data.get("needs_escalation", True),
                reason=data.get("reason", "Fallback response parser invoked."),
                used_sources=data.get("used_sources", [])
            )
        except Exception as e:
            logger.error(f"Fallback completion failed: {e}")
            return LLMStructuredOutput(
                answer="We encountered an issue processing your inquiry. A human support specialist will assist you.",
                needs_escalation=True,
                reason="LLM response parsing failure.",
                used_sources=[]
            )
