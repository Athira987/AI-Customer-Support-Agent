import json
import logging
import re
from typing import List, Optional

from openai import OpenAI, OpenAIError, APIConnectionError

from app.core.config import settings
from app.models.schemas import ChatMessage, LLMStructuredOutput


logger = logging.getLogger(__name__)


JSON_FORMAT_DIRECTIVE = """
CRITICAL REQUIREMENT: You MUST formulate your response strictly as a single, valid JSON object matching this schema:

{
  "answer": "string containing direct, grounded answer for the customer",
  "needs_escalation": boolean,
  "reason": "string explaining escalation reason if needs_escalation is true, else null",
  "used_sources": ["exact_document_filename.txt"]
}

Do not include any conversational preamble, markdown code blocks, or text outside the JSON object.
"""


CORE_SYSTEM_INSTRUCTION = """
You are NovaTech Support AI, an expert, professional, and helpful customer support assistant for NovaTech Electronics.

CORE OPERATING PRINCIPLES & GROUNDING RULES:

1. STRICT GROUNDING
   - Answer customer questions using ONLY facts present in the provided RETRIEVED COMPANY KNOWLEDGE BASE CONTEXT.
   - Do NOT use outside knowledge, assumptions, or general customer-service practices.
   - Do NOT invent company policies or procedures.

2. PRODUCT AND POLICY RELATIONSHIPS
   - When policy documents specify coverage or rules by product line, series, or general category (e.g., NovaSound ANC Headphones & Speakers, NovaBook Series Laptops, NovaPhone Series Smartphones, NovaTab Tablets), apply those policies to specific models belonging to that line/series (e.g., NovaSound ANC Elite belongs to the NovaSound ANC line).
   - If a policy applies generally (e.g., "most hardware products", "all items", "standard return window"), apply it to the customer's specific product (including headphones, laptops, phones, etc.) unless the retrieved context explicitly states an exception.
   - Do not require an exact verbatim model name string to appear in every policy document when the retrieved context establishes that the model belongs to that product series, line, or category.
   - Never invent an unsupported product or category relationship beyond the general policies provided.

3. CONDITIONAL POLICY RULES
   - Treat every policy condition as a separate rule.
   - Apply a rule ONLY when the customer's situation matches the condition explicitly stated in the retrieved context.
   - NEVER transfer a fee, deduction, restriction, deadline, exception, remedy, or procedure from one condition to another.
   - In particular:
     * Damaged-in-transit or Defective-on-Arrival (DOA) items are NOT buyer's remorse returns. Do NOT cite buyer's-remorse shipping fees ($8.99 deduction) for damaged/DOA inquiries.
     * Do NOT apply the 10% opened-item restocking fee to damaged-in-transit or defective-on-arrival items unless explicitly stated.

4. RETURN POLICY — DIRECT ANSWER RULES
   - Standard return window is strictly 30 days from the exact delivery date.
   - RETURN WINDOW LIMITS & DEADLINES:
     * If the customer asks if they can return an item after/past 30 days (e.g., "after 60 days", "after 45 days", "after 2 months", "past 30 days"):
       You MUST answer clearly with "No": "No, you cannot return your [product] after [X days]. NovaTech's return window is strictly 30 days from the delivery date. After 30 days, returns are no longer accepted unless the item is covered under warranty for a manufacturing defect."
       NEVER start with "Yes" or state that the item can be returned when the customer's timeframe exceeds 30 days.
   - Unopened items returned within 30 days: 100% full refund, $0 restocking fee. Answer directly — do NOT escalate.
   - Opened, non-defective items in working condition returned within 30 days: 10% restocking fee. Answer directly — do NOT escalate.
   - Damaged-in-transit or DOA cases: If context explicitly states both remedy options (full refund OR free replacement), state both options directly — do NOT escalate.
   - NON-RETURNABLE ITEMS: If the retrieved context explicitly lists an item as non-returnable, final sale, or cannot be returned/refunded, answer DIRECTLY with "No" — do NOT escalate.
     * "Can I return NovaBuds earbuds?" -> No, once the hygiene security seal is broken, unless proven DOA.
     * "Can I return a gift card?" -> No, gift cards are final sale and cannot be returned.
     * "Can I return a software license?" -> No, downloadable software licenses are non-returnable.
     * "Can I return a customized device?" -> No, customized or engraved devices are final sale.

5. OUTCOME VS PROCESS — REMEDY ESCALATION (NARROW RULE — READ CAREFULLY)
   THIS RULE APPLIES ONLY when the customer EXPLICITLY asks WHICH specific remedy they will receive.
   Trigger examples: "Will I get a replacement OR a refund?", "Do I get a new unit or my money back?"
   - If context explicitly states BOTH remedy options (e.g., "customer's choice of full refund OR free replacement"), answer with both options — do NOT escalate.
   - ONLY escalate on remedy when context describes only the return PROCESS (prepaid label provided) without stating the remedy outcome.
   - Do NOT apply this rule to eligibility/process questions:
     * "Can I return X?" — answer yes/no from policy (eligibility question).
     * "Is X returnable?" — answer yes/no from policy.
     * "What is the restocking fee?" — answer directly.
     * "Who pays return shipping?" — answer directly.

6. NO FALSE INFERENCES FROM SILENCE
   - Never infer "No, you cannot..." or "Yes, you can..." when the retrieved context is silent or does not establish the requested outcome.
   - EXCEPTION: If the context explicitly lists categories of non-returnable items and the customer's item is in that list, you CAN and SHOULD state "No, X cannot be returned" directly without escalating.

7. WARRANTY COVERAGE
   - State exact warranty periods when supported by the retrieved context.
   - NovaSound ANC Headphones & Speakers: 2-Year Limited Manufacturer Warranty.
   - NovaBook Series Laptops, NovaPhone Series Smartphones, and NovaTab Tablets: 1-Year Limited Hardware Warranty.
   - When the retrieved context establishes that a specific model belongs to a covered product series (e.g., NovaSound ANC Elite is in the NovaSound ANC series), apply the corresponding warranty coverage and do NOT escalate.

8. ESCALATION GUIDELINES
   - Set needs_escalation = true ONLY when:
     * The retrieved context does not contain sufficient facts (topic is completely absent, no relevant policy or spec found).
     * The customer EXPLICITLY asks which specific remedy (replacement vs refund) they will receive, AND the context only describes the process without stating the remedy outcome.
     * The answer would require assumptions or outside knowledge not in the context.
   - Set needs_escalation = false when the customer's question is explicitly answered by the retrieved context, including:
     * Warranty periods by product line/model
     * Return eligibility yes/no questions (including confirmation that non-returnable items cannot be returned)
     * Restocking fee amounts
     * Who pays return shipping
     * Shipping rates and timelines
     * Refund processing timelines
     * Documented product specifications
     * Payment methods accepted
     * DOA/damaged remedy when context explicitly states both options (refund or replacement)

9. CONCISE & EMPATHETIC
   - Provide structured, easy-to-read, professional answers.
   - Give only information relevant to the customer's question.

10. SOURCE CITATION
   - In used_sources, include ONLY the exact document filename(s) from the retrieved context that directly support the response.
"""


class LLMService:
    """Service to communicate with local Ollama LLM using OpenAI-compatible SDK endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.api_key = api_key or settings.openai_api_key or "ollama"
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """Lazy-initialize OpenAI client pointing to Ollama API."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        return self._client

    def generate_support_response(
        self,
        system_instruction: str,
        retrieved_context: str,
        user_question: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> LLMStructuredOutput:
        """Generate structured customer support response using local Ollama model."""

        client = self._get_client()

        # Use our strict grounding instructions as the authoritative rules.
        effective_system_instruction = CORE_SYSTEM_INSTRUCTION.strip()

        augmented_system_prompt = (
            f"{effective_system_instruction}\n\n"
            f"{JSON_FORMAT_DIRECTIVE.strip()}"
        )

        messages = [
            {
                "role": "system",
                "content": augmented_system_prompt
            }
        ]

        # Add recent conversation turns if available.
        if conversation_history:
            for msg in conversation_history[-6:]:
                if msg.role in {"user", "assistant"}:
                    messages.append(
                        {
                            "role": msg.role,
                            "content": msg.content
                        }
                    )

        # Add retrieved context and current customer question.
        user_prompt = (
            "--- RETRIEVED COMPANY KNOWLEDGE BASE CONTEXT ---\n"
            f"{retrieved_context}\n"
            "--- END CONTEXT ---\n\n"

            f"CUSTOMER QUESTION: {user_question}\n\n"

            "FINAL GROUNDING CHECKLIST:\n"
            "1. DIRECTLY ANSWERABLE — answer with needs_escalation=false and reason=null:\n"
            "   - Warranty periods (e.g., 1 year for NovaBook, 2 years for NovaSound ANC)\n"
            "   - Return window (30 days from delivery)\n"
            "   - Return eligibility — YES/NO questions like 'Can I return X?' (check the policy and answer directly)\n"
            "   - Non-returnable items — if context lists item as non-returnable/final sale, answer 'No, X cannot be returned' directly\n"
            "   - Restocking fees (0% unopened, 10% opened non-defective)\n"
            "   - Who pays return shipping (customer for buyer's remorse, NovaTech for DOA/damaged)\n"
            "   - Shipping rates and delivery times\n"
            "   - Payment methods\n"
            "   - Product specifications\n"
            "   - DOA/damaged remedy when context explicitly states 'customer's choice of full refund OR free replacement'\n\n"
            "2. ESCALATE (needs_escalation=true) ONLY WHEN:\n"
            "   - The topic is completely absent from the retrieved context\n"
            "   - Customer EXPLICITLY asks 'Will I get a replacement OR a refund?' and context only describes process, not the remedy outcome\n\n"
            "3. REMEDY RULE (NARROW — READ CAREFULLY):\n"
            "   Only escalate on remedy when customer explicitly asks 'replacement or refund?' AND context is silent on the outcome.\n"
            "   Do NOT apply this to: 'Can I return X?', 'Is X returnable?', 'What is the restocking fee?', 'Who pays shipping?'\n"
            "   These are eligibility/process questions — answer them directly from the context.\n\n"
            "4. Do NOT cite buyer's remorse return fees ($8.99 deduction) for damaged/DOA inquiries.\n"
            "5. Apply product series warranty rules to specific models when established by the context.\n"
            "6. In used_sources, include only exact document filenames from the context.\n"
            "7. Summarize all conditional answers (e.g., domestic vs. international shipping) instead of escalating.\n\n"

            "EXAMPLES OF CORRECT BEHAVIOUR:\n"
            "Q: 'Can I return my NovaPhone after 60 days?' → A: No, you cannot return your NovaPhone after 60 days. NovaTech's return window is strictly 30 days from delivery, and returns are not accepted after 30 days unless covered under warranty. needs_escalation=false.\n"
            "Q: 'Can I return an unopened product?' → A: Yes, 100% full refund with no restocking fee within 30 days. needs_escalation=false.\n"
            "Q: 'Can I return NovaBuds earbuds?' → A: No, once the hygiene seal is broken — unless proven defective on arrival. needs_escalation=false.\n"
            "Q: 'Can I return a gift card?' → A: No, gift cards are final sale and cannot be returned. needs_escalation=false.\n"
            "Q: 'Will I get a replacement or refund for my DOA item?' → A: Both options — customer's choice of full refund OR free replacement. needs_escalation=false.\n"
            "Q: 'My item was damaged in transit — what remedy do I get?' and context only says prepaid label provided but not what remedy → escalate.\n\n"

            "Respond strictly with the required JSON object."
        )

        messages.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )

        try:
            # Use standard OpenAI chat completion with JSON mode for Ollama.
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            content = response.choices[0].message.content or ""

            return self._parse_structured_output(content)

        except APIConnectionError as e:
            logger.error(
                f"Failed to connect to Ollama API at {self.base_url}: {e}"
            )

            raise RuntimeError(
                f"Cannot connect to local Ollama server at {self.base_url}. "
                "Ensure Ollama is running (`ollama serve` or Ollama background service)."
            ) from e

        except OpenAIError as e:
            logger.error(f"Ollama API error: {e}")
            return self._handle_error_fallback(str(e))

        except Exception as e:
            logger.error(
                f"Unexpected error during LLM generation: {e}",
                exc_info=True
            )
            return self._handle_error_fallback(str(e))

    def rewrite_query(
        self,
        user_question: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Rewrite the user question into a high-recall standalone search query.

        - With conversation history: incorporates prior context so follow-up
          questions resolve to self-contained queries.
        - Without conversation history (first turn): strips filler words and
          reformulates the query as concise topic keywords, improving cosine
          similarity matching against knowledge-base document chunks.
        """
        client = self._get_client()

        if not conversation_history:
            # First-turn: normalize the query into concise retrieval keywords
            # so that natural-language phrasings ("what is the return policy")
            # match KB chunks as reliably as keyword phrasings ("return policy").
            system_prompt = (
                "You are a search query optimization assistant for NovaTech Electronics. "
                "Your only job is to rewrite the user's question into 2-6 concise, "
                "high-recall search keywords suitable for semantic document retrieval. "
                "Remove filler words like 'what is', 'how do I', 'can you tell me', 'please'. "
                "Keep the core topic and any specific product/policy names. "
                "Do NOT answer the question. Output ONLY the rewritten search query, nothing else."
            )
            user_prompt = f"User question: {user_question}\nRewritten search query:"
        else:
            # Multi-turn: incorporate conversation context for follow-up resolution
            history_text = ""
            for msg in conversation_history[-4:]:
                if msg.role in {"user", "assistant"}:
                    history_text += f"{msg.role.capitalize()}: {msg.content}\n"

            if not history_text.strip():
                return user_question.strip()

            system_prompt = (
                "You are a search query rewriting assistant for NovaTech Electronics. "
                "Given the following conversation history and a new user question, "
                "rewrite the user question into a standalone search query that can be used to retrieve "
                "relevant documents from a knowledge base.\n"
                "If the new question depends on context (like a product name or topic) from the history, "
                "incorporate that context into the rewritten query.\n"
                "If the new question is completely unrelated to the history, just return the new question.\n"
                "Do NOT answer the question. Only output the standalone search query."
            )
            user_prompt = (
                f"--- CONVERSATION HISTORY ---\n{history_text}\n"
                f"--- NEW USER QUESTION ---\n{user_question}\n\n"
                "Rewritten standalone query:"
            )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            rewritten = response.choices[0].message.content
            if rewritten:
                # Clean up quotes if the model wraps it
                rewritten = rewritten.strip('"\' \n')
                logger.info(f"Query rewritten: '{user_question}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Failed to rewrite query via LLM: {e}. Falling back to original query.")

        return user_question.strip()

    def _parse_structured_output(
        self,
        raw_content: str
    ) -> LLMStructuredOutput:
        """Robust parser for model output, handling markdown blocks, JSON extraction, and type coercions."""

        cleaned = raw_content.strip()

        # Strip markdown fences if present.
        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*",
                "",
                cleaned
            )
            cleaned = re.sub(
                r"\s*```$",
                "",
                cleaned
            )
            cleaned = cleaned.strip()

        # Extract JSON object boundary if surrounded by other tokens.
        match = re.search(
            r"(\{.*\})",
            cleaned,
            re.DOTALL
        )

        json_str = match.group(1) if match else cleaned

        try:
            data = json.loads(json_str)

            if not isinstance(data, dict):
                raise ValueError(
                    "Parsed JSON root is not a dictionary."
                )

        except Exception as json_err:
            logger.warning(
                f"Direct JSON parse failed ({json_err}). "
                "Attempting fallback extraction..."
            )

            return self._fuzzy_extract_output(cleaned)

        # Sanitize and extract 'answer'.
        answer = data.get("answer")

        if not answer or not isinstance(answer, str):
            answer = (
                "I have reviewed your inquiry, but cannot formulate "
                "a complete answer from the available records."
            )

        # Sanitize and extract 'needs_escalation'.
        needs_escalation = data.get("needs_escalation")

        if isinstance(needs_escalation, str):
            needs_escalation = (
                needs_escalation.strip().lower()
                in {"true", "1", "yes"}
            )

        elif not isinstance(needs_escalation, bool):
            needs_escalation = (
                bool(needs_escalation)
                if needs_escalation is not None
                else False
            )

        # Sanitize and extract 'reason'.
        reason = data.get("reason")

        if reason is not None and not isinstance(reason, str):
            reason = str(reason)

        if reason == "":
            reason = None

        # Sanitize and extract 'used_sources'.
        used_sources_raw = data.get("used_sources", [])

        if isinstance(used_sources_raw, str):
            used_sources = (
                [used_sources_raw.strip()]
                if used_sources_raw.strip()
                else []
            )

        elif isinstance(used_sources_raw, list):
            used_sources = [
                str(source).strip()
                for source in used_sources_raw
                if source and str(source).strip()
            ]
        else:
            used_sources = []

        # Ensure consistency: if the answer text contains an explicit omission phrase
        # that the LLM forgot to pair with needs_escalation=True, promote it here.
        #
        # IMPORTANT — keep this list narrow and precise. Phrases such as
        # "support specialist will confirm" and "specialist will confirm" are
        # intentionally excluded because they appear in legitimate procedural
        # answers (e.g. "a specialist will confirm your refund once the item is
        # received") and caused false-positive escalations.  Only the four
        # "does not … whether" forms below are unambiguously indicative of a
        # missing-information omission that the LLM should have flagged.
        omission_indicators = [
            "does not specify whether",
            "does not state whether",
            "does not explicitly establish whether",
            "does not explicitly state whether",
        ]
        if not needs_escalation and any(ind in answer.lower() for ind in omission_indicators):
            needs_escalation = True
            if not reason:
                reason = "Available policy information does not establish the requested outcome."

        # If escalated but reason is missing, provide a descriptive reason.
        if needs_escalation and not reason:
            reason = "Available policy information does not establish the requested outcome."

        return LLMStructuredOutput(
            answer=answer,
            needs_escalation=needs_escalation,
            reason=reason,
            used_sources=used_sources
        )

    def _fuzzy_extract_output(
        self,
        text: str
    ) -> LLMStructuredOutput:
        """Extract output when model fails strict JSON formatting but returns plain text."""

        if not text:
            return self._handle_error_fallback(
                "Empty response content received."
            )

        return LLMStructuredOutput(
            answer=text,
            needs_escalation=False,
            reason=None,
            used_sources=[]
        )

    def _handle_error_fallback(
        self,
        error_msg: str
    ) -> LLMStructuredOutput:
        """Graceful fallback for unrecoverable errors."""

        return LLMStructuredOutput(
            answer=(
                "We encountered an issue processing your inquiry. "
                "A human support specialist will assist you."
            ),
            needs_escalation=True,
            reason="LLM response parsing failure.",
            used_sources=[]
        )