import logging
from fastapi import APIRouter, HTTPException, Depends, status
from app.models.schemas import HealthResponse, ChatRequest, SupportResponse
from app.services.rag import RAGPipeline
from app.services.retrieval import RetrievalService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection for RAG Pipeline
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    tags=["System"]
)
def health_check() -> HealthResponse:
    """Return health status of the customer support agent service."""
    return HealthResponse(
        status="ok",
        service="NovaTech AI Customer Support Agent",
        version="1.0.0"
    )


@router.post(
    "/chat",
    response_model=SupportResponse,
    status_code=status.HTTP_200_OK,
    summary="Customer Support Chat Endpoint",
    tags=["Chat & Support"]
)
def chat_endpoint(
    request: ChatRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline)
) -> SupportResponse:
    """
    Process incoming customer query through RAG pipeline.
    
    1. Embeds customer message
    2. Searches ChromaDB for relevant knowledge base context
    3. Synthesizes answer using OpenAI LLM
    4. Evaluates whether human escalation is required
    5. Returns grounded answer with cited source documents
    """
    try:
        response = rag.process_chat(request)
        return response
    except ValueError as e:
        logger.warning(f"Validation error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Internal server error processing chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your support request."
        )


@router.get(
    "/knowledge-base/status",
    summary="Knowledge Base Status",
    tags=["System"]
)
def kb_status() -> dict:
    """Return status and document chunk count in ChromaDB vector store."""
    try:
        retrieval = RetrievalService()
        count = retrieval.count()
        return {
            "collection_name": settings.chroma_collection_name,
            "total_chunks_indexed": count,
            "openai_model": settings.openai_model,
            "embedding_model": settings.embedding_model,
            "status": "ready" if count > 0 else "needs_ingestion"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
