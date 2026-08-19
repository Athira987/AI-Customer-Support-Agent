import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router as api_router
from app.core.config import settings

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("novatech_support")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Initializing NovaTech AI Customer Support Agent...")
    logger.info(f"LLM Model: {settings.openai_model} | Embedding Model: {settings.embedding_model}")
    logger.info(f"Vector Store Directory: {settings.chroma_persist_directory}")
    yield
    logger.info("Shutting down NovaTech AI Customer Support Agent.")


app = FastAPI(
    title="NovaTech AI Customer Support Agent",
    description=(
        "Production-ready RAG + Human Escalation Customer Support API for NovaTech Electronics. "
        "Built with FastAPI, ChromaDB, and OpenAI Python SDK."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Cross-Origin Resource Sharing (CORS) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Safety net exception handler to prevent leaking sensitive tracebacks."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please contact technical support."}
    )


# Register API routes
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
