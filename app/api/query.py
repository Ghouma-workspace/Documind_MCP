import logging
from typing import List, Optional, Dict, Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from aop.protocol import AgentType, MessageType, TaskType

logger = logging.getLogger(__name__)

# Helper function to get app state
def get_app_state():
    """Get application state from main module"""
    from app.main import app_state
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    return app_state

class QueryRequest(BaseModel):
    """Query request model"""
    question: str = Field(..., description="Question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: Optional[float] = None


async def run_query(question: str, top_k: int = 3) -> QueryResponse:
    """Core logic used by both API and ReasonerAgent - calls agents directly (NO MCP)"""
    app_state = get_app_state()
    logger.info(f"Running internal query: {question}")

    # Call retriever directly
    try:
        docs = await app_state.retriever_agent.retrieve(question, top_k=top_k)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Retrieval failed")

    if not docs:
        return QueryResponse(answer="No relevant documents found.", sources=[], confidence=0.0)

    context_parts = []
    sources = []
    for i, doc in enumerate(docs[:5], 1):
        context_parts.append(f"[Document {i}]\n{doc.content}...\n")
        sources.append({
            "id": doc.id,
            "file_name": doc.meta.get("file_name", "Unknown"),
            "score": doc.score if hasattr(doc, "score") else 0.0,
        })

    context = "\n".join(context_parts)
    prompt = f"""Based on the following context, answer the question concisely and accurately.

Context:
{context}

Question: {question}

Answer:"""

    # Call generator directly
    try:
        answer = await app_state.generator_agent.generate(
            prompt=prompt,
            context=context,
            max_length=300,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")

    return QueryResponse(answer=answer, sources=sources, confidence=None)