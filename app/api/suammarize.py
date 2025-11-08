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


class SummarizeRequest(BaseModel):
    """Summarize request model"""
    query: Optional[str] = Field(default="summarize all documents", description="Summarization query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of documents to include")
    max_length: int = Field(default=512, ge=100, le=2048, description="Maximum summary length")


class SummarizeResponse(BaseModel):
    """Summarize response model"""
    summary: str
    documents_used: int
    sources: List[str]


async def run_summarization(request: str, top_k: int = 10, max_length: int = 512) -> SummarizeResponse:
    """Core summarization logic - calls agents directly (NO MCP)"""
    try:
        app_state = get_app_state()
        
        logger.info(f"Summarize query: {request}")
        
        # Retrieve documents directly
        documents = await app_state.retriever_agent.retrieve(request, top_k=top_k, mode="hybrid")
        
        if not documents:
            return SummarizeResponse(
                summary="No documents found to summarize.",
                documents_used=0,
                sources=[]
            )
        
        # Build content to summarize
        content_parts = []
        sources = []
        
        for doc in documents:
            content_parts.append(doc.content)
            file_name = doc.meta.get("file_name", "Unknown")
            if file_name not in sources:
                sources.append(file_name)
        
        combined_content = "\n\n".join(content_parts)
        
        # Generate summary directly
        prompt = f"""Provide a comprehensive summary of the following documents:

{combined_content[:4000]}

Create a well-structured summary that captures the key points, main themes, and important details.

Summary:"""
        
        summary = await app_state.generator_agent.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=0.7,
            task_type="summarize"
        )
        
        return SummarizeResponse(
            summary=summary,
            documents_used=len(documents),
            sources=sources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    