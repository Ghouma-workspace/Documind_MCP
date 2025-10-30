import logging
from typing import List, Optional, Dict, Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from mcp.protocol import AgentType, MessageType, TaskType

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
    try:
        app_state = get_app_state()
        
        logger.info(f"Summarize query: {request}")
        
        # Retrieve documents
        retrieval_msg = app_state.mcp_protocol.create_message(
            message_type=MessageType.RETRIEVAL_REQUEST,
            sender=AgentType.REASONER,
            receiver=AgentType.RETRIEVER,
            payload={
                "query": request,
                "top_k": top_k,
                "retrieval_mode": "hybrid"
            }
        )
        
        retrieval_response = await app_state.retriever_agent.process(retrieval_msg)
        
        if retrieval_response.message_type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail="Retrieval failed")
        
        documents = retrieval_response.payload.get("documents", [])
        
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
            content_parts.append(doc["content"])
            file_name = doc["meta"].get("file_name", "Unknown")
            if file_name not in sources:
                sources.append(file_name)
        
        combined_content = "\n\n".join(content_parts)
        
        # Generate summary
        prompt = f"""Provide a comprehensive summary of the following documents:

{combined_content[:4000]}

Create a well-structured summary that captures the key points, main themes, and important details.

Summary:"""
        
        generation_msg = app_state.mcp_protocol.create_message(
            message_type=MessageType.GENERATION_REQUEST,
            sender=AgentType.REASONER,
            receiver=AgentType.GENERATOR,
            payload={
                "prompt": prompt,
                "max_length": max_length,
                "temperature": 0.7,
                "task_type": "summarize"
            }
        )
        
        generation_response = await app_state.generator_agent.process(generation_msg)
        
        if generation_response.message_type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail="Generation failed")
        
        summary = generation_response.payload.get("generated_text", "Unable to generate summary")
        
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
    