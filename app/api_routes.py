"""
API Routes for DocuMind
Defines all API endpoints
"""

import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.query import QueryRequest, QueryResponse, run_query
from app.api.suammarize import SummarizeRequest, SummarizeResponse, run_summarization
from mcp.protocol import AgentType, MessageType, TaskType

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")


class ChatResponse(BaseModel):
    """Chat response model"""
    message: str
    intent: str
    confidence: float
    actions_taken: List[str]
    documents_used: int
    conversation_id: Optional[str] = None


class GenerateReportRequest(BaseModel):
    """Generate report request model"""
    template: str = Field(default="default_report", description="Template name")
    report_topic: Optional[str] = Field(default="project summary", description="Report topic")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    output_format: str = Field(default="markdown", description="Output format (markdown, txt, json)")


class GenerateReportResponse(BaseModel):
    """Generate report response model"""
    success: bool
    output_path: Optional[str] = None
    report_preview: Optional[str] = None


class UploadResponse(BaseModel):
    """Upload response model"""
    success: bool
    filename: str
    message: str
    file_path: Optional[str] = None


# Helper function to get app state
def get_app_state():
    """Get application state from main module"""
    from app.main import app_state
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")
    return app_state


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document
    
    Supports: PDF, DOCX, TXT
    """
    try:
        app_state = get_app_state()
        
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save file to documents directory
        from config import settings
        file_path = settings.documents_path / file.filename
        
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"File uploaded: {file.filename}")
        
        # Ingest document
        success = app_state.retriever_agent.ingest_document(str(file_path))
        
        if success:
            return UploadResponse(
                success=True,
                filename=file.filename,
                message=f"Document uploaded and indexed successfully",
                file_path=str(file_path)
            )
        else:
            return UploadResponse(
                success=False,
                filename=file.filename,
                message="Document uploaded but failed to index"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)):
    """
    Chat with the AI assistant
    
    The assistant can:
    - Answer questions about documents
    - Provide summaries
    - Generate reports
    - Have general conversation
    
    Intent is automatically detected from your message.
    """
    try:
        app_state = get_app_state()
        
        logger.info(f"Processing chat message: {request.message[:50]}...")
        
        # Create MCP chat message
        chat_msg = app_state.mcp_protocol.create_message(
            message_type=MessageType.CHAT_REQUEST,
            sender=AgentType.AUTOMATION,
            receiver=AgentType.REASONER,
            payload={
                "message": request.message,
                "conversation_id": request.conversation_id,
                "context": {}
            }
        )
        
        # Process through reasoner
        response_msg = await app_state.reasoner_agent.process(chat_msg)
        
        # Extract response
        payload = response_msg.payload
        
        return ChatResponse(
            message=payload.get("message", "I'm not sure how to respond to that."),
            intent=payload.get("intent", "unknown"),
            confidence=payload.get("confidence", 0.0),
            actions_taken=payload.get("actions_taken", []),
            documents_used=payload.get("documents_used", 0),
            conversation_id=payload.get("conversation_id")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest = Body(...)):
    """
    Query documents and get an answer
    
    Uses hybrid retrieval and local LLM for answering
    """
    return await run_query(request.question, request.top_k)


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_documents(request: SummarizeRequest = Body(...)):
    """
    Summarize documents based on query
    
    Retrieves relevant documents and generates a comprehensive summary
    """
    return await run_summarization(request)


@router.post("/generate-report", response_model=GenerateReportResponse)
async def generate_report(request: GenerateReportRequest = Body(...)):
    """
    Generate a report from documents
    
    Retrieves relevant information and fills a template
    """
    try:
        app_state = get_app_state()
        
        logger.info(f"Generating report: {request.template}")
        
        # Retrieve documents
        retrieval_msg = app_state.mcp_protocol.create_message(
            message_type=MessageType.RETRIEVAL_REQUEST,
            sender=AgentType.REASONER,
            receiver=AgentType.RETRIEVER,
            payload={
                "query": request.report_topic,
                "top_k": 10,
                "retrieval_mode": "hybrid"
            }
        )
        
        retrieval_response = await app_state.retriever_agent.process(retrieval_msg)
        documents = retrieval_response.payload.get("documents", [])
        
        # Generate report content
        content_parts = [doc["content"] for doc in documents[:5]]
        combined_content = "\n\n".join(content_parts)
        
        prompt = f"""Generate a professional report on: {request.report_topic}

Based on the following information:
{combined_content}

Create a structured report with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Recommendations

Report:"""
        
        generation_msg = app_state.mcp_protocol.create_message(
            message_type=MessageType.GENERATION_REQUEST,
            sender=AgentType.REASONER,
            receiver=AgentType.GENERATOR,
            payload={
                "prompt": prompt,
                "max_length": 1024,
                "temperature": 0.7,
                "task_type": "report"
            }
        )
        
        generation_response = await app_state.generator_agent.process(generation_msg)
        report_content = generation_response.payload.get("generated_text", "")
        
        # Fill template
        automation_msg = app_state.mcp_protocol.create_message(
            message_type=MessageType.AUTOMATION_REQUEST,
            sender=AgentType.REASONER,
            receiver=AgentType.AUTOMATION,
            payload={
                "action": "fill_template",
                "template_name": request.template,
                "data": {
                    "title": request.report_topic,
                    "content": report_content,
                    "date": "Generated on " + str(Path(__file__).stat().st_mtime),
                    "metadata": request.context
                },
                "output_format": request.output_format
            }
        )
        
        automation_response = await app_state.automation_agent.process(automation_msg)
        
        if automation_response.message_type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail="Report generation failed")
        
        payload = automation_response.payload
        
        return GenerateReportResponse(
            success=payload.get("success", False),
            output_path=payload.get("output_path"),
            report_preview=report_content + "..." if len(report_content) > 500 else report_content
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    """List all indexed documents"""
    try:
        app_state = get_app_state()
        doc_count = app_state.retriever_agent.get_document_count()
        
        from config import settings
        documents_dir = settings.documents_path
        
        files = []
        if documents_dir.exists():
            for file_path in documents_dir.glob('*'):
                if file_path.is_file():
                    files.append({
                        "filename": file_path.name,
                        "size": file_path.stat().st_size,
                        "path": str(file_path)
                    })
        
        return {
            "total_indexed": doc_count,
            "files": files
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/outputs")
async def list_outputs():
    """List all generated outputs"""
    try:
        app_state = get_app_state()
        outputs = app_state.automation_agent.list_outputs()
        
        return {
            "total": len(outputs),
            "outputs": outputs
        }
        
    except Exception as e:
        logger.error(f"Error listing outputs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/outputs/{filename}")
async def download_output(filename: str):
    """Download a generated output file"""
    try:
        from config import settings
        file_path = settings.outputs_path / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def list_templates():
    """List available templates"""
    try:
        app_state = get_app_state()
        templates = app_state.automation_agent.list_templates()
        
        return {
            "total": len(templates),
            "templates": templates
        }
        
    except Exception as e:
        logger.error(f"Error listing templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
