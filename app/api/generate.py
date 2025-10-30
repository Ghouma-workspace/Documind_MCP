import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

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


async def run_generate(template: str, report_topic: str = "default_report", context: str = "project summary", output_format: str = "txt") -> GenerateReportResponse:
    try:
        app_state = get_app_state()
        
        logger.info(f"Generating report: {template}")
        
        # Retrieve documents
        retrieval_msg = app_state.mcp_protocol.create_message(
            message_type=MessageType.RETRIEVAL_REQUEST,
            sender=AgentType.REASONER,
            receiver=AgentType.RETRIEVER,
            payload={
                "query": report_topic,
                "top_k": 10,
                "retrieval_mode": "hybrid"
            }
        )
        
        retrieval_response = await app_state.retriever_agent.process(retrieval_msg)
        documents = retrieval_response.payload.get("documents", [])
        
        # Generate report content
        content_parts = [doc["content"] for doc in documents[:5]]
        combined_content = "\n\n".join(content_parts)

        prompt = f"""Generate a professional report on: {report_topic}

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
        # Prepare metadata - ensure it's a dict or None
        metadata_dict = None
        if isinstance(context, dict):
            metadata_dict = context
        elif isinstance(context, str) and context:
            metadata_dict = {"description": context}
        
        automation_msg = app_state.mcp_protocol.create_message(
            message_type=MessageType.AUTOMATION_REQUEST,
            sender=AgentType.REASONER,
            receiver=AgentType.AUTOMATION,
            payload={
                "action": "fill_template",
                "template_name": template,
                "data": {
                    "title": report_topic,
                    "content": report_content,
                    "date": "Generated on " + str(Path(__file__).stat().st_mtime),
                    "metadata": metadata_dict
                },
                "output_format": output_format
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