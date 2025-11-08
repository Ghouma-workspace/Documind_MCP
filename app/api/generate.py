import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

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
    """Core report generation logic - calls agents directly (NO MCP)"""
    try:
        app_state = get_app_state()
        
        logger.info(f"Generating report: {template}")
        
        # Retrieve documents directly
        documents = await app_state.retriever_agent.retrieve(report_topic, top_k=10, mode="hybrid")
        
        # Generate report content directly
        content_parts = [doc.content for doc in documents[:5]]
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
        
        report_content = await app_state.generator_agent.generate(
            prompt=prompt,
            max_length=1024,
            temperature=0.7,
            task_type="report"
        )
        
        # Fill template directly
        metadata_dict = None
        if isinstance(context, dict):
            metadata_dict = context
        elif isinstance(context, str) and context:
            metadata_dict = {"description": context}
        
        result = await app_state.automation_agent.fill_and_save(
            template_name=template,
            data={
                "title": report_topic,
                "content": report_content,
                "date": "Generated on " + str(Path(__file__).stat().st_mtime),
                "metadata": metadata_dict
            },
            output_format=output_format
        )
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail="Report generation failed")
        
        return GenerateReportResponse(
            success=result["success"],
            output_path=result.get("output_path"),
            report_preview=report_content[:500] + "..." if len(report_content) > 500 else report_content
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))