"""
Generator Agent - Handles text generation using Hugging Face LLMs
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from agents.base_agent import BaseAgent
from mcp.protocol import (
    MCPMessage,
    MCPProtocol,
    AgentType,
    MessageType,
    GenerationRequest,
    GenerationResponse
)
from pipelines.haystack_pipeline import HaystackPipeline

logger = logging.getLogger(__name__)


class GeneratorAgent(BaseAgent):
    """
    Generator Agent - Text generation and synthesis
    - Uses local Hugging Face models
    - Performs summarization, Q&A, content generation
    - Synthesizes information from retrieved documents
    """
    
    def __init__(self, protocol: MCPProtocol, haystack_pipeline: HaystackPipeline):
        super().__init__(AgentType.GENERATOR, "GeneratorAgent")
        self.protocol = protocol
        self.pipeline = haystack_pipeline
        self.generation_history = []
    
    async def process(self, message: MCPMessage) -> MCPMessage:
        """Process incoming generation request"""
        self.log_info(f"Processing message: {message.message_type}")
        
        try:
            if message.message_type == MessageType.GENERATION_REQUEST:
                return await self._handle_generation_request(message)
            else:
                return self._create_error_response(
                    message,
                    "Unsupported message type",
                    f"Cannot process message type: {message.message_type}"
                )
        except Exception as e:
            self.log_error(f"Error processing message: {str(e)}", exc_info=True)
            return self._create_error_response(message, "ProcessingError", str(e))
    
    async def _handle_generation_request(self, message: MCPMessage) -> MCPMessage:
        """Handle generation request"""
        try:
            # Parse request
            gen_req = GenerationRequest(**message.payload)
            
            self.log_info(f"Generating text for task: {gen_req.task_type}")
            self.log_debug(f"Prompt preview: {gen_req.prompt[:100]}...")
            
            start_time = datetime.utcnow()
            
            # Build full prompt based on context
            full_prompt = self._build_full_prompt(gen_req)
            
            # Generate text
            generated_text = self.pipeline.generate_with_hf(
                prompt=full_prompt,
                max_new_tokens=gen_req.max_length,
                temperature=gen_req.temperature,
                do_sample=True
            )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            self.log_info(f"Generated {len(generated_text)} characters in {execution_time:.2f}s")
            
            # Store in history
            self.generation_history.append({
                "task_type": gen_req.task_type,
                "prompt_length": len(full_prompt),
                "generated_length": len(generated_text),
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            })
            
            # Create response
            gen_resp = GenerationResponse(
                generated_text=generated_text,
                prompt=gen_req.prompt[:200] + "..." if len(gen_req.prompt) > 200 else gen_req.prompt,
                model_used=self.pipeline.model_name,
                tokens_generated=len(generated_text.split())
            )
            
            return self.protocol.create_message(
                message_type=MessageType.GENERATION_RESPONSE,
                sender=self.agent_type,
                receiver=message.sender,
                payload=gen_resp.model_dump(),
                parent_message_id=message.message_id,
                metadata={"execution_time": execution_time}
            )
            
        except Exception as e:
            self.log_error(f"Error handling generation request: {str(e)}", exc_info=True)
            return self._create_error_response(message, "GenerationError", str(e))
    
    def _build_full_prompt(self, gen_req: GenerationRequest) -> str:
        """Build full prompt with context"""
        prompt_parts = []
        
        # Add system instruction based on task type
        system_instruction = self._get_system_instruction(gen_req.task_type)
        if system_instruction:
            prompt_parts.append(system_instruction)
        
        # Add context if available
        if gen_req.context:
            prompt_parts.append(f"\nContext:\n{gen_req.context}\n")
        
        # Add main prompt
        prompt_parts.append(gen_req.prompt)
        
        return "\n".join(prompt_parts)
    
    def _get_system_instruction(self, task_type: str) -> str:
        """Get system instruction based on task type"""
        instructions = {
            "summarize": "You are a helpful AI assistant that creates clear, concise summaries of documents.",
            "answer": "You are a helpful AI assistant that answers questions based on provided context.",
            "generate": "You are a helpful AI assistant that generates structured content.",
            "report": "You are a professional report writer that creates comprehensive, well-structured reports.",
            "extract": "You are a data extraction specialist that identifies and extracts specific information.",
            "general": "You are a helpful AI assistant."
        }
        return instructions.get(task_type, instructions["general"])
    
    def summarize_text(self, text: str, max_length: int = 512) -> str:
        """Convenience method to summarize text"""
        self.log_info(f"Summarizing text of length {len(text)}")
        
        prompt = f"""Summarize the following text concisely:

{text}

Summary:"""
        
        summary = self.pipeline.generate_with_hf(
            prompt=prompt,
            max_new_tokens=max_length,
            temperature=0.7
        )
        
        return summary
    
    def answer_question(self, question: str, context: str, max_length: int = 300) -> str:
        """Convenience method to answer a question"""
        self.log_info(f"Answering question: {question}")
        
        prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.pipeline.generate_with_hf(
            prompt=prompt,
            max_new_tokens=max_length,
            temperature=0.7
        )
        
        return answer
    
    def generate_from_template(self, template: str, data: Dict[str, Any]) -> str:
        """Generate content from template and data"""
        self.log_info("Generating from template")
        
        # Simple template variable replacement
        content = template
        for key, value in data.items():
            placeholder = f"{{{key}}}"
            content = content.replace(placeholder, str(value))
        
        # If there are still unfilled placeholders, use generation
        if "{" in content and "}" in content:
            prompt = f"""Complete the following document by filling in the missing information:

{content}

Completed document:"""
            
            completed = self.pipeline.generate_with_hf(
                prompt=prompt,
                max_new_tokens=512,
                temperature=0.5
            )
            return completed
        
        return content
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        if not self.generation_history:
            return {
                "total_generations": 0,
                "avg_execution_time": 0,
                "total_tokens_generated": 0
            }
        
        total_time = sum(h["execution_time"] for h in self.generation_history)
        total_tokens = sum(h["generated_length"] for h in self.generation_history)
        
        return {
            "total_generations": len(self.generation_history),
            "avg_execution_time": total_time / len(self.generation_history),
            "total_tokens_generated": total_tokens,
            "recent_tasks": [h["task_type"] for h in self.generation_history[-10:]]
        }
    
    def _create_error_response(self, original_message: MCPMessage, error_type: str, error_message: str) -> MCPMessage:
        """Create error response"""
        return self.protocol.create_error_response(
            sender=self.agent_type,
            receiver=original_message.sender,
            error_type=error_type,
            error_message=error_message,
            parent_message_id=original_message.message_id,
            recoverable=True
        )
