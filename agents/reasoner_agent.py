"""
Reasoner Agent - Orchestrates tasks and delegates to other agents
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from agents.base_agent import BaseAgent
from aop.protocol import (
    AOPMessage,
    AOPProtocol,
    AgentType,
    MessageType,
    TaskType,
    TaskRequest,
    TaskResponse,
    ChatRequest,
    ChatResponse
)

logger = logging.getLogger(__name__)


class ReasonerAgent(BaseAgent):
    """
    Reasoner Agent - The brain of the system
    - Interprets user requests
    - Decides which tasks to trigger
    - Delegates to appropriate agents
    - Coordinates multi-step workflows
    """
    
    def __init__(self, protocol: AOPProtocol, router=None):
        super().__init__(AgentType.REASONER, "ReasonerAgent")
        self.protocol = protocol
        self.router = router  # Router reference for sending messages
        self.task_history = []
        self.conversation_history = []
    
    async def process(self, message: AOPMessage) -> AOPMessage:
        """Process incoming message and orchestrate workflow"""
        self.log_info(f"Processing message: {message.message_type}")
        
        try:
            if message.message_type == MessageType.CHAT_REQUEST:
                return await self._handle_chat_request(message)
            elif message.message_type == MessageType.TASK_REQUEST:
                return await self._handle_task_request(message)
            elif message.message_type == MessageType.TASK_RESPONSE:
                return await self._handle_task_response(message)
            else:
                return self._create_error_response(
                    message,
                    "Unsupported message type",
                    f"Cannot process message type: {message.message_type}"
                )
        except Exception as e:
            self.log_error(f"Error processing message: {str(e)}", exc_info=True)
            return self._create_error_response(message, "ProcessingError", str(e))
    
    async def _handle_task_request(self, message: AOPMessage) -> AOPMessage:
        """Handle incoming task request"""
        try:
            task_request = TaskRequest(**message.payload)
            self.log_info(f"Handling task: {task_request.task_type}")
            
            # Determine execution plan
            execution_plan = self._create_execution_plan(task_request)
            
            # Store in history
            self.task_history.append({
                "message_id": message.message_id,
                "task_type": task_request.task_type,
                "timestamp": datetime.utcnow().isoformat(),
                "execution_plan": execution_plan
            })
            
            # Execute plan
            result = await self._execute_plan(execution_plan, message)
            
            # Create response
            response = TaskResponse(
                task_type=task_request.task_type,
                result=result,
                success=True,
                execution_time=None,
                metadata={"execution_plan": execution_plan}
            )
            
            return self.protocol.create_message(
                message_type=MessageType.TASK_RESPONSE,
                sender=self.agent_type,
                receiver=message.sender,
                payload=response.model_dump(),
                parent_message_id=message.message_id
            )
            
        except Exception as e:
            self.log_error(f"Error handling task request: {str(e)}", exc_info=True)
            return self._create_error_response(message, "TaskError", str(e))
    
    def _create_execution_plan(self, task_request: TaskRequest) -> Dict[str, Any]:
        """Create execution plan based on task type"""
        task_type = task_request.task_type
        parameters = task_request.parameters
        
        plan = {
            "task_type": task_type,
            "steps": []
        }
        
        if task_type == TaskType.SUMMARIZE:
            # Summarization workflow: Retrieve -> Generate
            plan["steps"] = [
                {
                    "step": 1,
                    "action": "retrieve",
                    "agent": AgentType.RETRIEVER,
                    "params": {
                        "query": parameters.get("query", "summarize all documents"),
                        "top_k": parameters.get("top_k", 10),
                        "mode": parameters.get("retrieval_mode", "hybrid")
                    }
                },
                {
                    "step": 2,
                    "action": "generate",
                    "agent": AgentType.GENERATOR,
                    "params": {
                        "task": "summarize",
                        "max_length": parameters.get("max_length", 512),
                        "temperature": parameters.get("temperature", 0.7)
                    }
                }
            ]
        
        elif task_type == TaskType.QUERY:
            # Query workflow: Retrieve -> Generate answer
            plan["steps"] = [
                {
                    "step": 1,
                    "action": "retrieve",
                    "agent": AgentType.RETRIEVER,
                    "params": {
                        "query": parameters.get("question", ""),
                        "top_k": parameters.get("top_k", 5),
                        "mode": "hybrid"
                    }
                },
                {
                    "step": 2,
                    "action": "generate",
                    "agent": AgentType.GENERATOR,
                    "params": {
                        "task": "answer",
                        "question": parameters.get("question", ""),
                        "max_length": parameters.get("max_length", 300)
                    }
                }
            ]
        
        elif task_type == TaskType.GENERATE_REPORT:
            # Report generation: Retrieve -> Generate -> Fill template
            plan["steps"] = [
                {
                    "step": 1,
                    "action": "retrieve",
                    "agent": AgentType.RETRIEVER,
                    "params": {
                        "query": parameters.get("report_topic", "project summary"),
                        "top_k": 10,
                        "mode": "hybrid"
                    }
                },
                {
                    "step": 2,
                    "action": "generate",
                    "agent": AgentType.GENERATOR,
                    "params": {
                        "task": "report",
                        "max_length": 1024
                    }
                },
                {
                    "step": 3,
                    "action": "automate",
                    "agent": AgentType.AUTOMATION,
                    "params": {
                        "action": "fill_template",
                        "template_name": parameters.get("template", "default_report"),
                        "output_format": parameters.get("format", "markdown")
                    }
                }
            ]
        
        elif task_type == TaskType.EXTRACT:
            # Extraction workflow: Retrieve -> Generate extraction
            plan["steps"] = [
                {
                    "step": 1,
                    "action": "retrieve",
                    "agent": AgentType.RETRIEVER,
                    "params": {
                        "query": parameters.get("extraction_query", ""),
                        "top_k": parameters.get("top_k", 5),
                        "mode": "semantic"
                    }
                },
                {
                    "step": 2,
                    "action": "generate",
                    "agent": AgentType.GENERATOR,
                    "params": {
                        "task": "extract",
                        "extraction_fields": parameters.get("fields", [])
                    }
                }
            ]
        
        else:
            # Default: just retrieve
            plan["steps"] = [
                {
                    "step": 1,
                    "action": "retrieve",
                    "agent": AgentType.RETRIEVER,
                    "params": parameters
                }
            ]
        
        self.log_debug(f"Created execution plan with {len(plan['steps'])} steps")
        return plan
    
    async def _execute_plan(self, plan: Dict[str, Any], original_message: AOPMessage) -> Dict[str, Any]:
        """Execute the planned workflow by delegating to agents via MCP"""
        results = {
            "steps_completed": 0,
            "total_steps": len(plan["steps"]),
            "outputs": []
        }
        
        context = {}
        
        for step_config in plan["steps"]:
            step_num = step_config["step"]
            action = step_config["action"]
            agent = step_config["agent"]
            params = step_config["params"]
            
            self.log_info(f"Executing step {step_num}: {action} via {agent}")
            
            try:
                # Add context from previous steps
                if context:
                    params["context"] = context
                
                # Create message to agent based on action type
                if action == "retrieve":
                    agent_message = self._create_retrieval_message(agent, params, original_message.message_id)
                elif action == "generate":
                    agent_message = self._create_generation_message(agent, params, context, original_message.message_id)
                elif action == "automate":
                    agent_message = self._create_automation_message(agent, params, context, original_message.message_id)
                else:
                    self.log_warning(f"Unknown action: {action}")
                    continue
                
                # Send message via router if available, otherwise add to protocol
                if self.router:
                    # Send via router and wait for response
                    response = await self.router.process_and_wait(agent_message, timeout=30.0)
                    
                    if response:
                        # Store the result from the response
                        step_result = {
                            "step": step_num,
                            "action": action,
                            "agent": agent,
                            "message_id": agent_message.message_id,
                            "response_id": response.message_id,
                            "status": "completed",
                            "result": response.payload
                        }
                        
                        # Update context with the actual response for next step
                        context[f"step_{step_num}_result"] = response.payload
                        context[f"step_{step_num}_message_id"] = response.message_id
                    else:
                        # Timeout or no response
                        step_result = {
                            "step": step_num,
                            "action": action,
                            "agent": agent,
                            "message_id": agent_message.message_id,
                            "status": "timeout",
                            "error": "No response received within timeout"
                        }
                        self.log_warning(f"Step {step_num} timed out")
                else:
                    # No router - just create the message in protocol (backward compatibility)
                    step_result = {
                        "step": step_num,
                        "action": action,
                        "agent": agent,
                        "message_id": agent_message.message_id,
                        "status": "delegated"
                    }
                    context[f"step_{step_num}_result"] = step_result
                
                results["outputs"].append(step_result)
                results["steps_completed"] += 1
                
            except Exception as e:
                self.log_error(f"Error executing step {step_num}: {str(e)}")
                results["outputs"].append({
                    "step": step_num,
                    "error": str(e),
                    "status": "failed"
                })
        
        return results
    
    def _create_retrieval_message(self, receiver: AgentType, params: Dict, parent_id: str) -> AOPMessage:
        """Create retrieval request message"""
        from aop.protocol import RetrievalRequest
        
        retrieval_req = RetrievalRequest(
            query=params.get("query", ""),
            top_k=params.get("top_k", 5),
            retrieval_mode=params.get("mode", "hybrid")
        )
        
        return self.protocol.create_message(
            message_type=MessageType.RETRIEVAL_REQUEST,
            sender=self.agent_type,
            receiver=receiver,
            payload=retrieval_req.model_dump(),
            parent_message_id=parent_id
        )
    
    def _create_generation_message(self, receiver: AgentType, params: Dict, context: Dict, parent_id: str) -> AOPMessage:
        """Create generation request message"""
        from aop.protocol import GenerationRequest
        
        # Build prompt based on context
        prompt = self._build_prompt(params, context)
        
        gen_req = GenerationRequest(
            prompt=prompt,
            context=str(context),
            max_length=params.get("max_length", 512),
            temperature=params.get("temperature", 0.7),
            task_type=params.get("task", "general")
        )
        
        return self.protocol.create_message(
            message_type=MessageType.GENERATION_REQUEST,
            sender=self.agent_type,
            receiver=receiver,
            payload=gen_req.model_dump(),
            parent_message_id=parent_id
        )
    
    def _create_automation_message(self, receiver: AgentType, params: Dict, context: Dict, parent_id: str) -> AOPMessage:
        """Create automation request message"""
        from aop.protocol import AutomationRequest
        
        auto_req = AutomationRequest(
            action=params.get("action", "save_file"),
            template_name=params.get("template_name"),
            data=context,
            output_format=params.get("output_format", "markdown"),
            output_path=params.get("output_path")
        )
        
        return self.protocol.create_message(
            message_type=MessageType.AUTOMATION_REQUEST,
            sender=self.agent_type,
            receiver=receiver,
            payload=auto_req.model_dump(),
            parent_message_id=parent_id
        )
    
    def _build_prompt(self, params: Dict, context: Dict) -> str:
        """Build prompt for generation based on task and context"""
        task = params.get("task", "general")
        
        if task == "summarize":
            prompt = "Summarize the following documents:\n\n"
            if "step_1_result" in context:
                prompt += "[Retrieved documents will be inserted here]\n\n"
            prompt += "Provide a comprehensive summary."
        
        elif task == "answer":
            question = params.get("question", "")
            prompt = f"Question: {question}\n\nContext:\n[Retrieved documents]\n\nAnswer:"
        
        elif task == "report":
            prompt = "Generate a detailed project report based on the following information:\n\n"
            prompt += "[Retrieved documents]\n\n"
            prompt += "Include: Executive Summary, Key Findings, Recommendations"
        
        elif task == "extract":
            fields = params.get("extraction_fields", [])
            prompt = f"Extract the following fields from the documents: {', '.join(fields)}\n\n"
            prompt += "[Documents]\n\nProvide structured output."
        
        else:
            prompt = "Process the following information:\n\n[Context]"
        
        return prompt
    
    async def _handle_chat_request(self, message: AOPMessage) -> AOPMessage:
        """Handle incoming chat request with intent detection"""
        try:
            chat_request = ChatRequest(**message.payload)
            user_message = chat_request.message.lower().strip()
            
            self.log_info(f"Processing chat: '{user_message[:50]}...'")
            
            # Detect intent
            intent, confidence = self._detect_intent(user_message)
            self.log_info(f"Detected intent: {intent} (confidence: {confidence:.2f})")
            
            # Store in conversation history
            self.conversation_history.append({
                "message": chat_request.message,
                "intent": intent,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Handle based on intent
            if intent == "greeting":
                response_text, actions = await self._handle_greeting(user_message)
                docs_used = 0
            
            elif intent == "question":
                response_text, actions, docs_used = await self._handle_question(user_message)
            
            elif intent == "summarize":
                response_text, actions, docs_used = await self._handle_summarize(user_message)
            
            elif intent == "report":
                response_text, actions, docs_used = await self._handle_report_request(user_message)
            
            else:
                response_text = "I understand you want to chat, but I'm not sure how to help. I can answer questions about your documents, create summaries, or generate reports. What would you like me to do?"
                actions = []
                docs_used = 0
            
            # Create response
            chat_response = ChatResponse(
                message=response_text,
                intent=intent,
                confidence=confidence,
                actions_taken=actions,
                documents_used=docs_used,
                conversation_id=chat_request.conversation_id
            )
            
            return self.protocol.create_message(
                message_type=MessageType.CHAT_RESPONSE,
                sender=self.agent_type,
                receiver=message.sender,
                payload=chat_response.model_dump(),
                parent_message_id=message.message_id
            )
            
        except Exception as e:
            self.log_error(f"Error handling chat request: {str(e)}", exc_info=True)
            return self._create_error_response(message, "ChatError", str(e))
    
    def _detect_intent(self, user_message: str) -> tuple[str, float]:
        """Detect user intent from message"""
        msg = user_message.lower().strip()
        
        # Greeting patterns
        greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
        if any(keyword in msg for keyword in greeting_keywords):
            return "greeting", 0.95
        
        # Question patterns
        question_keywords = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'can you tell', 'do you know']
        question_indicators = msg.endswith('?') or any(keyword in msg for keyword in question_keywords)
        if question_indicators and len(msg.split()) > 3:
            return "question", 0.90
        
        # Summary patterns
        summary_keywords = ['summarize', 'summary', 'overview', 'brief', 'recap', 'review', 'main points']
        if any(keyword in msg for keyword in summary_keywords):
            return "summarize", 0.85
        
        # Report generation patterns
        report_keywords = ['report', 'generate report', 'create report', 'full report', 'detailed report']
        if any(keyword in msg for keyword in report_keywords):
            return "report", 0.85
        
        # Default to question if ends with ?
        if msg.endswith('?'):
            return "question", 0.70
        
        # Default to general chat
        return "general", 0.50
    
    async def _handle_greeting(self, user_message: str) -> tuple[str, list]:
        """Handle greeting messages"""
        greetings = [
            "Hello! I'm your document assistant. I can help you find information in your documents, create summaries, or generate reports. What would you like to know?",
            "Hi there! I'm here to help you with your documents. You can ask me questions, request summaries, or generate reports. How can I assist you today?",
            "Greetings! I'm your AI document assistant. I can answer questions about your documents, provide summaries, or create detailed reports. What do you need?"
        ]
        
        # Simple variety in responses
        import random
        response = random.choice(greetings)
        
        return response, ["greeting_acknowledged"]
    
    async def _handle_question(self, user_message: str) -> tuple[str, list, int]:
        """Handle question about documents using AOP router"""
        self.log_info("Processing question with document retrieval via MCP")
        
        if not self.router:
            self.log_error("Router not available, cannot delegate")
            return "Sorry, I cannot process questions right now.", ["error"], 0
        
        # Step 1: Retrieve documents via MCP
        retrieval_msg = self.protocol.create_message(
            message_type=MessageType.RETRIEVAL_REQUEST,
            sender=self.agent_type,
            receiver=AgentType.RETRIEVER,
            payload={"query": user_message, "top_k": 3, "retrieval_mode": "hybrid"}
        )
        
        retrieval_response = await self.router.process_and_wait(retrieval_msg, timeout=30.0)
        
        # Check for timeout (None response)
        if retrieval_response is None:
            self.log_error("Timeout waiting for retrieval response")
            return "Request timed out. Please try again.", ["timeout"], 0
        
        if retrieval_response.message_type == MessageType.ERROR:
            return "I encountered an error retrieving documents.", ["retrieval_failed"], 0
        
        docs = retrieval_response.payload.get("documents", [])
        if not docs:
            return "No relevant documents found.", ["no_documents"], 0
        
        # Build context
        context_parts = [f"[Document {i+1}]\n{doc['content']}...\n" for i, doc in enumerate(docs[:5])]
        context = "\n".join(context_parts)
        
        # Step 2: Generate answer via MCP
        prompt = f"""Based on the following context, answer the question concisely and accurately.

Context:
{context}

Question: {user_message}

Answer:"""
        
        generation_msg = self.protocol.create_message(
            message_type=MessageType.GENERATION_REQUEST,
            sender=self.agent_type,
            receiver=AgentType.GENERATOR,
            payload={
                "prompt": prompt,
                "context": context,
                "max_length": 300,
                "temperature": 0.7,
                "task_type": "answer"
            }
        )
        
        generation_response = await self.router.process_and_wait(generation_msg, timeout=30.0)
        
        # Check for timeout
        if generation_response is None:
            self.log_error("Timeout waiting for generation response")
            return "Request timed out while generating answer. Please try again.", ["timeout"], len(docs)
        
        if generation_response.message_type == MessageType.ERROR:
            return "I encountered an error generating an answer.", ["generation_failed"], len(docs)
        
        answer = generation_response.payload.get("generated_text", "Unable to generate answer")
        return answer, ["retrieve_documents", "generate_answer"], len(docs)

    
    async def _handle_summarize(self, user_message: str) -> tuple[str, list, int]:
        """Handle summarization request using AOP router"""
        self.log_info("Processing summarization request via MCP")
        
        if not self.router:
            self.log_error("Router not available, cannot delegate")
            return "Sorry, I cannot process summaries right now.", ["error"], 0
        
        # Step 1: Retrieve documents via MCP
        retrieval_msg = self.protocol.create_message(
            message_type=MessageType.RETRIEVAL_REQUEST,
            sender=self.agent_type,
            receiver=AgentType.RETRIEVER,
            payload={"query": user_message, "top_k": 10, "retrieval_mode": "hybrid"}
        )
        
        retrieval_response = await self.router.process_and_wait(retrieval_msg, timeout=30.0)
        
        if retrieval_response is None:
            self.log_error("Timeout waiting for retrieval response")
            return "Request timed out. Please try again.", ["timeout"], 0
        
        if retrieval_response.message_type == MessageType.ERROR:
            return "I encountered an error retrieving documents.", ["retrieval_failed"], 0
        
        docs = retrieval_response.payload.get("documents", [])
        if not docs:
            return "No documents found to summarize.", ["no_documents"], 0
        
        # Build content to summarize
        content_parts = [doc["content"] for doc in docs]
        combined_content = "\n\n".join(content_parts)
        
        # Step 2: Generate summary via MCP
        prompt = f"""Provide a comprehensive summary of the following documents:

{combined_content[:4000]}

Create a well-structured summary that captures the key points, main themes, and important details.

Summary:"""
        
        generation_msg = self.protocol.create_message(
            message_type=MessageType.GENERATION_REQUEST,
            sender=self.agent_type,
            receiver=AgentType.GENERATOR,
            payload={
                "prompt": prompt,
                "max_length": 512,
                "temperature": 0.7,
                "task_type": "summarize"
            }
        )
        
        generation_response = await self.router.process_and_wait(generation_msg, timeout=30.0)
        
        if generation_response is None:
            self.log_error("Timeout waiting for generation response")
            return "Request timed out while generating summary. Please try again.", ["timeout"], len(docs)
        
        if generation_response.message_type == MessageType.ERROR:
            return "I encountered an error generating a summary.", ["generation_failed"], len(docs)
        
        summary = generation_response.payload.get("generated_text", "Unable to generate summary")
        return summary, ["retrieve_all_documents", "generate_summary"], len(docs)

    
    async def _handle_report_request(self, topic: str) -> tuple[str, list, int]:
        """Handle report generation request using AOP router"""
        self.log_info("Processing report generation request via MCP")
        
        if not self.router:
            self.log_error("Router not available, cannot delegate")
            return "Sorry, I cannot generate reports right now.", ["error"], 0
        
        # Step 1: Retrieve documents via MCP
        retrieval_msg = self.protocol.create_message(
            message_type=MessageType.RETRIEVAL_REQUEST,
            sender=self.agent_type,
            receiver=AgentType.RETRIEVER,
            payload={"query": topic, "top_k": 10, "retrieval_mode": "hybrid"}
        )
        
        retrieval_response = await self.router.process_and_wait(retrieval_msg, timeout=30.0)
        
        if retrieval_response is None:
            self.log_error("Timeout waiting for retrieval response")
            return "Request timed out. Please try again.", ["timeout"], 0
        
        if retrieval_response.message_type == MessageType.ERROR:
            return "I encountered an error retrieving documents.", ["retrieval_failed"], 0
        
        docs = retrieval_response.payload.get("documents", [])
        
        # Build content
        content_parts = [doc["content"] for doc in docs[:5]]
        combined_content = "\n\n".join(content_parts)
        
        # Step 2: Generate report content via MCP
        prompt = f"""Generate a professional report on: {topic}

Based on the following information:
{combined_content}

Create a structured report with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Recommendations

Report:"""
        
        generation_msg = self.protocol.create_message(
            message_type=MessageType.GENERATION_REQUEST,
            sender=self.agent_type,
            receiver=AgentType.GENERATOR,
            payload={
                "prompt": prompt,
                "max_length": 1024,
                "temperature": 0.7,
                "task_type": "report"
            }
        )
        
        generation_response = await self.router.process_and_wait(generation_msg, timeout=30.0)
        
        if generation_response is None:
            self.log_error("Timeout waiting for generation response")
            return "Request timed out while generating report. Please try again.", ["timeout"], len(docs)
        
        if generation_response.message_type == MessageType.ERROR:
            return "I encountered an error generating the report.", ["generation_failed"], len(docs)
        
        report_content = generation_response.payload.get("generated_text", "")
        
        # Step 3: Fill template via MCP
        automation_msg = self.protocol.create_message(
            message_type=MessageType.AUTOMATION_REQUEST,
            sender=self.agent_type,
            receiver=AgentType.AUTOMATION,
            payload={
                "action": "fill_template",
                "template_name": "default_report",
                "data": {
                    "title": topic,
                    "content": report_content,
                    "date": f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "metadata": {"description": "project summary"}
                },
                "output_format": "txt"
            }
        )
        
        automation_response = await self.router.process_and_wait(automation_msg, timeout=10.0)
        
        if automation_response is None:
            self.log_error("Timeout waiting for automation response")
            return f"Report generated but not saved (timeout): {report_content[:200]}...", ["generate_report"], len(docs)
        
        if automation_response.message_type == MessageType.ERROR:
            return f"Report generated but not saved: {report_content[:200]}...", ["generate_report"], len(docs)
        
        output_path = automation_response.payload.get("output_path", "unknown")
        return f"Report generated successfully and saved to: {output_path}", ["retrieve_documents", "generate_report", "fill_template"], len(docs)

        
        # Extract string message from GenerateReportResponse
        if response.success:
            message = f"Report generated successfully! Preview:\n\n{response.report_preview}"
            if response.output_path:
                message += f"\n\nFull report saved to: {response.output_path}"
        else:
            message = "Failed to generate report. Please try again."
        
        return message, actions, docs_used
    
    async def _handle_task_response(self, message: AOPMessage) -> AOPMessage:
        """Handle response from delegated task"""
        self.log_info(f"Received task response from {message.sender}")
        
        # In a full implementation, this would update the execution state
        # and potentially trigger next steps in the workflow
        
        return self.protocol.create_message(
            message_type=MessageType.STATUS,
            sender=self.agent_type,
            receiver=message.sender,
            payload={"status": "acknowledged"},
            parent_message_id=message.message_id
        )
    
    def _create_error_response(self, original_message: AOPMessage, error_type: str, error_message: str) -> AOPMessage:
        """Create error response"""
        return self.protocol.create_error_response(
            sender=self.agent_type,
            receiver=original_message.sender,
            error_type=error_type,
            error_message=error_message,
            parent_message_id=original_message.message_id,
            recoverable=False
        )
    
    def get_task_history(self) -> list:
        """Get task execution history"""
        return self.task_history
