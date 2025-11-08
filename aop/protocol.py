"""
Agent Orchestration Protocol (AOP) Implementation
Defines message schemas and protocol logic for agent communication
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import uuid
import logging

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Types of messages in the AOP protocol"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    QUERY = "query"
    RETRIEVAL_REQUEST = "retrieval_request"
    RETRIEVAL_RESPONSE = "retrieval_response"
    GENERATION_REQUEST = "generation_request"
    GENERATION_RESPONSE = "generation_response"
    AUTOMATION_REQUEST = "automation_request"
    AUTOMATION_RESPONSE = "automation_response"
    CHAT_REQUEST = "chat_request"
    CHAT_RESPONSE = "chat_response"
    ERROR = "error"
    STATUS = "status"


class AgentType(str, Enum):
    """Types of agents in the system"""
    REASONER = "reasoner"
    RETRIEVER = "retriever"
    GENERATOR = "generator"
    AUTOMATION = "automation"


class TaskType(str, Enum):
    """Types of tasks that can be performed"""
    SUMMARIZE = "summarize"
    QUERY = "query"
    EXTRACT = "extract"
    GENERATE_REPORT = "generate_report"
    CLASSIFY = "classify"
    FILL_TEMPLATE = "fill_template"


class MessageStatus(str, Enum):
    """Status of message processing"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AOPMessage(BaseModel):
    """Base message structure for AOP protocol"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    sender: AgentType
    receiver: AgentType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_message_id: Optional[str] = None
    status: MessageStatus = MessageStatus.PENDING

    class Config:
        use_enum_values = True


class TaskRequest(BaseModel):
    """Request to perform a specific task"""
    task_type: TaskType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)


class TaskResponse(BaseModel):
    """Response from a completed task"""
    task_type: TaskType
    result: Any
    success: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    """Request to retrieve documents"""
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None
    retrieval_mode: str = Field(default="hybrid")  # hybrid, semantic, keyword


class RetrievalResponse(BaseModel):
    """Response with retrieved documents"""
    documents: List[Dict[str, Any]]
    scores: List[float]
    total_found: int
    query: str


class GenerationRequest(BaseModel):
    """Request to generate content"""
    prompt: str
    context: Optional[str] = None
    max_length: int = Field(default=512, ge=50, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    task_type: str = "general"  # summarize, generate, answer, etc.


class GenerationResponse(BaseModel):
    """Response with generated content"""
    generated_text: str
    prompt: str
    model_used: str
    tokens_generated: Optional[int] = None


class AutomationRequest(BaseModel):
    """Request to perform automation task"""
    action: str  # fill_template, save_file, export, etc.
    template_name: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    output_format: str = "markdown"
    output_path: Optional[str] = None


class AutomationResponse(BaseModel):
    """Response from automation task"""
    action: str
    success: bool
    output_path: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    """Error response structure"""
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    recoverable: bool = False


class ChatRequest(BaseModel):
    """Chat request from user"""
    message: str
    conversation_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Chat response to user"""
    message: str
    intent: str  # greeting, question, summarize, report, etc.
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    actions_taken: List[str] = Field(default_factory=list)
    documents_used: int = 0
    conversation_id: Optional[str] = None


class AOPProtocol:
    """Handles AOP protocol operations and message validation"""
    
    def __init__(self):
        self.message_history: List[AOPMessage] = []
        
    def create_message(
        self,
        message_type: MessageType,
        sender: AgentType,
        receiver: AgentType,
        payload: Dict[str, Any],
        parent_message_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AOPMessage:
        """Create a new AOP message"""
        message = AOPMessage(
            message_type=message_type,
            sender=sender,
            receiver=receiver,
            payload=payload,
            parent_message_id=parent_message_id,
            metadata=metadata or {}
        )
        
        # Response messages should start as COMPLETED, not PENDING
        # This prevents them from being picked up by the router again
        response_types = [
            MessageType.TASK_RESPONSE,
            MessageType.RETRIEVAL_RESPONSE,
            MessageType.GENERATION_RESPONSE,
            MessageType.AUTOMATION_RESPONSE,
            MessageType.CHAT_RESPONSE,
            MessageType.ERROR,
            MessageType.STATUS
        ]
        if message.message_type in response_types:
            message.status = MessageStatus.COMPLETED
        
        self.message_history.append(message)
        logger.debug(f"Created message {message.message_id}: {sender} -> {receiver} (status: {message.status})")
        return message
    
    def create_task_request(
        self,
        task_type: TaskType,
        sender: AgentType,
        receiver: AgentType,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        priority: int = 5
    ) -> AOPMessage:
        """Create a task request message"""
        task_request = TaskRequest(
            task_type=task_type,
            parameters=parameters,
            context=context or {},
            priority=priority
        )
        return self.create_message(
            message_type=MessageType.TASK_REQUEST,
            sender=sender,
            receiver=receiver,
            payload=task_request.model_dump()
        )
    
    def create_task_response(
        self,
        task_type: TaskType,
        sender: AgentType,
        receiver: AgentType,
        result: Any,
        success: bool,
        parent_message_id: str,
        error_message: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> AOPMessage:
        """Create a task response message"""
        task_response = TaskResponse(
            task_type=task_type,
            result=result,
            success=success,
            error_message=error_message,
            execution_time=execution_time
        )
        return self.create_message(
            message_type=MessageType.TASK_RESPONSE,
            sender=sender,
            receiver=receiver,
            payload=task_response.model_dump(),
            parent_message_id=parent_message_id
        )
    
    def create_error_response(
        self,
        sender: AgentType,
        receiver: AgentType,
        error_type: str,
        error_message: str,
        parent_message_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
        recoverable: bool = False
    ) -> AOPMessage:
        """Create an error response message"""
        error_response = ErrorResponse(
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            recoverable=recoverable
        )
        return self.create_message(
            message_type=MessageType.ERROR,
            sender=sender,
            receiver=receiver,
            payload=error_response.model_dump(),
            parent_message_id=parent_message_id
        )
    
    def get_message_chain(self, message_id: str) -> List[AOPMessage]:
        """Get the chain of messages related to a specific message"""
        chain = []
        current_id = message_id
        
        while current_id:
            message = next(
                (m for m in self.message_history if m.message_id == current_id),
                None
            )
            if message:
                chain.insert(0, message)
                current_id = message.parent_message_id
            else:
                break
        
        return chain
    
    def update_message_status(self, message_id: str, status: MessageStatus) -> bool:
        """Update the status of a message"""
        message = next(
            (m for m in self.message_history if m.message_id == message_id),
            None
        )
        if message:
            message.status = status
            logger.debug(f"Updated message {message_id} status to {status}")
            return True
        return False
    
    def get_pending_messages(self, receiver: AgentType) -> List[AOPMessage]:
        """Get all pending messages for a specific agent"""
        return [
            m for m in self.message_history
            if m.receiver == receiver and m.status == MessageStatus.PENDING
        ]
    
    def clear_history(self):
        """Clear message history"""
        self.message_history.clear()
        logger.info("Message history cleared")
