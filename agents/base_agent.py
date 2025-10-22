"""
Base Agent class for all DocuMind agents
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from mcp.protocol import MCPMessage, AgentType

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_type: AgentType, name: Optional[str] = None):
        self.agent_type = agent_type
        self.name = name or agent_type.value
        self.logger = logging.getLogger(f"agent.{self.name}")
    
    @abstractmethod
    async def process(self, message: MCPMessage) -> MCPMessage:
        """
        Process an incoming message and return a response
        
        Args:
            message: Incoming MCP message
            
        Returns:
            Response MCP message
        """
        pass
    
    def log_info(self, msg: str):
        """Log info message"""
        self.logger.info(f"[{self.name}] {msg}")
    
    def log_warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(f"[{self.name}] {msg}")
    
    def log_error(self, msg: str, exc_info: bool = False):
        """Log error message"""
        self.logger.error(f"[{self.name}] {msg}", exc_info=exc_info)
    
    def log_debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(f"[{self.name}] {msg}")
    
    def extract_payload(self, message: MCPMessage, key: str, default: Any = None) -> Any:
        """Extract a value from message payload"""
        return message.payload.get(key, default)
