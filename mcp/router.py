"""
MCP Router - Routes messages between agents
Handles message delivery and agent coordination
"""

import asyncio
import logging
from typing import Dict, Optional, Callable, Awaitable, List
from collections import defaultdict
from datetime import datetime

from mcp.protocol import (
    MCPMessage,
    MCPProtocol,
    MessageType,
    MessageStatus,
    AgentType
)

logger = logging.getLogger(__name__)


class MCPRouter:
    """Routes messages between agents in the MCP system"""
    
    def __init__(self, protocol: Optional[MCPProtocol] = None):
        self.protocol = protocol or MCPProtocol()
        self.agents: Dict[AgentType, Callable[[MCPMessage], Awaitable[MCPMessage]]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.stats = defaultdict(int)
        
    def register_agent(
        self,
        agent_type: AgentType,
        handler: Callable[[MCPMessage], Awaitable[MCPMessage]]
    ):
        """Register an agent handler for a specific agent type"""
        self.agents[agent_type] = handler
        logger.info(f"Registered agent: {agent_type}")
    
    def unregister_agent(self, agent_type: AgentType):
        """Unregister an agent"""
        if agent_type in self.agents:
            del self.agents[agent_type]
            logger.info(f"Unregistered agent: {agent_type}")
    
    async def send_message(self, message: MCPMessage) -> str:
        """Send a message to the queue for routing"""
        await self.message_queue.put(message)
        self.stats['messages_sent'] += 1
        logger.debug(f"Message {message.message_id} queued for {message.receiver}")
        return message.message_id
    
    async def route_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Route a message to the appropriate agent"""
        receiver = message.receiver
        
        if receiver not in self.agents:
            error_msg = f"No agent registered for type: {receiver}"
            logger.error(error_msg)
            self.stats['routing_errors'] += 1
            
            # Send error response back to sender
            error_response = self.protocol.create_error_response(
                sender=receiver,
                receiver=message.sender,
                error_type="AgentNotFound",
                error_message=error_msg,
                parent_message_id=message.message_id,
                recoverable=False
            )
            return error_response
        
        try:
            # Update message status
            self.protocol.update_message_status(message.message_id, MessageStatus.IN_PROGRESS)
            
            # Get agent handler
            handler = self.agents[receiver]
            
            # Process message
            logger.info(f"Routing message {message.message_id} to {receiver}")
            response = await handler(message)
            
            # Update status based on response
            if response and response.message_type != MessageType.ERROR:
                self.protocol.update_message_status(message.message_id, MessageStatus.COMPLETED)
                self.stats['messages_completed'] += 1
            else:
                self.protocol.update_message_status(message.message_id, MessageStatus.FAILED)
                self.stats['messages_failed'] += 1
            
            return response
            
        except Exception as e:
            logger.error(f"Error routing message {message.message_id}: {str(e)}", exc_info=True)
            self.protocol.update_message_status(message.message_id, MessageStatus.FAILED)
            self.stats['routing_errors'] += 1
            
            # Create error response
            error_response = self.protocol.create_error_response(
                sender=receiver,
                receiver=message.sender,
                error_type=type(e).__name__,
                error_message=str(e),
                parent_message_id=message.message_id,
                stack_trace=None,
                recoverable=True
            )
            return error_response
    
    async def start(self):
        """Start the message router"""
        if self.running:
            logger.warning("Router is already running")
            return
        
        self.running = True
        logger.info("MCP Router started")
        
        while self.running:
            try:
                # Get message from queue with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Route the message
                response = await self.route_message(message)
                
                # If there's a response and it's not back to the original sender, queue it
                if response and response.receiver != message.sender:
                    await self.message_queue.put(response)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except asyncio.TimeoutError:
                # No messages in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error in router loop: {str(e)}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def stop(self):
        """Stop the message router"""
        self.running = False
        logger.info("MCP Router stopped")
    
    async def process_and_wait(self, message: MCPMessage, timeout: float = 30.0) -> Optional[MCPMessage]:
        """Send a message and wait for response"""
        # Send message
        message_id = await self.send_message(message)
        
        # Wait for response
        start_time = datetime.utcnow()
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            # Check if message is completed
            msg_status = next(
                (m for m in self.protocol.message_history if m.message_id == message_id),
                None
            )
            
            if msg_status and msg_status.status in [MessageStatus.COMPLETED, MessageStatus.FAILED]:
                # Find response message
                response = next(
                    (m for m in self.protocol.message_history 
                     if m.parent_message_id == message_id and m.sender == message.receiver),
                    None
                )
                return response
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"Timeout waiting for response to message {message_id}")
        return None
    
    async def broadcast_message(
        self,
        message_type: MessageType,
        sender: AgentType,
        payload: Dict,
        exclude: Optional[List[AgentType]] = None
    ) -> List[str]:
        """Broadcast a message to all registered agents (except excluded ones)"""
        exclude = exclude or []
        message_ids = []
        
        for agent_type in self.agents.keys():
            if agent_type not in exclude and agent_type != sender:
                message = self.protocol.create_message(
                    message_type=message_type,
                    sender=sender,
                    receiver=agent_type,
                    payload=payload
                )
                message_id = await self.send_message(message)
                message_ids.append(message_id)
        
        return message_ids
    
    def get_stats(self) -> Dict[str, int]:
        """Get router statistics"""
        stats = dict(self.stats)
        stats['queue_size'] = self.message_queue.qsize()
        stats['registered_agents'] = len(self.agents)
        return stats
    
    def reset_stats(self):
        """Reset router statistics"""
        self.stats.clear()
        logger.info("Router statistics reset")
