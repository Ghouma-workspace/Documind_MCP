"""
AOP Router - Routes messages between agents
Handles message delivery and agent coordination
"""

import asyncio
import logging
from typing import Dict, Optional, Callable, Awaitable, List
from collections import defaultdict
from datetime import datetime

from aop.protocol import (
    AOPMessage,
    AOPProtocol,
    MessageType,
    MessageStatus,
    AgentType
)

logger = logging.getLogger(__name__)


class AOPRouter:
    """Routes messages between agents in the AOP system"""
    
    def __init__(self, protocol: Optional[AOPProtocol] = None):
        self.protocol = protocol or AOPProtocol()
        self.agents: Dict[AgentType, Callable[[AOPMessage], Awaitable[AOPMessage]]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.stats = defaultdict(int)
        
    def register_agent(
        self,
        agent_type: AgentType,
        handler: Callable[[AOPMessage], Awaitable[AOPMessage]]
    ):
        """Register an agent handler for a specific agent type"""
        self.agents[agent_type] = handler
        logger.info(f"Registered agent: {agent_type}")
    
    def unregister_agent(self, agent_type: AgentType):
        """Unregister an agent"""
        if agent_type in self.agents:
            del self.agents[agent_type]
            logger.info(f"Unregistered agent: {agent_type}")
    
    async def send_message(self, message: AOPMessage) -> str:
        """Send a message to the queue for routing"""
        await self.message_queue.put(message)
        self.stats['messages_sent'] += 1
        logger.debug(f"Message {message.message_id} queued for {message.receiver}")
        return message.message_id
    
    async def route_message(self, message: AOPMessage) -> Optional[AOPMessage]:
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
        """Start the message router - processes both queued and pending messages"""
        if self.running:
            logger.warning("Router is already running")
            return
        
        self.running = True
        logger.info("AOP Router started")
        
        while self.running:
            try:
                # Process pending messages from protocol and add to queue
                await self._check_and_queue_pending_messages()
                
                # Now process messages from queue with timeout
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=0.5  # Reduced timeout to check pending more frequently
                    )
                    
                    # Route the message
                    response = await self.route_message(message)
                    
                    # Only queue response if it's NOT a response type itself
                    # This prevents infinite loops where responses get re-queued
                    if response and not self._is_response_message(response):
                        await self.message_queue.put(response)
                    
                    # Mark task as done
                    self.message_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No messages in queue, continue to check pending
                    continue
                
            except Exception as e:
                logger.error(f"Error in router loop: {str(e)}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def stop(self):
        """Stop the message router"""
        self.running = False
        logger.info("AOP Router stopped")
    
    def _is_response_message(self, message: AOPMessage) -> bool:
        """Check if a message is a response type (to prevent infinite loops)"""
        response_types = [
            MessageType.TASK_RESPONSE,
            MessageType.RETRIEVAL_RESPONSE,
            MessageType.GENERATION_RESPONSE,
            MessageType.AUTOMATION_RESPONSE,
            MessageType.CHAT_RESPONSE,
            MessageType.ERROR,
            MessageType.STATUS
        ]
        return message.message_type in response_types
    
    async def _check_and_queue_pending_messages(self):
        """Check for pending messages in protocol and add them to the queue"""
        for agent_type in self.agents.keys():
            pending = self.protocol.get_pending_messages(agent_type)
            for message in pending:
                # Skip if already in queue or being processed
                if message.status != MessageStatus.PENDING:
                    continue
                logger.debug(f"Found pending message {message.message_id} for {agent_type}")
                await self.message_queue.put(message)
    
    async def process_pending_messages(self):
        """
        Process all pending messages in the protocol - useful for one-time batch processing.
        This is different from _check_and_queue_pending_messages which only queues them.
        This method actually routes them and waits for responses.
        """
        logger.info("Processing all pending messages...")
        processed_count = 0
        
        for agent_type in self.agents.keys():
            pending = self.protocol.get_pending_messages(agent_type)
            for message in pending:
                if message.status != MessageStatus.PENDING:
                    continue
                    
                logger.debug(f"Processing pending message {message.message_id} for {agent_type}")
                response = await self.route_message(message)
                if response and not self._is_response_message(response):
                    # Add response back to queue for further routing
                    await self.message_queue.put(response)
                processed_count += 1
        
        logger.info(f"Processed {processed_count} pending messages")
        return processed_count
    
    async def process_and_wait(self, message: AOPMessage, timeout: float = 30.0) -> Optional[AOPMessage]:
        """Send a message and wait for response"""
        # Add message to protocol's history (it should already be there if created via protocol)
        # Then mark it for processing
        await self.message_queue.put(message)
        
        # Wait for response
        start_time = datetime.utcnow()
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            # Check if message is completed
            msg_status = next(
                (m for m in self.protocol.message_history if m.message_id == message.message_id),
                None
            )
            
            if msg_status and msg_status.status in [MessageStatus.COMPLETED, MessageStatus.FAILED]:
                # Find response message (child message from the receiver back to sender)
                response = next(
                    (m for m in self.protocol.message_history 
                     if m.parent_message_id == message.message_id and m.sender == message.receiver),
                    None
                )
                return response
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"Timeout waiting for response to message {message.message_id}")
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
