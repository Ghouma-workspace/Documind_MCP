"""
Unit tests for DocuMind agents
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.protocol import MCPProtocol, AgentType, MessageType, TaskType
from agents.reasoner_agent import ReasonerAgent
from agents.base_agent import BaseAgent


class TestBaseAgent:
    """Test base agent functionality"""
    
    def test_base_agent_initialization(self):
        """Test base agent can be initialized"""
        
        class TestAgent(BaseAgent):
            async def process(self, message):
                return message
        
        agent = TestAgent(AgentType.REASONER, "TestAgent")
        assert agent.agent_type == AgentType.REASONER
        assert agent.name == "TestAgent"
    
    def test_base_agent_logging(self):
        """Test base agent logging methods"""
        
        class TestAgent(BaseAgent):
            async def process(self, message):
                return message
        
        agent = TestAgent(AgentType.REASONER)
        
        # These should not raise exceptions
        agent.log_info("Test info message")
        agent.log_error("Test error message")
        agent.log_debug("Test debug message")


class TestReasonerAgent:
    """Test reasoner agent functionality"""
    
    @pytest.fixture
    def protocol(self):
        """Create MCP protocol instance"""
        return MCPProtocol()
    
    @pytest.fixture
    def reasoner(self, protocol):
        """Create reasoner agent instance"""
        return ReasonerAgent(protocol)
    
    def test_reasoner_initialization(self, reasoner):
        """Test reasoner agent initialization"""
        assert reasoner.agent_type == AgentType.REASONER
        assert reasoner.name == "ReasonerAgent"
        assert len(reasoner.task_history) == 0
    
    @pytest.mark.asyncio
    async def test_reasoner_task_request(self, reasoner, protocol):
        """Test reasoner handling task request"""
        
        # Create task request message
        message = protocol.create_task_request(
            task_type=TaskType.SUMMARIZE,
            sender=AgentType.REASONER,
            receiver=AgentType.REASONER,
            parameters={"query": "test query"}
        )
        
        # Process message
        response = await reasoner.process(message)
        
        # Verify response
        assert response is not None
        assert response.message_type == MessageType.TASK_RESPONSE
    
    def test_execution_plan_creation(self, reasoner, protocol):
        """Test execution plan creation for different task types"""
        from mcp.protocol import TaskRequest
        
        # Test summarize task
        summarize_request = TaskRequest(
            task_type=TaskType.SUMMARIZE,
            parameters={"query": "test"}
        )
        plan = reasoner._create_execution_plan(summarize_request)
        assert len(plan["steps"]) >= 2  # Should have retrieve + generate
        
        # Test query task
        query_request = TaskRequest(
            task_type=TaskType.QUERY,
            parameters={"question": "test question"}
        )
        plan = reasoner._create_execution_plan(query_request)
        assert len(plan["steps"]) >= 2
        
        # Test report generation task
        report_request = TaskRequest(
            task_type=TaskType.GENERATE_REPORT,
            parameters={"report_topic": "test"}
        )
        plan = reasoner._create_execution_plan(report_request)
        assert len(plan["steps"]) >= 3  # Should have retrieve + generate + automate


class TestMCPIntegration:
    """Test MCP protocol integration with agents"""
    
    @pytest.fixture
    def protocol(self):
        """Create MCP protocol instance"""
        return MCPProtocol()
    
    def test_message_creation(self, protocol):
        """Test creating MCP messages"""
        message = protocol.create_message(
            message_type=MessageType.QUERY,
            sender=AgentType.REASONER,
            receiver=AgentType.RETRIEVER,
            payload={"query": "test"}
        )
        
        assert message.message_id is not None
        assert message.message_type == MessageType.QUERY
        assert message.sender == AgentType.REASONER
        assert message.receiver == AgentType.RETRIEVER
    
    def test_message_chain(self, protocol):
        """Test message chain tracking"""
        # Create parent message
        parent = protocol.create_message(
            message_type=MessageType.QUERY,
            sender=AgentType.REASONER,
            receiver=AgentType.RETRIEVER,
            payload={}
        )
        
        # Create child message
        child = protocol.create_message(
            message_type=MessageType.RETRIEVAL_RESPONSE,
            sender=AgentType.RETRIEVER,
            receiver=AgentType.REASONER,
            payload={},
            parent_message_id=parent.message_id
        )
        
        # Get message chain
        chain = protocol.get_message_chain(child.message_id)
        assert len(chain) == 2
        assert chain[0].message_id == parent.message_id
        assert chain[1].message_id == child.message_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
