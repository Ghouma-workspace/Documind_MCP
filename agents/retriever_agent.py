"""
Retriever Agent - Handles document retrieval using Haystack
"""

import logging
from typing import List, Dict, Any, Optional

from agents.base_agent import BaseAgent
from aop.protocol import (
    AOPMessage,
    AOPProtocol,
    AgentType,
    MessageType,
    RetrievalRequest,
    RetrievalResponse
)
from pipelines.haystack_pipeline import HaystackPipeline

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """
    Retriever Agent - Document search and retrieval
    - Interfaces with Haystack pipeline
    - Performs BM25, semantic, and hybrid search
    - Returns relevant document chunks
    """
    
    def __init__(self, protocol: AOPProtocol, haystack_pipeline: HaystackPipeline):
        super().__init__(AgentType.RETRIEVER, "RetrieverAgent")
        self.protocol = protocol
        self.pipeline = haystack_pipeline
        self.retrieval_history = []
    
    async def process(self, message: AOPMessage) -> AOPMessage:
        """Process incoming retrieval request"""
        self.log_info(f"Processing message: {message.message_type}")
        
        try:
            if message.message_type == MessageType.RETRIEVAL_REQUEST:
                return await self._handle_retrieval_request(message)
            elif message.message_type == MessageType.QUERY:
                return await self._handle_query(message)
            else:
                return self._create_error_response(
                    message,
                    "Unsupported message type",
                    f"Cannot process message type: {message.message_type}"
                )
        except Exception as e:
            self.log_error(f"Error processing message: {str(e)}", exc_info=True)
            return self._create_error_response(message, "ProcessingError", str(e))
    
    async def _handle_retrieval_request(self, message: AOPMessage) -> AOPMessage:
        """Handle retrieval request"""
        try:
            # Parse request
            retrieval_req = RetrievalRequest(**message.payload)
            
            self.log_info(f"Retrieving documents for query: '{retrieval_req.query}' (top_k={retrieval_req.top_k})")
            
            # Perform retrieval
            documents = self.pipeline.retrieve(
                query=retrieval_req.query,
                top_k=retrieval_req.top_k,
                mode=retrieval_req.retrieval_mode
            )
            
            # Convert Haystack Documents to dict format
            doc_dicts = []
            scores = []
            
            for doc in documents:
                doc_dict = {
                    "id": doc.id,
                    "content": doc.content,
                    "meta": doc.meta,
                    "score": doc.score if hasattr(doc, 'score') else 0.0
                }
                doc_dicts.append(doc_dict)
                scores.append(doc.score if hasattr(doc, 'score') else 0.0)
            
            self.log_info(f"Retrieved {len(doc_dicts)} documents")
            
            # Store in history
            self.retrieval_history.append({
                "query": retrieval_req.query,
                "num_results": len(doc_dicts),
                "mode": retrieval_req.retrieval_mode
            })
            
            # Create response
            retrieval_resp = RetrievalResponse(
                documents=doc_dicts,
                scores=scores,
                total_found=len(doc_dicts),
                query=retrieval_req.query
            )
            
            return self.protocol.create_message(
                message_type=MessageType.RETRIEVAL_RESPONSE,
                sender=self.agent_type,
                receiver=message.sender,
                payload=retrieval_resp.model_dump(),
                parent_message_id=message.message_id
            )
            
        except Exception as e:
            self.log_error(f"Error handling retrieval request: {str(e)}", exc_info=True)
            return self._create_error_response(message, "RetrievalError", str(e))
    
    async def _handle_query(self, message: AOPMessage) -> AOPMessage:
        """Handle general query (convenience method)"""
        try:
            query_text = message.payload.get("query", message.payload.get("question", ""))
            top_k = message.payload.get("top_k", 5)
            
            # Create retrieval request
            retrieval_req = RetrievalRequest(
                query=query_text,
                top_k=top_k,
                retrieval_mode="hybrid"
            )
            
            # Update message payload and process
            message.payload = retrieval_req.model_dump()
            message.message_type = MessageType.RETRIEVAL_REQUEST
            
            return await self._handle_retrieval_request(message)
            
        except Exception as e:
            self.log_error(f"Error handling query: {str(e)}", exc_info=True)
            return self._create_error_response(message, "QueryError", str(e))
    
    async def retrieve(self, query: str, top_k: int = 5, mode: str = "hybrid"):
        """
        Direct retrieval method (no MCP) - for API endpoints
        Returns: List of Haystack Documents
        """
        self.log_info(f"Direct retrieve: '{query}' (top_k={top_k}, mode={mode})")
        documents = self.pipeline.retrieve(query=query, top_k=top_k, mode=mode)
        self.log_info(f"Retrieved {len(documents)} documents")
        return documents

    
    def ingest_documents(self, directory_path: str) -> Dict[str, Any]:
        """Ingest documents from directory"""
        self.log_info(f"Ingesting documents from: {directory_path}")
        result = self.pipeline.ingest_directory(directory_path)
        self.log_info(f"Ingestion complete: {result['ingested']}/{result['total']} documents")
        return result
    
    def ingest_document(self, file_path: str) -> bool:
        """Ingest single document"""
        self.log_info(f"Ingesting document: {file_path}")
        success = self.pipeline.ingest_document(file_path)
        if success:
            self.log_info(f"Successfully ingested: {file_path}")
        else:
            self.log_error(f"Failed to ingest: {file_path}")
        return success
    
    def get_document_count(self) -> int:
        """Get total number of indexed documents"""
        count = self.pipeline.get_document_count()
        self.log_debug(f"Total documents in store: {count}")
        return count
    
    def clear_documents(self):
        """Clear all documents from store"""
        self.log_info("Clearing document store")
        self.pipeline.clear_documents()
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "total_retrievals": len(self.retrieval_history),
            "document_count": self.get_document_count(),
            "recent_queries": self.retrieval_history[-10:] if self.retrieval_history else []
        }
    
    def _create_error_response(self, original_message: AOPMessage, error_type: str, error_message: str) -> AOPMessage:
        """Create error response"""
        return self.protocol.create_error_response(
            sender=self.agent_type,
            receiver=original_message.sender,
            error_type=error_type,
            error_message=error_message,
            parent_message_id=original_message.message_id,
            recoverable=True
        )
