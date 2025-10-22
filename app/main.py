"""
FastAPI Main Application
Entry point for the DocuMind API
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from app.api_routes import router as api_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('documind.log')
    ]
)

logger = logging.getLogger(__name__)


# Application state
class AppState:
    """Global application state"""
    def __init__(self):
        self.haystack_pipeline = None
        self.mcp_protocol = None
        self.mcp_router = None
        self.reasoner_agent = None
        self.retriever_agent = None
        self.generator_agent = None
        self.automation_agent = None
        self.initialized = False


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting DocuMind application...")
    
    try:
        # Import dependencies
        from pipelines.haystack_pipeline import HaystackPipeline
        from mcp.protocol import MCPProtocol
        from mcp.router import MCPRouter
        from agents.reasoner_agent import ReasonerAgent
        from agents.retriever_agent import RetrieverAgent
        from agents.generator_agent import GeneratorAgent
        from agents.automation_agent import AutomationAgent
        
        # Initialize Haystack pipeline
        logger.info("Initializing Haystack pipeline...")
        app_state.haystack_pipeline = HaystackPipeline(
            model_name=settings.model_name,
            embedding_model=settings.embedding_model,
            device=settings.device,
            max_length=settings.max_length,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            use_hf_inference_api=settings.use_hf_inference_api,
            hf_api_key=settings.hf_api_key,
            use_cerebras_api=settings.use_cerebras_api,
            cerebras_api_key=settings.cerebras_api_key,
            cerebras_model=settings.cerebras_model
        )
        
        # Initialize MCP protocol and router
        logger.info("Initializing MCP protocol...")
        app_state.mcp_protocol = MCPProtocol()
        app_state.mcp_router = MCPRouter(app_state.mcp_protocol)
        
        # Initialize agents
        logger.info("Initializing agents...")
        app_state.reasoner_agent = ReasonerAgent(app_state.mcp_protocol)
        app_state.retriever_agent = RetrieverAgent(
            app_state.mcp_protocol,
            app_state.haystack_pipeline
        )
        app_state.generator_agent = GeneratorAgent(
            app_state.mcp_protocol,
            app_state.haystack_pipeline
        )
        app_state.automation_agent = AutomationAgent(
            app_state.mcp_protocol,
            str(settings.templates_path),
            str(settings.outputs_path)
        )
        
        # Register agents with router
        from mcp.protocol import AgentType
        app_state.mcp_router.register_agent(AgentType.REASONER, app_state.reasoner_agent.process)
        app_state.mcp_router.register_agent(AgentType.RETRIEVER, app_state.retriever_agent.process)
        app_state.mcp_router.register_agent(AgentType.GENERATOR, app_state.generator_agent.process)
        app_state.mcp_router.register_agent(AgentType.AUTOMATION, app_state.automation_agent.process)
        
        # Auto-ingest documents from data directory if any exist
        documents_dir = settings.documents_path
        if documents_dir.exists():
            logger.info(f"Checking for documents in {documents_dir}...")
            result = app_state.retriever_agent.ingest_documents(str(documents_dir))
            if result["total"] > 0:
                logger.info(f"Auto-ingested {result['ingested']}/{result['total']} documents")
        
        app_state.initialized = True
        logger.info("DocuMind application started successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down DocuMind application...")
    if app_state.mcp_router:
        await app_state.mcp_router.stop()
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="DocuMind API",
    description="Agentic document automation system using MCP, Haystack, and open-source LLMs",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to DocuMind API",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "running" if app_state.initialized else "initializing"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not app_state.initialized:
        return JSONResponse(
            status_code=503,
            content={"status": "initializing", "message": "System is still starting up"}
        )
    
    try:
        # Check system components
        doc_count = app_state.retriever_agent.get_document_count()
        router_stats = app_state.mcp_router.get_stats()
        
        return {
            "status": "healthy",
            "components": {
                "haystack_pipeline": "ok",
                "mcp_router": "ok",
                "agents": "ok"
            },
            "stats": {
                "documents_indexed": doc_count,
                "messages_processed": router_stats.get("messages_completed", 0)
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not app_state.initialized:
        return JSONResponse(
            status_code=503,
            content={"error": "System not initialized"}
        )
    
    return {
        "retriever": app_state.retriever_agent.get_retrieval_stats(),
        "generator": app_state.generator_agent.get_generation_stats(),
        "automation": app_state.automation_agent.get_automation_stats(),
        "router": app_state.mcp_router.get_stats()
    }


def main():
    """Main entry point"""
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
