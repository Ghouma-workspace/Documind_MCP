"""
Demo script for DocuMind
Demonstrates end-to-end workflow
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from pipelines.haystack_pipeline import HaystackPipeline
from mcp.protocol import MCPProtocol, AgentType, MessageType, TaskType
from agents.reasoner_agent import ReasonerAgent
from agents.retriever_agent import RetrieverAgent
from agents.generator_agent import GeneratorAgent
from agents.automation_agent import AutomationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def demo_workflow():
    """Demonstrate complete DocuMind workflow"""
    
    print("=" * 80)
    print("🧠 DocuMind Demo - Agentic Document Automation")
    print("=" * 80)
    print()
    
    # Step 1: Initialize system
    print("📦 Step 1: Initializing system components...")
    print("-" * 80)
    
    # Initialize Haystack pipeline
    print("  • Initializing Haystack pipeline...")
    
    # Determine inference method
    if settings.use_cerebras_api:
        inference_method = "Cerebras API"
    elif settings.use_hf_inference_api:
        inference_method = "Hugging Face API"
    else:
        inference_method = "Local Model"
    
    print(f"  • Using {inference_method} for generation")
    
    pipeline = HaystackPipeline(
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
    
    # Warm up pipeline components
    print("  • Warming up pipeline components...")
    try:
        pipeline.warm_up()
        print("    ✓ Pipeline components warmed up")
    except Exception as e:
        print(f"❌ ERROR: Failed to warm up pipeline components: {str(e)}")
        print("Demo cannot continue without properly initialized components.")
        return
    
    print("    ✓ Haystack pipeline ready")
    
    # Initialize MCP protocol
    print("  • Initializing MCP protocol...")
    protocol = MCPProtocol()
    print("    ✓ MCP protocol ready")
    
    # Initialize agents
    print("  • Initializing agents...")
    reasoner = ReasonerAgent(protocol)
    retriever = RetrieverAgent(protocol, pipeline)
    generator = GeneratorAgent(protocol, pipeline)
    automation = AutomationAgent(
        protocol,
        str(settings.templates_path),
        str(settings.outputs_path)
    )
    print("    ✓ All agents ready")
    print()
    
    # Step 2: Create sample documents
    print("📄 Step 2: Creating sample documents...")
    print("-" * 80)
    
    sample_docs = [
        ("project_overview.txt", """
Project Name: AI Document Processor
Start Date: January 2024
Status: In Progress

Overview:
The AI Document Processor project aims to create an intelligent system for 
automated document analysis and processing. The system uses machine learning
models to understand, classify, and extract information from various document types.

Key Features:
- Multi-format document support (PDF, DOCX, TXT)
- Intelligent text extraction and classification
- Automated summarization and report generation
- Integration with existing workflows

Team Size: 5 developers
Budget: $150,000
Timeline: 6 months
        """),
        
        ("technical_requirements.txt", """
Technical Requirements Document

System Architecture:
- Python-based backend using FastAPI
- Vector database for document storage (FAISS)
- Open-source LLMs for text generation
- RESTful API for integration

Required Technologies:
- Python 3.9+
- Haystack framework
- Transformers library
- FastAPI for API development
- Streamlit for user interface

Performance Requirements:
- Process documents in under 10 seconds
- Support concurrent users (10+)
- 99% uptime SLA
- Scalable architecture
        """),
        
        ("meeting_notes.txt", """
Project Meeting Notes - Week 12

Date: March 15, 2024
Attendees: Alice (PM), Bob (Dev Lead), Carol (ML Engineer), Dave (QA)

Discussion Points:
1. Document processing pipeline is complete
2. Testing phase has begun
3. UI improvements needed based on user feedback
4. API documentation is ready

Action Items:
- Bob: Fix edge cases in PDF extraction (Due: Mar 20)
- Carol: Optimize model inference speed (Due: Mar 25)
- Dave: Complete integration testing (Due: Mar 22)
- Alice: Prepare demo for stakeholders (Due: Mar 30)

Risks:
- Potential delay in model optimization
- Need more test data for edge cases

Next Meeting: March 22, 2024
        """)
    ]
    
    # Save sample documents
    docs_dir = settings.documents_path
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, content in sample_docs:
        file_path = docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"  ✓ Created: {filename}")
    print()
    
    # Step 3: Ingest documents
    print("📥 Step 3: Ingesting documents into system...")
    print("-" * 80)
    
    result = retriever.ingest_documents(str(docs_dir))
    print(f"  ✓ Ingested {result['ingested']}/{result['total']} documents")
    
    # Check if ingestion was successful
    if result['ingested'] == 0:
        print("❌ ERROR: No documents were ingested successfully!")
        print("Demo cannot continue without indexed documents.")
        return
    elif result['failed'] > 0:
        print(f"⚠️  WARNING: {result['failed']} documents failed to ingest")
        print("Continuing with available documents...")
    
    print(f"  • Total chunks indexed: {retriever.get_document_count()}")
    print()
    
    # Step 4: Query documents
    print("❓ Step 4: Querying documents...")
    print("-" * 80)
    
    query = "What is the project budget and timeline?"
    print(f"  Query: '{query}'")
    print()
    
    # Create retrieval request
    retrieval_msg = protocol.create_message(
        message_type=MessageType.RETRIEVAL_REQUEST,
        sender=AgentType.REASONER,
        receiver=AgentType.RETRIEVER,
        payload={
            "query": query,
            "top_k": 3,
            "retrieval_mode": "hybrid"
        }
    )
    
    retrieval_response = await retriever.process(retrieval_msg)
    documents = retrieval_response.payload.get("documents", [])
    
    print(f"  ✓ Retrieved {len(documents)} relevant chunks")
    
    # Check if retrieval was successful
    if len(documents) == 0:
        print("❌ ERROR: No relevant documents were retrieved!")
        print("This could indicate issues with indexing or component warm-up.")
        print("Demo cannot continue without retrieved context.")
        return
    
    print()
    
    # Display retrieved content
    for i, doc in enumerate(documents[:2], 1):
        print(f"  Document {i}:")
        print(f"    Source: {doc['meta'].get('file_name', 'Unknown')}")
        print(f"    Preview: {doc['content'][:150]}...")
        print()
    
    # Step 5: Generate summary
    print("📝 Step 5: Generating summary...")
    print("-" * 80)
    
    # Build context
    context = "\n\n".join([doc["content"][:500] for doc in documents])
    
    prompt = f"""Based on the following project documents, answer this question concisely:

Question: {query}

Context:
{context}

Answer:"""
    
    generation_msg = protocol.create_message(
        message_type=MessageType.GENERATION_REQUEST,
        sender=AgentType.REASONER,
        receiver=AgentType.GENERATOR,
        payload={
            "prompt": prompt,
            "max_length": 100,
            "temperature": 0.7,
            "task_type": "answer"
        }
    )
    
    try:
        generation_response = await generator.process(generation_msg)
        answer = generation_response.payload.get("generated_text", "")
        
        # Check if generation was successful
        if not answer or answer.startswith("Error:"):
            print(f"❌ ERROR: Text generation failed: {answer}")
            print("Demo cannot continue without successful text generation.")
            return
        
        print(f"  Answer: {answer}")
    except Exception as e:
        print(f"❌ ERROR: Generation failed with exception: {str(e)}")
        print("Demo cannot continue without successful text generation.")
        return
    print()
    
    # Step 6: Create report
    print("📊 Step 6: Generating project report...")
    print("-" * 80)
    
    automation_msg = protocol.create_message(
        message_type=MessageType.AUTOMATION_REQUEST,
        sender=AgentType.REASONER,
        receiver=AgentType.AUTOMATION,
        payload={
            "action": "fill_template",
            "template_name": "project_summary",
            "data": {
                "project_name": "AI Document Processor",
                "date": "March 2024",
                "status": "In Progress",
                "overview": "Automated document analysis system using AI",
                "completion": "75",
                "completed_tasks": [
                    "Document processing pipeline",
                    "API development",
                    "Basic UI implementation"
                ],
                "in_progress_tasks": [
                    "Model optimization",
                    "Integration testing"
                ],
                "budget": "$150,000",
                "timeline": "6 months",
                "team_size": "5"
            },
            "output_format": "markdown"
        }
    )
    
    automation_response = await automation.process(automation_msg)
    
    if automation_response.payload.get("success"):
        output_path = automation_response.payload.get("output_path")
        print(f"  ✓ Report generated successfully")
        print(f"  • Output: {output_path}")
    else:
        error_msg = automation_response.payload.get("error", "Unknown error")
        print(f"❌ ERROR: Report generation failed: {error_msg}")
        print("Demo cannot continue without successful report generation.")
        return
    print()
    
    # Step 7: Statistics
    print("📈 Step 7: System statistics...")
    print("-" * 80)
    
    retriever_stats = retriever.get_retrieval_stats()
    generator_stats = generator.get_generation_stats()
    automation_stats = automation.get_automation_stats()
    
    print(f"  Retriever:")
    print(f"    • Total retrievals: {retriever_stats['total_retrievals']}")
    print(f"    • Documents indexed: {retriever_stats['document_count']}")
    print()
    print(f"  Generator:")
    print(f"    • Total generations: {generator_stats['total_generations']}")
    print(f"    • Avg execution time: {generator_stats.get('avg_execution_time', 0):.2f}s")
    print()
    print(f"  Automation:")
    print(f"    • Total automations: {automation_stats['total_automations']}")
    print(f"    • Outputs generated: {automation_stats['outputs_generated']}")
    print()
    
    # Summary
    print("=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Start the API server: uvicorn app.main:app --reload")
    print("  2. Start the UI: streamlit run app/ui.py")
    print("  3. Access API docs: http://localhost:8000/docs")
    print("  4. Access UI: http://localhost:8501")
    print()
    print(f"Sample documents created in: {docs_dir}")
    print(f"Generated reports in: {settings.outputs_path}")
    print()


def main():
    """Main entry point"""
    try:
        asyncio.run(demo_workflow())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
