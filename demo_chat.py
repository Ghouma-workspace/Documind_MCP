#!/usr/bin/env python3
"""
Chat Demo for DocuMind
Interactive chat with the AI assistant
"""

import asyncio
import sys
from config import settings
from pipelines.haystack_pipeline import HaystackPipeline
from mcp.protocol import MCPProtocol, AgentType, MessageType
from agents.reasoner_agent import ReasonerAgent
from agents.retriever_agent import RetrieverAgent
from agents.generator_agent import GeneratorAgent
from agents.automation_agent import AutomationAgent


async def main():
    """Main chat demo function"""
    print("=" * 80)
    print("🤖 DocuMind Chat Demo")
    print("=" * 80)
    print("\nInitializing system...")
    
    # Initialize pipeline
    pipeline = HaystackPipeline(
        model_name=settings.model_name,
        embedding_model=settings.embedding_model,
        device=settings.device,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        use_hf_inference_api=settings.use_hf_inference_api,
        hf_api_key=settings.hf_api_key,
        use_cerebras_api=settings.use_cerebras_api,
        cerebras_api_key=settings.cerebras_api_key,
        cerebras_model=settings.cerebras_model
    )
    
    # Warm up components
    print("Warming up components...")
    try:
        pipeline.warm_up()
    except Exception as e:
        print(f"⚠️  Warning: {str(e)}")
    
    # Initialize MCP protocol
    protocol = MCPProtocol()
    
    # Initialize agents
    reasoner = ReasonerAgent(protocol)
    retriever = RetrieverAgent(protocol, pipeline)
    generator = GeneratorAgent(protocol, pipeline)
    automation = AutomationAgent(
        protocol,
        str(settings.templates_path),
        str(settings.outputs_path)
    )
    
    print("\n✅ System ready!")
    print("\nYou can:")
    print("  • Ask questions about your documents")
    print("  • Request summaries")
    print("  • Generate reports")
    print("  • Have a general conversation")
    print("\nType 'quit' or 'exit' to end the chat.\n")
    print("=" * 80)
    
    # Ingest sample documents if available
    docs_dir = settings.documents_path
    if docs_dir.exists():
        doc_count = len(list(docs_dir.glob('*.txt'))) + len(list(docs_dir.glob('*.pdf')))
        if doc_count > 0:
            print(f"\n📄 Found {doc_count} documents. Ingesting...")
            result = retriever.ingest_documents(str(docs_dir))
            print(f"✅ Ingested {result['ingested']}/{result['total']} documents\n")
    
    conversation_id = "chat_demo_001"
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n👋 Goodbye! Thanks for using DocuMind!")
                break
            
            # Create chat message
            chat_msg = protocol.create_message(
                message_type=MessageType.CHAT_REQUEST,
                sender=AgentType.AUTOMATION,  # Representing the user
                receiver=AgentType.REASONER,
                payload={
                    "message": user_input,
                    "conversation_id": conversation_id,
                    "context": {}
                }
            )
            
            # Process through reasoner
            print("\n🤔 Assistant is thinking...")
            response_msg = await reasoner.process(chat_msg)
            
            # Extract and display response
            payload = response_msg.payload
            intent = payload.get("intent", "unknown")
            confidence = payload.get("confidence", 0.0)
            actions = payload.get("actions_taken", [])
            docs_used = payload.get("documents_used", 0)
            message = payload.get("message", "I'm not sure how to respond.")
            
            print(f"\n🤖 Assistant: {message}")
            print(f"\n   [Intent: {intent} | Confidence: {confidence:.0%}", end="")
            if docs_used > 0:
                print(f" | Documents: {docs_used}", end="")
            if actions:
                print(f" | Actions: {', '.join(actions)}", end="")
            print("]")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
