"""
Streamlit UI for DocuMind
Interactive web interface for document automation
"""

import streamlit as st
import requests
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure page
st.set_page_config(
    page_title="DocuMind - Document Automation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api"


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def upload_document(file):
    """Upload document to API"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        return response.json()
    except Exception as e:
        return {"success": False, "message": str(e)}


def query_documents(question, top_k=5):
    """Query documents"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "top_k": top_k}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def summarize_documents(query, top_k=10, max_length=512):
    """Summarize documents"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/summarize",
            json={"query": query, "top_k": top_k, "max_length": max_length}
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def generate_report(template, report_topic, output_format="markdown"):
    """Generate report"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-report",
            json={
                "template": template,
                "report_topic": report_topic,
                "output_format": output_format
            }
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def chat_with_assistant(message, conversation_id=None):
    """Chat with the AI assistant"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": message,
                "conversation_id": conversation_id
            }
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_documents():
    """Get list of documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        return response.json()
    except:
        return {"total_indexed": 0, "files": []}


def get_outputs():
    """Get list of outputs"""
    try:
        response = requests.get(f"{API_BASE_URL}/outputs")
        return response.json()
    except:
        return {"total": 0, "outputs": []}


def get_stats():
    """Get system statistics"""
    try:
        response = requests.get("http://localhost:8000/stats")
        return response.json()
    except:
        return {}


# Main UI
def main():
    # Header
    st.title("🧠 DocuMind")
    st.markdown("**Agentic Document Automation System**")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running! Please start the API server first:")
        st.code("uvicorn app.main:app --reload", language="bash")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Get stats
        stats = get_stats()
        
        if stats:
            # Document stats
            retriever_stats = stats.get("retriever", {})
            st.metric("Documents Indexed", retriever_stats.get("document_count", 0))
            st.metric("Total Retrievals", retriever_stats.get("total_retrievals", 0))
            
            # Generator stats
            generator_stats = stats.get("generator", {})
            st.metric("Generations", generator_stats.get("total_generations", 0))
            
            # Automation stats
            automation_stats = stats.get("automation", {})
            st.metric("Outputs Created", automation_stats.get("outputs_generated", 0))
        
        st.divider()
        
        # Navigation
        st.header("🧭 Navigation")
        page = st.radio(
            "Select Page",
            ["📤 Upload Documents", "💬 Query & Chat", "📝 Summarize", "📊 Generate Report", "📁 Browse Files"],
            label_visibility="collapsed"
        )
    
    # Main content area
    if page == "📤 Upload Documents":
        show_upload_page()
    elif page == "💬 Query & Chat":
        show_query_page()
    elif page == "📝 Summarize":
        show_summarize_page()
    elif page == "📊 Generate Report":
        show_report_page()
    elif page == "📁 Browse Files":
        show_files_page()


def show_upload_page():
    """Upload documents page"""
    st.header("📤 Upload Documents")
    st.markdown("Upload PDF, DOCX, or TXT files for processing")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Upload and Index", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Uploading {file.name}...")
                result = upload_document(file)
                results.append((file.name, result))
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.empty()
            progress_bar.empty()
            
            # Show results
            for filename, result in results:
                if result.get("success"):
                    st.success(f"✅ {filename}: {result.get('message')}")
                else:
                    st.error(f"❌ {filename}: {result.get('message')}")


def show_query_page():
    """Query documents and chat page"""
    st.header("💬 Query & Chat")
    
    # Create tabs for different modes
    tab1, tab2 = st.tabs(["💭 Chat", "🔍 Search Documents"])
    
    # Chat Tab
    with tab1:
        st.markdown("Chat with your AI assistant about your documents")
        
        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "conversation_id" not in st.session_state:
            import uuid
            st.session_state.conversation_id = str(uuid.uuid4())
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["user"])
            with st.chat_message("assistant"):
                st.write(chat["assistant"])
                if chat.get("intent"):
                    st.caption(f"💡 Intent: {chat['intent']} | Confidence: {chat['confidence']:.0%} | Docs used: {chat['docs_used']}")
        
        # Chat input
        user_message = st.chat_input("Type your message here...")
        
        if user_message:
            # Add user message to history
            with st.chat_message("user"):
                st.write(user_message)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = chat_with_assistant(
                        user_message,
                        st.session_state.conversation_id
                    )
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                        response_text = "Sorry, I encountered an error. Please try again."
                        intent = "error"
                        confidence = 0.0
                        docs_used = 0
                    else:
                        response_text = result.get("message", "I'm not sure how to respond.")
                        intent = result.get("intent", "unknown")
                        confidence = result.get("confidence", 0.0)
                        docs_used = result.get("documents_used", 0)
                        
                        st.write(response_text)
                        st.caption(f"💡 Intent: {intent} | Confidence: {confidence:.0%} | Docs used: {docs_used}")
            
            # Save to history
            st.session_state.chat_history.append({
                "user": user_message,
                "assistant": response_text,
                "intent": intent,
                "confidence": confidence,
                "docs_used": docs_used
            })
            st.rerun()
        
        # Clear history button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                import uuid
                st.session_state.conversation_id = str(uuid.uuid4())
                st.rerun()
    
    # Search Tab
    with tab2:
        st.markdown("Search documents with specific questions")
        
        # Query input
        question = st.text_input("Enter your question:", placeholder="What are the payment terms?")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of documents to search", 1, 20, 5)
        
        if st.button("Search", type="primary") and question:
            with st.spinner("Searching and generating answer..."):
                result = query_documents(question, top_k)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.subheader("Answer:")
                    st.write(result.get("answer", "No answer generated"))
                    
                    # Show sources
                    sources = result.get("sources", [])
                    if sources:
                        st.subheader("Sources:")
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"📄 {source.get('file_name', 'Unknown')} (Score: {source.get('score', 0):.2f})"):
                                st.write(f"Document ID: `{source.get('id')}`")


def show_summarize_page():
    """Summarize documents page"""
    st.header("📝 Summarize Documents")
    st.markdown("Generate summaries of your documents")
    
    # Query input
    query = st.text_area(
        "Summarization query (optional):",
        value="summarize all documents",
        help="Specify what to summarize or leave as default for all documents"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of documents", 1, 50, 10)
    with col2:
        max_length = st.slider("Summary length", 100, 2048, 512)
    
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating summary..."):
            result = summarize_documents(query, top_k, max_length)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.subheader("Summary:")
                st.write(result.get("summary", "No summary generated"))
                
                st.info(f"📊 Used {result.get('documents_used', 0)} documents")
                
                # Show sources
                sources = result.get("sources", [])
                if sources:
                    with st.expander("📚 Source Documents"):
                        for source in sources:
                            st.write(f"- {source}")


def show_report_page():
    """Generate report page"""
    st.header("📊 Generate Report")
    st.markdown("Create structured reports from your documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_topic = st.text_input(
            "Report Topic:",
            value="project summary",
            placeholder="e.g., project summary, financial overview"
        )
    
    with col2:
        output_format = st.selectbox(
            "Output Format:",
            ["markdown", "txt", "json"]
        )
    
    template = st.text_input(
        "Template Name:",
        value="default_report",
        help="Name of the template to use (without extension)"
    )
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            result = generate_report(template, report_topic, output_format)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                if result.get("success"):
                    st.success("✅ Report generated successfully!")
                    
                    # Show preview
                    preview = result.get("report_preview")
                    if preview:
                        st.subheader("Preview:")
                        st.write(preview)
                    
                    # Show output path
                    output_path = result.get("output_path")
                    if output_path:
                        st.info(f"📁 Saved to: `{output_path}`")
                else:
                    st.error("Failed to generate report")


def show_files_page():
    """Browse files page"""
    st.header("📁 Browse Files")
    
    # Tabs for documents and outputs
    tab1, tab2 = st.tabs(["📄 Documents", "📤 Generated Outputs"])
    
    with tab1:
        st.subheader("Uploaded Documents")
        docs = get_documents()
        
        st.metric("Total Indexed", docs.get("total_indexed", 0))
        
        files = docs.get("files", [])
        if files:
            for file in files:
                with st.expander(f"📄 {file['filename']}"):
                    st.write(f"**Size:** {file['size']:,} bytes")
                    st.write(f"**Path:** `{file['path']}`")
        else:
            st.info("No documents uploaded yet")
    
    with tab2:
        st.subheader("Generated Outputs")
        outputs = get_outputs()
        
        st.metric("Total Outputs", outputs.get("total", 0))
        
        output_files = outputs.get("outputs", [])
        if output_files:
            for output in output_files:
                with st.expander(f"📤 {output['filename']}"):
                    st.write(f"**Size:** {output['size']:,} bytes")
                    st.write(f"**Modified:** {output['modified']}")
                    st.write(f"**Path:** `{output['path']}`")
                    
                    # Download button
                    if st.button(f"Download {output['filename']}", key=output['filename']):
                        st.info(f"Download: {API_BASE_URL}/outputs/{output['filename']}")
        else:
            st.info("No outputs generated yet")


if __name__ == "__main__":
    main()
