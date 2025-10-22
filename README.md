# 🧠 DocuMind - Agentic Document Automation System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DocuMind is a production-ready, modular AI system for intelligent document automation. It combines the **Model Context Protocol (MCP)** for agent orchestration, **Haystack** for RAG-based retrieval, and **open-source Hugging Face LLMs** for local inference—no paid APIs required.

## 🎯 Features

- 📄 **Multi-format Support**: Process PDF, DOCX, and TXT documents
- 🤖 **Agentic Architecture**: Coordinated agents using MCP protocol
- � **Intelligent Chat Interface**: Natural conversation with intent detection
- �🔍 **Intelligent Retrieval**: Hybrid search with BM25 + semantic embeddings
- 🧠 **Flexible LLM Options**: 
  - ⚡ **Cerebras API** (fastest): Ultra-fast inference with cutting-edge models
  - ☁️ **Hugging Face API** (recommended): No downloads, instant start
  - 💻 **Local Models**: Complete privacy, offline capable
- 📊 **Document Classification**: Automatic categorization and metadata extraction
- ✍️ **Template Automation**: Auto-fill reports, invoices, and summaries
- 🌐 **REST API**: FastAPI endpoints for programmatic access
- 💻 **Web UI**: Streamlit interface for easy interaction

## 🏗️ Architecture

```
DocuMind/
├── agents/              # AI agents for different tasks
│   ├── reasoner_agent.py      # Task delegation and planning
│   ├── retriever_agent.py     # Document search and retrieval
│   ├── generator_agent.py     # Content generation and summarization
│   └── automation_agent.py    # Template filling and output management
├── mcp/                 # Model Context Protocol layer
│   ├── protocol.py            # Message schemas and MCP logic
│   └── router.py              # Message routing between agents
├── pipelines/           # Haystack pipelines
│   └── haystack_pipeline.py   # RAG setup with FAISS
├── data/
│   ├── documents/             # Input documents
│   ├── templates/             # Report/invoice templates
│   └── outputs/               # Generated files
├── app/                 # Web application
│   ├── main.py                # FastAPI server
│   ├── api_routes.py          # API endpoints
│   └── ui.py                  # Streamlit interface
└── tests/               # Unit tests
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 8GB+ RAM (16GB recommended for larger models)
- GPU optional (CPU inference supported)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/documind.git
cd documind
```

2. **Create virtual environment using uv**
```bash
uv venv .venv
```

3. **Activate the virtual environment**

Windows (cmd):
```cmd
.venv\Scripts\activate
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Linux/Mac:
```bash
source .venv/bin/activate
```

4. **Install dependencies**
```bash
uv pip install -r requirements.txt
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Configuration

**Option 1: Cerebras Inference API (Fastest)**

1. Get your API key from [Cerebras AI Platform](https://cerebras.ai)
2. Set environment variable:
   ```bash
   # Windows
   set CEREBRAS_API_KEY=your_api_key_here
   
   # Linux/Mac
   export CEREBRAS_API_KEY=your_api_key_here
   ```
3. The system will automatically detect and use Cerebras API when available

**Option 2: Hugging Face Inference API (Recommended)**

1. Get your API key from [Hugging Face](https://huggingface.co/settings/tokens)
2. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
3. Edit `.env` and add your key:
   ```env
   HF_API_KEY=hf_your_token_here
   USE_HF_INFERENCE_API=True
   MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
   ```

**Option 3: Local Model Inference**

Edit `.env`:
```env
USE_HF_INFERENCE_API=False
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
DEVICE=cpu  # or cuda for GPU
```

📖 **Detailed Guides**: 
- [CEREBRAS_SETUP.md](CEREBRAS_SETUP.md) for Cerebras API configuration
- [HF_API_GUIDE.md](HF_API_GUIDE.md) for Hugging Face API setup

**Other Settings:**
```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
API_HOST=0.0.0.0
API_PORT=8000
```

## 💻 Usage

### 1. Start the FastAPI Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

### 2. Launch Streamlit UI

```bash
streamlit run app/ui.py
```

Access the UI at: `http://localhost:8501`

### 3. Use the API

**Upload and process a document:**

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@document.pdf"
```

**Summarize documents:**

```bash
curl -X POST "http://localhost:8000/api/summarize" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize all uploaded documents"}'
```

**Query documents:**

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms?"}'
```

**Generate report:**

```bash
curl -X POST "http://localhost:8000/api/generate-report" \
  -H "Content-Type: application/json" \
  -d '{"template": "project_summary", "context": {}}'
```

## 🤖 Agent System

### Reasoner Agent
- Interprets user requests
- Delegates tasks to appropriate agents
- Coordinates multi-step workflows

### Retriever Agent
- Searches document store using Haystack
- Combines BM25 and semantic search
- Returns relevant document chunks

### Generator Agent
- Summarizes documents
- Generates structured content
- Uses local Hugging Face models

### Automation Agent
- Fills templates with extracted data
- Saves outputs to disk
- Manages file exports

## 📊 Document Processing Flow

1. **Ingestion**: Upload documents via API or UI
2. **Parsing**: Extract text from PDF/DOCX/TXT
3. **Indexing**: Store in FAISS vector database
4. **Reasoning**: Reasoner agent analyzes user query
5. **Retrieval**: Retriever agent finds relevant content
6. **Generation**: Generator agent produces summary/output
7. **Automation**: Automation agent saves results

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific tests:

```bash
pytest tests/test_agents.py -v
pytest tests/test_pipeline.py -v
```

### Use Different Models

Update `.env` or pass to pipeline:

```python
from pipelines.haystack_pipeline import HaystackPipeline

pipeline = HaystackPipeline(
    model_name="meta-llama/Llama-3-8B-Instruct",
    device="cuda"
)
```


### Key Technologies

- **Haystack**: Document retrieval and RAG
- **Transformers**: Hugging Face model loading
- **FAISS**: Vector similarity search
- **FastAPI**: REST API framework
- **Streamlit**: Web UI
- **Pydantic**: Data validation
- **MCP**: Agent coordination protocol


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Haystack](https://haystack.deepset.ai/) by deepset
- [Model Context Protocol](https://modelcontextprotocol.io/)


---
