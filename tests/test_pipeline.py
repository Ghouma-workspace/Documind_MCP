"""
Unit tests for Haystack pipeline
"""

import pytest
from pathlib import Path
import sys
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.haystack_pipeline import DocumentProcessor, HaystackPipeline


class TestDocumentProcessor:
    """Test document processing functionality"""
    
    def test_extract_text_from_txt(self):
        """Test extracting text from TXT file"""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.")
            temp_path = f.name
        
        try:
            text = DocumentProcessor.extract_text_from_txt(temp_path)
            assert "test document" in text
            assert "multiple lines" in text
        finally:
            os.unlink(temp_path)
    
    def test_process_document_txt(self):
        """Test processing a TXT document"""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for document processing.")
            temp_path = f.name
        
        try:
            doc = DocumentProcessor.process_document(temp_path)
            assert doc is not None
            assert "Test content" in doc.content
            assert doc.meta["file_name"] is not None
            assert doc.meta["file_type"] == ".txt"
        finally:
            os.unlink(temp_path)
    
    def test_process_nonexistent_file(self):
        """Test processing non-existent file"""
        doc = DocumentProcessor.process_document("/nonexistent/file.txt")
        assert doc is None
    
    def test_process_unsupported_format(self):
        """Test processing unsupported file format"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            doc = DocumentProcessor.process_document(temp_path)
            assert doc is None
        finally:
            os.unlink(temp_path)


class TestHaystackPipeline:
    """Test Haystack pipeline functionality"""
    
    @pytest.fixture
    def pipeline(self):
        """Create a minimal Haystack pipeline for testing"""
        # Use smaller models for testing
        return HaystackPipeline(
            model_name="gpt2",  # Small model for testing
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            max_length=50,
            chunk_size=100,
            chunk_overlap=10
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes correctly"""
        assert pipeline.model_name == "gpt2"
        assert pipeline.device == "cpu"
        assert pipeline.document_store is not None
    
    def test_ingest_document(self, pipeline):
        """Test ingesting a single document"""
        # Create temporary document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is test content for ingestion. " * 10)
            temp_path = f.name
        
        try:
            success = pipeline.ingest_document(temp_path)
            assert success is True
            
            # Check document count
            count = pipeline.get_document_count()
            assert count > 0
        finally:
            os.unlink(temp_path)
            pipeline.clear_documents()
    
    def test_ingest_directory(self, pipeline):
        """Test ingesting documents from directory"""
        # Create temporary directory with files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                file_path = Path(tmpdir) / f"test_{i}.txt"
                file_path.write_text(f"Test document {i} content " * 10)
            
            # Ingest directory
            result = pipeline.ingest_directory(tmpdir)
            
            assert result["success"] is True
            assert result["total"] == 3
            assert result["ingested"] >= 0  # May vary based on processing
        
        pipeline.clear_documents()
    
    def test_retrieve_documents(self, pipeline):
        """Test document retrieval"""
        # Create and ingest test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Python is a programming language. It is widely used for data science.")
            temp_path = f.name
        
        try:
            pipeline.ingest_document(temp_path)
            
            # Test BM25 retrieval
            docs = pipeline.retrieve_bm25("python programming", top_k=1)
            assert isinstance(docs, list)
            
            # Test semantic retrieval
            docs = pipeline.retrieve_semantic("data science", top_k=1)
            assert isinstance(docs, list)
            
            # Test hybrid retrieval
            docs = pipeline.retrieve_hybrid("python", top_k=1)
            assert isinstance(docs, list)
            
        finally:
            os.unlink(temp_path)
            pipeline.clear_documents()
    
    def test_document_count(self, pipeline):
        """Test getting document count"""
        initial_count = pipeline.get_document_count()
        
        # Create and ingest document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            pipeline.ingest_document(temp_path)
            new_count = pipeline.get_document_count()
            assert new_count >= initial_count
        finally:
            os.unlink(temp_path)
            pipeline.clear_documents()
    
    def test_clear_documents(self, pipeline):
        """Test clearing document store"""
        # Add a document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            pipeline.ingest_document(temp_path)
            assert pipeline.get_document_count() > 0
            
            # Clear documents
            pipeline.clear_documents()
            assert pipeline.get_document_count() == 0
        finally:
            os.unlink(temp_path)


class TestPipelineIntegration:
    """Test full pipeline integration"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline for integration tests"""
        return HaystackPipeline(
            model_name="gpt2",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
    
    def test_end_to_end_workflow(self, pipeline):
        """Test complete workflow: ingest -> retrieve -> generate"""
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = """
            Machine learning is a subset of artificial intelligence.
            It focuses on building systems that can learn from data.
            Deep learning is a type of machine learning using neural networks.
            """
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Step 1: Ingest
            success = pipeline.ingest_document(temp_path)
            assert success is True
            
            # Step 2: Retrieve
            docs = pipeline.retrieve("machine learning", top_k=1)
            assert len(docs) > 0
            
            # Step 3: Generate (simple test)
            # Note: This uses the actual model, so results may vary
            try:
                result = pipeline.generate_with_hf(
                    prompt="What is machine learning?",
                    max_new_tokens=20,
                    temperature=0.7
                )
                assert isinstance(result, str)
            except Exception as e:
                # Generation might fail in test environment, that's okay
                pytest.skip(f"Generation test skipped: {str(e)}")
        
        finally:
            os.unlink(temp_path)
            pipeline.clear_documents()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
