"""
Configuration management for DocuMind
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Hugging Face Configuration
    hf_api_key: str = Field(
        default='',
        env="HF_API_KEY",
        description="Hugging Face API key for inference"
    )
    use_hf_inference_api: bool = Field(
        default=True,
        description="Use Hugging Face Inference API instead of local models"
    )
    
    # Cerebras Configuration
    cerebras_api_key: str = Field(
        default='',
        env="CEREBRAS_API_KEY",
        description="Cerebras API key for inference"
    )
    cerebras_model: str = Field(
        default="llama-4-scout-17b-16e-instruct",
        description="Cerebras model name for text generation"
    )
    
    @property
    def use_cerebras_api(self) -> bool:
        """Use Cerebras API if API key is provided"""
        return bool(self.cerebras_api_key)
    
    # Model Configuration
    model_name: str = Field(
        default="meta-llama/Llama-2-7b-chat-hf",
        description="Hugging Face model name for text generation"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    device: str = Field(
        default="cpu",
        description="Device to use for inference (cpu or cuda) - only used when use_hf_inference_api=False"
    )
    max_length: int = Field(
        default=512,
        description="Maximum length for generation"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for text generation"
    )
    chunk_size: int = Field(
        default=500,
        description="Document chunk size in words"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between document chunks"
    )
    
    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    api_port: int = Field(
        default=8000,
        description="API port"
    )
    
    # Storage Paths
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent,
        description="Base directory of the project"
    )
    
    @property
    def faiss_index_path(self) -> Path:
        path = self.base_dir / "data" / "faiss_index"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def document_store_path(self) -> Path:
        path = self.base_dir / "data" / "document_store"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def documents_path(self) -> Path:
        path = self.base_dir / "data" / "documents"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def templates_path(self) -> Path:
        path = self.base_dir / "data" / "templates"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def outputs_path(self) -> Path:
        path = self.base_dir / "data" / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
