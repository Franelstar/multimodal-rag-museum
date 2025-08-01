# src/utils/config.py

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration centralisée du projet"""
    
    # Modèles
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"
    TEXT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2" 
    LLM_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Base de données vectorielle
    VECTOR_DB_PATH: str = "./data/vector_store"
    COLLECTION_NAME: str = "museum_artworks"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    HF_API_TOKEN: Optional[str] = "???"
    
    # Monitoring
    MLFLOW_TRACKING_URI: str = "./mlruns"
    EXPERIMENT_NAME: str = "multimodal_rag_museum"
    
    # Déploiement
    MAX_QUERY_LENGTH: int = 500
    MAX_IMAGE_SIZE: tuple = (512, 512)
    TOP_K_RESULTS: int = 5

    # LLM Settings
    LLM_MAX_TOKENS: int = 300
    LLM_TEMPERATURE: float = 0.7
    LLM_TIMEOUT: int = 30

config = Config()