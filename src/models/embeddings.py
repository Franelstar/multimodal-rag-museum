# src/models/embeddings.py
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalEmbeddings:
    """
    Générateur d'embeddings multimodaux pour images et textes
    Combine CLIP pour les images et SentenceTransformers pour le texte
    """
    
    def __init__(self, clip_model_name: str, text_model_name: str):
        """
        Initialise les modèles d'embeddings
        
        Args:
            clip_model_name: Nom du modèle CLIP pour les images
            text_model_name: Nom du modèle pour les embeddings texte
        """
        logger.info("Chargement des modèles d'embeddings...")
        
        # Modèle CLIP pour images et texte
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        
        # Modèle SentenceTransformer pour texte uniquement
        self.text_model = SentenceTransformer(text_model_name)
        
        # Dimensions des embeddings
        self.clip_dim = self.clip_model.config.projection_dim
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        
        logger.info(f"Modèles chargés - CLIP dim: {self.clip_dim}, Text dim: {self.text_dim}")
    
    def encode_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        """
        Encode une image en vecteur d'embedding
        
        Args:
            image: Image PIL ou chemin vers l'image
            
        Returns:
            np.ndarray: Vecteur d'embedding de l'image
        """
        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # Preprocessing et encoding avec CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalisation L2
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.numpy().flatten()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage de l'image: {e}")
            raise
    
    def encode_text(self, text: str, use_clip: bool = False) -> np.ndarray:
        """
        Encode un texte en vecteur d'embedding
        
        Args:
            text: Texte à encoder
            use_clip: Si True, utilise CLIP, sinon SentenceTransformer
            
        Returns:
            np.ndarray: Vecteur d'embedding du texte
        """
        try:
            if use_clip:
                # Utilisation de CLIP pour la cohérence multimodale
                inputs = self.clip_processor(text=text, return_tensors="pt", 
                                           padding=True, truncation=True)
                
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                return text_features.numpy().flatten()
            else:
                # Utilisation de SentenceTransformer pour la richesse sémantique
                embedding = self.text_model.encode(text, normalize_embeddings=True)
                return embedding
                
        except Exception as e:
            logger.error(f"Erreur lors de l'encodage du texte: {e}")
            raise
    
    def encode_multimodal(self, text: str, image: Image.Image = None, 
                         text_weight: float = 0.7) -> np.ndarray:
        """
        Crée un embedding multimodal en combinant texte et image
        
        Args:
            text: Texte descriptif
            image: Image optionnelle
            text_weight: Poids du texte dans la fusion (0-1)
            
        Returns:
            np.ndarray: Embedding multimodal fusionné
        """
        try:
            # Embedding textuel avec CLIP pour cohérence dimensionnelle
            text_embedding = self.encode_text(text, use_clip=True)
            
            if image is not None:
                # Embedding visuel
                image_embedding = self.encode_image(image)
                
                # Fusion pondérée
                multimodal_embedding = (text_weight * text_embedding + 
                                      (1 - text_weight) * image_embedding)
                
                # Renormalisation
                multimodal_embedding = multimodal_embedding / np.linalg.norm(multimodal_embedding)
                
                return multimodal_embedding
            else:
                return text_embedding
                
        except Exception as e:
            logger.error(f"Erreur lors de la fusion multimodale: {e}")
            raise

    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux embeddings
        
        Args:
            embedding1, embedding2: Vecteurs d'embeddings
            
        Returns:
            float: Score de similarité (0-1)
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )