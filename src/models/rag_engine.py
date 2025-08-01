# src/models/rag_engine.py
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import logging
from dataclasses import dataclass
import requests
import time
from openai import OpenAI

from .embeddings import MultimodalEmbeddings
from ..utils.config import config

logger = logging.getLogger(__name__)

@dataclass
class ArtworkDocument:
    """Structure pour un document d'œuvre d'art"""
    id: str
    title: str
    artist: str
    year: Optional[int]
    style: str
    description: str
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = None

class MultimodalRAGEngine:
    """
    Moteur RAG multimodal pour la recherche d'œuvres d'art
    Combine recherche vectorielle et génération de réponses
    """
    
    def __init__(self):
        """Initialise le moteur RAG avec base vectorielle et modèles"""
        logger.info("Initialisation du moteur RAG multimodal...")
        
        # Modèles d'embeddings
        self.embeddings = MultimodalEmbeddings(
            clip_model_name=config.CLIP_MODEL,
            text_model_name=config.TEXT_MODEL
        )
        
        # Base de données vectorielle ChromaDB
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Collection pour les œuvres d'art
        try:
            self.collection = self.client.get_collection(
                name=config.COLLECTION_NAME
            )
            logger.info(f"Collection existante chargée: {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"description": "Collection multimodale d'œuvres d'art"}
            )
            logger.info("Nouvelle collection créée")
    
    def add_artwork(self, artwork: ArtworkDocument, image: Image.Image = None) -> str:
        """
        Ajoute une œuvre d'art à la base vectorielle
        
        Args:
            artwork: Document artwork à ajouter
            image: Image associée (optionnelle)
            
        Returns:
            str: ID du document ajouté
        """
        try:
            # Création du texte descriptif complet
            description_text = f"""
            Titre: {artwork.title}
            Artiste: {artwork.artist}
            Année: {artwork.year or 'Inconnue'}
            Style: {artwork.style}
            Description: {artwork.description}
            """
            
            # Génération de l'embedding multimodal
            embedding = self.embeddings.encode_multimodal(
                text=description_text,
                image=image,
                text_weight=0.7
            )
            
            # Métadonnées pour la recherche
            metadata = {
                "title": artwork.title,
                "artist": artwork.artist,
                "year": artwork.year,
                "style": artwork.style,
                **(artwork.metadata or {})
            }
            
            # Ajout à la collection
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[description_text],
                metadatas=[metadata],
                ids=[artwork.id]
            )
            
            logger.info(f"Œuvre ajoutée: {artwork.title} par {artwork.artist}")
            return artwork.id
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de l'œuvre: {e}")
            raise
    
    def search_similar(self, query: str = None, image: Image.Image = None, 
                      top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recherche d'œuvres similaires par requête textuelle et/ou image
        
        Args:
            query: Requête textuelle (optionnelle)
            image: Image de requête (optionnelle)
            top_k: Nombre de résultats à retourner
            
        Returns:
            List[Dict]: Liste des œuvres similaires avec scores
        """
        if not query and image is None:
            raise ValueError("Au moins une requête (texte ou image) est requise")
        
        top_k = top_k or config.TOP_K_RESULTS
        
        try:
            # Création de l'embedding de requête
            if query and image is not None:
                # Requête multimodale
                query_embedding = self.embeddings.encode_multimodal(
                    text=query, image=image, text_weight=0.6
                )
            elif query:
                # Requête textuelle uniquement
                query_embedding = self.embeddings.encode_text(query, use_clip=True)
            else:
                # Requête par image uniquement
                query_embedding = self.embeddings.encode_image(image)
            
            # Recherche vectorielle
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Formatage des résultats
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(result)
            
            logger.info(f"Recherche effectuée: {len(formatted_results)} résultats trouvés")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            raise

    def query_mistral_free(self, prompt: str) -> str:
        """
        Utilise l'API Inference Providers de Hugging Face avec Mistral-7B-Instruct-v0.2
        
        Args:
            prompt: Le prompt à envoyer au modèle
            
        Returns:
            str: La réponse générée par le modèle
        """
        try:
            
            # Configuration du client HF avec interface OpenAI
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=config.HF_API_TOKEN,
            )
            
            completion = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
                messages=[
                    {
                        "role": "system",
                        "content": """Tu es un assistant culturel spécialisé en art. Réponds naturellement et directement.

Respecte les règles suivantes : Si on te salue : réponds poliment et propose ton aide
Si on te pose une question sur l'art, utilise le contexte fourni et la description de l'image pour répondre simplement.
Si le message ne contient pas de question, n'utilise pas le contexte et la description.
Si le contexte ne contient pas d'information pertinente, dis "Je n'ai pas trouvé d'information sur ce sujet dans ma base de données".
Si on te pose une question hors sujet (non liée à l'art), dis "Je suis spécialisé dans l'art et la culture. Posez-moi une question sur ce domaine".
Reste concis et direct.
Ne justifie jamais tes choix de réponse.
Ne répète pas le prompt dans ta réponse.
Ne donne pas d'informations sur toi-même ou sur le modèle.
Ne fais pas de suppositions sur les intentions de l'utilisateur.
Ne fais pas de digressions.
Ne donne pas de conseils ou d'opinions personnelles.
Ne parle jamais de ce qui se trouve dans le contexte si ce n'est pas dans la question.
Ne fait allidion allusion au contexte dans ta réponse, cela doit être transparent pour l'utilisateur.
Réponds en français."""
                    },  
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                timeout=config.LLM_TIMEOUT
            )
            
            response_text = completion.choices[0].message.content
            
            if response_text and response_text.strip():
                return response_text.strip()
            else:
                logger.warning("Réponse vide de Mistral")
                return self._fallback_response()
            
        except Exception as e:
            logger.error(f"Erreur avec Mistral-7B-Instruct-v0.2: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        """Réponse de fallback quand tous les LLM échouent"""
        return """Je suis désolé, mais je rencontre actuellement des difficultés techniques 
        pour générer une réponse détaillée avec les modèles de langage. Cependant, je peux 
        vous fournir les informations directement depuis ma base de données d'œuvres d'art. 
        Les résultats de recherche que j'ai trouvés sont pertinents à votre question."""
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], 
                         image_description: str = None) -> str:
        """
        Génère une réponse basée sur le contexte récupéré
        
        Args:
            query: Question de l'utilisateur
            context_docs: Documents de contexte trouvés
            image_description: Description de l'image uploadée (optionnelle)
            
        Returns:
            str: Réponse générée
        """
        try:
            # Vérification du token HF
            if not config.HF_API_TOKEN:
                logger.warning("Token HF manquant, utilisation du fallback")
                return self._generate_fallback_response(query, context_docs, image_description)
            
            # Construction du contexte
            context_text = "\n\n".join([
                f"Œuvre {i+1}: {doc['document']}" 
                for i, doc in enumerate(context_docs[:3])
            ])
            
            # Template de prompt optimisé pour Mistral
            prompt = f""" Contexte pour la question suivante: {context_text}

            {f"Description de l'image uploadée: {image_description}" if image_description else ""}

            Question: {query}"""

            # Appel à l'API Mistral
            response = self.query_mistral_free(prompt)
            
            # Nettoyage de la réponse
            if response:
                # Supprime le prompt s'il est répété dans la réponse
                if "[/INST]" in response:
                    response = response.split("[/INST]")[-1].strip()
                
                return response
            else:
                return self._generate_fallback_response(query, context_docs, image_description)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse: {e}")
            return self._generate_fallback_response(query, context_docs, image_description)
    
    
    def _generate_fallback_response(self, query: str, context_docs: List[Dict[str, Any]], 
                                   image_description: str = None) -> str:
        """Génère une réponse de secours basée sur le contexte sans LLM"""
        try:
            if context_docs:
                best_match = context_docs[0]
                metadata = best_match['metadata']
                
                response = f"""Basé sur ma recherche, je peux vous parler de "{metadata.get('title', 'cette œuvre')}".

                Cette œuvre a été créée par {metadata.get('artist', 'un artiste')} {f"en {metadata['year']}" if metadata.get('year') else ""} et appartient au style {metadata.get('style', 'artistique')}.

                {best_match['document'][:300]}{'...' if len(best_match['document']) > 300 else ''}

                Score de pertinence: {best_match['similarity_score']:.2f}

                {f"Note: Analyse basée sur l'image uploadée - {image_description}" if image_description else ""}"""
                                
                return response.strip()
            else:
                return "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question. Essayez de reformuler ou d'ajouter plus de détails."
                
        except Exception as e:
            logger.error(f"Erreur dans la réponse de fallback: {e}")
            return "Désolé, je n'ai pas pu générer une réponse appropriée. Veuillez réessayer."
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la collection"""
        try:
            count = self.collection.count()
            return {
                "total_artworks": count,
                "collection_name": config.COLLECTION_NAME,
                "embedding_dimension": self.embeddings.clip_dim
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {"error": str(e)}