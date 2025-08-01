# src/api/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
from typing import Optional, List, Dict, Any
import mlflow
import time

from ..models.rag_engine import MultimodalRAGEngine
from ..utils.config import config
from ..utils.monitoring import log_query_metrics

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Multimodal RAG Museum Assistant",
    description="Assistant culturel multimodal pour l'analyse d'œuvres d'art",
    version="1.0.0"
)

# Configuration CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du moteur RAG global
rag_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'application"""
    global rag_engine
    logger.info("Démarrage de l'API Multimodal RAG...")
    
    try:
        # Initialisation du moteur RAG
        rag_engine = MultimodalRAGEngine()
        
        # Configuration MLflow pour le monitoring
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.EXPERIMENT_NAME)
        
        logger.info("API initialisée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        raise

@app.get("/")
async def root():
    """Endpoint de santé de l'API"""
    return {
        "message": "Assistant Culturel Multimodal",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Vérification de santé détaillée"""
    try:
        stats = rag_engine.get_collection_stats()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "database": stats
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Endpoint principal pour poser une question avec image optionnelle
    
    Args:
        question: Question textuelle de l'utilisateur
        image: Image uploadée (optionnelle)
        
    Returns:
        Dict: Réponse avec contexte et métadonnées
    """
    start_time = time.time()
    
    try:
        # Validation de la longueur de la question
        if len(question) > config.MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Question trop longue (max {config.MAX_QUERY_LENGTH} caractères)"
            )
        
        # Traitement de l'image si présente
        pil_image = None
        image_description = None
        
        if image and image.content_type.startswith('image/'):
            try:
                # Lecture et conversion de l'image
                image_data = await image.read()
                pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Redimensionnement si nécessaire
                if pil_image.size[0] > config.MAX_IMAGE_SIZE[0] or pil_image.size[1] > config.MAX_IMAGE_SIZE[1]:
                    pil_image.thumbnail(config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                image_description = f"Image uploadée de taille {pil_image.size}"
                logger.info(f"Image traitée: {pil_image.size}")
                
            except Exception as e:
                logger.error(f"Erreur traitement image: {e}")
                raise HTTPException(status_code=400, detail="Erreur lors du traitement de l'image")
        
        # Recherche dans la base vectorielle
        search_results = rag_engine.search_similar(
            query=question,
            image=pil_image,
            top_k=config.TOP_K_RESULTS
        )
        
        # Génération de la réponse
        response_text = rag_engine.generate_response(
            query=question,
            context_docs=search_results,
            image_description=image_description
        )
        
        # Calcul du temps de traitement
        processing_time = time.time() - start_time
        
        # Logging des métriques
        log_query_metrics(
            query=question,
            has_image=pil_image is not None,
            num_results=len(search_results),
            processing_time=processing_time,
            avg_similarity=sum(r['similarity_score'] for r in search_results) / len(search_results) if search_results else 0
        )
        
        # Préparation de la réponse
        response = {
            "question": question,
            "answer": response_text,
            "context": [
                {
                    "title": result['metadata'].get('title', ''),
                    "artist": result['metadata'].get('artist', ''),
                    "similarity_score": result['similarity_score'],
                    "excerpt": result['document'][:200] + "..." if len(result['document']) > 200 else result['document']
                }
                for result in search_results[:3]
            ],
            "metadata": {
                "processing_time": processing_time,
                "has_image": pil_image is not None,
                "results_count": len(search_results),
                "timestamp": time.time()
            }
        }
        
        logger.info(f"Question traitée en {processing_time:.2f}s: {question[:50]}...")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la question: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.post("/search-image")
async def search_by_image(image: UploadFile = File(...)):
    """
    Recherche par image uniquement
    
    Args:
        image: Image à analyser
        
    Returns:
        Dict: Œuvres similaires trouvées
    """
    start_time = time.time()
    
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Fichier image requis")
        
        # Traitement de l'image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        pil_image.thumbnail(config.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Recherche par similarité visuelle
        results = rag_engine.search_similar(image=pil_image, top_k=config.TOP_K_RESULTS)
        
        processing_time = time.time() - start_time
        
        return {
            "similar_artworks": [
                {
                    "title": result['metadata'].get('title', ''),
                    "artist": result['metadata'].get('artist', ''),
                    "year": result['metadata'].get('year'),
                    "style": result['metadata'].get('style', ''),
                    "similarity_score": result['similarity_score'],
                    "description": result['document'][:300] + "..."
                }
                for result in results
            ],
            "processing_time": processing_time,
            "image_size": pil_image.size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la recherche par image: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la recherche par image")

@app.get("/stats")
async def get_statistics():
    """Statistiques de la base de données et performance"""
    try:
        stats = rag_engine.get_collection_stats()
        
        # Ajout de métriques supplémentaires
        with mlflow.start_run():
            # Log des métriques courantes
            mlflow.log_metric("total_artworks", stats.get("total_artworks", 0))
        
        return {
            "database": stats,
            "api_version": "1.0.0",
            "models": {
                "clip_model": config.CLIP_MODEL,
                "text_model": config.TEXT_MODEL,
                "llm_model": config.LLM_MODEL
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des statistiques: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des statistiques")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=config.API_HOST, 
        port=config.API_PORT, 
        reload=True,
        log_level="info"
    )