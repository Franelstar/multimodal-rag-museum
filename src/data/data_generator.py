# src/data/data_generator.py
import requests
import json
import os
from PIL import Image
import io
from typing import List, Dict, Any
import logging
import time
from pathlib import Path

from ..models.rag_engine import MultimodalRAGEngine, ArtworkDocument

logger = logging.getLogger(__name__)

class ArtDatasetGenerator:
    """
    Générateur de dataset d'œuvres d'art pour alimenter le système RAG
    Utilise des APIs publiques et des sources ouvertes
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialise le générateur de données
        
        Args:
            data_dir: Répertoire pour sauvegarder les données
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Création des répertoires
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Générateur initialisé - Données dans {self.data_dir}")
    
    def download_met_museum_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Télécharge des données du Metropolitan Museum of Art
        
        Args:
            limit: Nombre maximum d'œuvres à télécharger
            
        Returns:
            List[Dict]: Liste des œuvres téléchargées
        """
        logger.info(f"Téléchargement de {limit} œuvres du Met Museum...")
        
        artworks = []
        base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
        
        try:
            # Récupération des IDs d'objets avec images
            search_url = f"{base_url}/search"
            params = {
                "hasImages": "true",
                "q": "painting",
                "isHighlight": "true"
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            object_ids = response.json().get("objectIDs", [])[:limit]
            logger.info(f"Trouvé {len(object_ids)} objets à traiter")
            
            # Téléchargement des détails pour chaque objet
            for i, object_id in enumerate(object_ids):
                try:
                    # Délai pour éviter la surcharge de l'API
                    time.sleep(0.5)
                    
                    object_url = f"{base_url}/objects/{object_id}"
                    obj_response = requests.get(object_url, timeout=10)
                    obj_response.raise_for_status()
                    
                    obj_data = obj_response.json()
                    
                    # Filtrage des objets avec informations complètes
                    if (obj_data.get("primaryImage") and 
                        obj_data.get("title") and 
                        obj_data.get("artistDisplayName")):
                        
                        artwork = {
                            "id": f"met_{object_id}",
                            "title": obj_data.get("title", ""),
                            "artist": obj_data.get("artistDisplayName", ""),
                            "year": obj_data.get("objectDate", ""),
                            "style": obj_data.get("classification", ""),
                            "description": self._create_description(obj_data),
                            "image_url": obj_data.get("primaryImage", ""),
                            "culture": obj_data.get("culture", ""),
                            "medium": obj_data.get("medium", ""),
                            "dimensions": obj_data.get("dimensions", ""),
                            "source": "Met Museum"
                        }
                        
                        artworks.append(artwork)
                        
                        if len(artworks) % 10 == 0:
                            logger.info(f"Traité {len(artworks)}/{limit} œuvres")
                
                except Exception as e:
                    logger.warning(f"Erreur pour l'objet {object_id}: {e}")
                    continue
            
            logger.info(f"Téléchargement terminé: {len(artworks)} œuvres collectées")
            
            # Sauvegarde des données brutes
            output_file = self.raw_dir / "met_museum_artworks.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(artworks, f, ensure_ascii=False, indent=2)
            
            return artworks
            
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement Met Museum: {e}")
            return []
    
    def _create_description(self, obj_data: Dict[str, Any]) -> str:
        """Crée une description riche à partir des métadonnées"""
        description_parts = []
        
        if obj_data.get("title"):
            description_parts.append(f"Titre: {obj_data['title']}")
        
        if obj_data.get("artistDisplayName"):
            description_parts.append(f"Artiste: {obj_data['artistDisplayName']}")
        
        if obj_data.get("objectDate"):
            description_parts.append(f"Date: {obj_data['objectDate']}")
        
        if obj_data.get("medium"):
            description_parts.append(f"Technique: {obj_data['medium']}")
        
        if obj_data.get("culture"):
            description_parts.append(f"Culture: {obj_data['culture']}")
        
        if obj_data.get("dimensions"):
            description_parts.append(f"Dimensions: {obj_data['dimensions']}")
        
        return ". ".join(description_parts)
    
    def download_images(self, artworks: List[Dict[str, Any]], 
                       max_images: int = 50) -> List[Dict[str, Any]]:
        """
        Télécharge les images des œuvres d'art
        
        Args:
            artworks: Liste des œuvres avec URLs d'images
            max_images: Nombre maximum d'images à télécharger
            
        Returns:
            List[Dict]: Œuvres avec chemins d'images locales
        """
        logger.info(f"Téléchargement des images (max: {max_images})...")
        
        images_dir = self.raw_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        updated_artworks = []
        downloaded_count = 0
        
        for artwork in artworks:
            if downloaded_count >= max_images:
                break
            
            image_url = artwork.get("image_url")
            if not image_url:
                continue
            
            try:
                # Téléchargement de l'image
                response = requests.get(image_url, timeout=15)
                response.raise_for_status()
                
                # Sauvegarde de l'image
                image_filename = f"{artwork['id']}.jpg"
                image_path = images_dir / image_filename
                
                # Conversion et redimensionnement
                image = Image.open(io.BytesIO(response.content))
                image = image.convert('RGB')
                
                # Redimensionnement pour économiser l'espace
                if image.size[0] > 512 or image.size[1] > 512:
                    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                
                image.save(image_path, 'JPEG', quality=85)
                
                # Mise à jour du chemin local
                artwork_copy = artwork.copy()
                artwork_copy["local_image_path"] = str(image_path)
                updated_artworks.append(artwork_copy)
                
                downloaded_count += 1
                
                if downloaded_count % 5 == 0:
                    logger.info(f"Images téléchargées: {downloaded_count}/{max_images}")
                
                # Délai pour éviter la surcharge
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Erreur téléchargement image {artwork['id']}: {e}")
                # Garde l'œuvre même sans image locale
                updated_artworks.append(artwork)
                continue
        
        logger.info(f"Téléchargement d'images terminé: {downloaded_count} images")
        return updated_artworks
    
    def populate_rag_engine(self, artworks: List[Dict[str, Any]]) -> None:
        """
        Alimente le moteur RAG avec les œuvres d'art
        
        Args:
            artworks: Liste des œuvres à ajouter
        """
        logger.info(f"Alimentation du moteur RAG avec {len(artworks)} œuvres...")
        
        rag_engine = MultimodalRAGEngine()
        
        added_count = 0
        for artwork in artworks:
            try:
                # Création du document artwork
                artwork_doc = ArtworkDocument(
                    id=artwork["id"],
                    title=artwork["title"],
                    artist=artwork["artist"],
                    year=self._parse_year(artwork.get("year", "")),
                    style=artwork["style"],
                    description=artwork["description"],
                    image_path=artwork.get("local_image_path"),
                    metadata={
                        "culture": artwork.get("culture", ""),
                        "medium": artwork.get("medium", ""),
                        "dimensions": artwork.get("dimensions", ""),
                        "source": artwork.get("source", "")
                    }
                )
                
                # Chargement de l'image si disponible
                image = None
                if artwork.get("local_image_path") and os.path.exists(artwork["local_image_path"]):
                    try:
                        image = Image.open(artwork["local_image_path"])
                    except Exception as e:
                        logger.warning(f"Erreur chargement image {artwork['id']}: {e}")
                
                # Ajout au moteur RAG
                rag_engine.add_artwork(artwork_doc, image)
                added_count += 1
                
                if added_count % 10 == 0:
                    logger.info(f"Œuvres ajoutées au RAG: {added_count}/{len(artworks)}")
                
            except Exception as e:
                logger.error(f"Erreur ajout œuvre {artwork.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Population du RAG terminée: {added_count} œuvres ajoutées")
        
        # Sauvegarde des statistiques
        stats = rag_engine.get_collection_stats()
        logger.info(f"Statistiques finales: {stats}")
    
    def _parse_year(self, year_string: str) -> int:
        """Extrait l'année depuis une chaîne de date"""
        import re
        
        if not year_string:
            return None
        
        # Recherche d'un nombre à 4 chiffres (année)
        match = re.search(r'\b(1[5-9]\d{2}|20\d{2})\b', str(year_string))
        if match:
            return int(match.group(1))
        
        return None
    
    def generate_sample_dataset(self) -> None:
        """Génère un dataset d'exemple complet"""
        logger.info("Génération du dataset d'exemple...")
        
        # 1. Téléchargement des données du Met Museum
        artworks = self.download_met_museum_data(limit=100)
        
        if not artworks:
            logger.error("Aucune œuvre téléchargée, arrêt du processus")
            return
        
        # 2. Téléchargement des images
        artworks_with_images = self.download_images(artworks, max_images=30)
        
        # 3. Alimentation du moteur RAG
        self.populate_rag_engine(artworks_with_images)
        
        logger.info("Génération du dataset terminée avec succès!")

if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Génération du dataset
    generator = ArtDatasetGenerator()
    generator.generate_sample_dataset()