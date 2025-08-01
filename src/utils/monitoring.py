# src/utils/monitoring.py
import mlflow
import mlflow.sklearn
import logging
import time
from typing import Dict, Any, Optional
import psutil
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MLOpsMonitoring:
    """
    Classe pour le monitoring MLOps du système RAG multimodal
    Suit les performances, la qualité et l'utilisation des ressources
    """
    
    def __init__(self, experiment_name: str):
        """
        Initialise le système de monitoring
        
        Args:
            experiment_name: Nom de l'expérience MLflow
        """
        self.experiment_name = experiment_name
        self.setup_mlflow()
        
        # Métriques en temps réel
        self.current_run = None
        self.query_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        logger.info(f"Monitoring initialisé pour l'expérience: {experiment_name}")
    
    def setup_mlflow(self):
        """Configure MLflow pour le tracking"""
        try:
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow configuré avec succès")
        except Exception as e:
            logger.error(f"Erreur configuration MLflow: {e}")
    
    def start_run(self, run_name: Optional[str] = None):
        """Démarre un run MLflow"""
        try:
            self.current_run = mlflow.start_run(run_name=run_name)
            return self.current_run
        except Exception as e:
            logger.error(f"Erreur démarrage run MLflow: {e}")
            return None
    
    def end_run(self):
        """Termine le run MLflow actuel"""
        try:
            if self.current_run:
                mlflow.end_run()
                self.current_run = None
        except Exception as e:
            logger.error(f"Erreur fin de run MLflow: {e}")
    
    def log_system_metrics(self):
        """Log les métriques système (CPU, RAM, etc.)"""
        try:
            with mlflow.start_run():
                # Métriques système
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                mlflow.log_metric("cpu_usage_percent", cpu_percent)
                mlflow.log_metric("memory_usage_percent", memory.percent)
                mlflow.log_metric("memory_available_gb", memory.available / (1024**3))
                mlflow.log_metric("disk_usage_percent", disk.percent)
                
                logger.debug(f"Métriques système loggées - CPU: {cpu_percent}%, RAM: {memory.percent}%")
                
        except Exception as e:
            logger.error(f"Erreur logging métriques système: {e}")
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """
        Log les métriques d'un modèle spécifique
        
        Args:
            model_name: Nom du modèle
            metrics: Dictionnaire des métriques
        """
        try:
            with mlflow.start_run():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", value)
                
                logger.info(f"Métriques loggées pour {model_name}: {metrics}")
                
        except Exception as e:
            logger.error(f"Erreur logging métriques modèle: {e}")

def log_query_metrics(query: str, has_image: bool, num_results: int, 
                     processing_time: float, avg_similarity: float):
    """
    Log les métriques d'une requête utilisateur
    
    Args:
        query: Requête de l'utilisateur
        has_image: Si une image était incluse
        num_results: Nombre de résultats trouvés
        processing_time: Temps de traitement en secondes
        avg_similarity: Score de similarité moyen
    """
    try:
        # Création d'un dictionnaire de métriques
        metrics = {
            "query_length": len(query),
            "has_image": int(has_image),
            "num_results": num_results,
            "processing_time": processing_time,
            "avg_similarity_score": avg_similarity,
            "timestamp": time.time()
        }
        
        # Log avec MLflow
        with mlflow.start_run():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            # Log de la requête comme paramètre (tronquée pour éviter la surcharge)
            mlflow.log_param("query_sample", query[:100] + "..." if len(query) > 100 else query)
            
            # Tag pour identification
            mlflow.set_tag("query_type", "multimodal" if has_image else "text_only")
        
        logger.info(f"Métriques de requête loggées: {processing_time:.2f}s, {num_results} résultats")
        
    except Exception as e:
        logger.error(f"Erreur logging métriques de requête: {e}")

def log_embedding_quality_metrics(embeddings: np.ndarray, labels: list = None):
    """
    Log les métriques de qualité des embeddings
    
    Args:
        embeddings: Matrice des embeddings
        labels: Labels optionnels pour évaluation
    """
    try:
        # Calcul de métriques sur les embeddings
        embedding_stats = {
            "embedding_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings),
            "num_embeddings": embeddings.shape[0] if len(embeddings.shape) > 1 else 1,
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "sparsity": float(np.mean(embeddings == 0))
        }
        
        # Log avec MLflow
        with mlflow.start_run():
            for metric, value in embedding_stats.items():
                mlflow.log_metric(f"embedding_{metric}", value)
        
        logger.info(f"Métriques d'embeddings loggées: {embedding_stats}")
        
    except Exception as e:
        logger.error(f"Erreur logging métriques embeddings: {e}")

class PerformanceProfiler:
    """Profiler pour analyser les performances en temps réel"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
    
    def start(self):
        """Démarre le profiling"""
        self.start_time = time.time()
        self.checkpoints = {"start": self.start_time}
    
    def checkpoint(self, name: str):
        """Ajoute un checkpoint"""
        if self.start_time is None:
            logger.warning("Profiler non démarré")
            return
        
        current_time = time.time()
        self.checkpoints[name] = current_time
        
        # Temps depuis le début
        elapsed = current_time - self.start_time
        logger.debug(f"Checkpoint '{name}': {elapsed:.3f}s depuis le début")
    
    def get_summary(self) -> Dict[str, float]:
        """Retourne un résumé des temps"""
        if not self.checkpoints or self.start_time is None:
            return {}
        
        summary = {}
        checkpoint_names = list(self.checkpoints.keys())
        
        for i in range(len(checkpoint_names) - 1):
            current_name = checkpoint_names[i]
            next_name = checkpoint_names[i + 1]
            
            duration = self.checkpoints[next_name] - self.checkpoints[current_name]
            summary[f"{current_name}_to_{next_name}"] = duration
        
        # Temps total
        summary["total_time"] = self.checkpoints[checkpoint_names[-1]] - self.start_time
        
        return summary

def create_performance_dashboard():
    """Crée un dashboard de performance pour monitoring"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from datetime import datetime, timedelta
        
        # Récupération des données MLflow
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("multimodal_rag_museum")
        
        if experiment is None:
            logger.warning("Expérience MLflow non trouvée")
            return None
        
        # Récupération des runs récents
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=100,
            order_by=["start_time DESC"]
        )
        
        if not runs:
            logger.warning("Aucun run trouvé")
            return None
        
        # Extraction des métriques pour visualisation
        processing_times = []
        timestamps = []
        similarity_scores = []
        
        for run in runs:
            metrics = run.data.metrics
            if "processing_time" in metrics:
                processing_times.append(metrics["processing_time"])
                timestamps.append(datetime.fromtimestamp(run.info.start_time / 1000))
                similarity_scores.append(metrics.get("avg_similarity_score", 0))
        
        # Création des graphiques
        fig_performance = go.Figure()
        fig_performance.add_trace(go.Scatter(
            x=timestamps,
            y=processing_times,
            mode='lines+markers',
            name='Temps de traitement (s)',
            line=dict(color='blue')
        ))
        
        fig_performance.update_layout(
            title="Performance du système en temps réel",
            xaxis_title="Temps",
            yaxis_title="Temps de traitement (secondes)",
            template="plotly_white"
        )
        
        return fig_performance
        
    except Exception as e:
        logger.error(f"Erreur création dashboard: {e}")
        return None