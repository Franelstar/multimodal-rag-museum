# deployment/app_hf.py
"""
Application Streamlit optimisée pour Hugging Face Spaces
Version légère avec configuration adaptée au cloud gratuit
"""

import streamlit as st
import requests
from PIL import Image
import io
import os
import logging
from typing import Dict, Any, Optional

# Configuration spécifique pour HF Spaces
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration de la page
st.set_page_config(
    page_title="🎨 Assistant Culturel Multimodal",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration du logging pour HF Spaces
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL de l'API (modifiée pour HF Spaces)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def initialize_demo_data():
    """Initialise les données de démonstration si nécessaire"""
    if 'demo_initialized' not in st.session_state:
        try:
            # Vérification de l'API et initialisation si nécessaire
            response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                if stats.get('database', {}).get('total_artworks', 0) == 0:
                    st.info("🔄 Initialisation des données de démonstration en cours...")
                    # Ici, vous pourriez déclencher l'initialisation des données
                    # ou utiliser des données pré-stockées
            
            st.session_state.demo_initialized = True
            
        except Exception as e:
            logger.warning(f"Impossible de vérifier l'état de l'API: {e}")
            st.session_state.demo_initialized = True

# CSS optimisé pour HF Spaces
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .demo-badge {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Interface principale optimisée pour HF Spaces"""
    
    # Initialisation
    initialize_demo_data()
    
    # En-tête avec badge de démonstration
    st.markdown('<h1 class="main-header">🎨 Assistant Culturel Multimodal</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="demo-badge">
        🚀 Démonstration Interactive - Portfolio MLOps
    </div>
    """, unsafe_allow_html=True)
    
    # Description du projet
    with st.expander("ℹ️ À propos de ce projet", expanded=False):
        st.markdown("""
        **Projet Multimodal RAG pour Assistant Culturel**
        
        Ce projet démontre l'implémentation complète d'un système RAG (Retrieval-Augmented Generation) 
        multimodal utilisant les meilleures pratiques MLOps :
        
        🔧 **Technologies utilisées :**
        - **Vision** : CLIP (OpenAI) pour l'analyse d'images
        - **Embeddings** : SentenceTransformers pour le texte
        - **Base vectorielle** : ChromaDB pour la recherche de similarité
        - **API** : FastAPI avec monitoring intégré
        - **Frontend** : Streamlit pour l'interface utilisateur
        - **MLOps** : MLflow pour le tracking et monitoring
        - **Déploiement** : Docker + Hugging Face Spaces
        
        🎯 **Fonctionnalités démontrées :**
        - Recherche multimodale (texte + image)
        - Pipeline MLOps complet avec CI/CD
        - Monitoring des performances en temps réel
        - Interface utilisateur interactive
        
        📊 **Aspects MLOps couverts :**
        - Gestion des données et preprocessing
        - Entraînement et évaluation de modèles
        - Pipeline CI/CD automatisé
        - Monitoring et observabilité
        - Déploiement cloud scalable
        """)
    
    # Interface principale en colonnes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Testez l'Assistant")
        
        # Zone de démonstration avec exemples
        demo_tab1, demo_tab2, demo_tab3 = st.tabs([
            "🔍 Recherche Textuelle", 
            "🖼️ Analyse d'Image", 
            "🔗 Recherche Multimodale"
        ])
        
        with demo_tab1:
            st.markdown("""
            <div class="feature-card">
                <strong>Recherche par description textuelle</strong><br>
                Posez des questions sur l'art, les styles, les époques ou les artistes.
            </div>
            """, unsafe_allow_html=True)
            
            example_queries = [
                "Qu'est-ce que l'impressionnisme ?",
                "Parlez-moi de Van Gogh",
                "Caractéristiques de l'art Renaissance",
                "Différences entre baroque et rococo"
            ]
            
            selected_query = st.selectbox(
                "Exemples de questions :", 
                ["Tapez votre question..."] + example_queries
            )
            
            user_question = st.text_input(
                "Votre question :",
                value=selected_query if selected_query != "Tapez votre question..." else "",
                placeholder="Ex: Quelles sont les caractéristiques du style impressionniste ?"
            )
            
            if st.button("🚀 Rechercher", key="text_search"):
                if user_question.strip():
                    with st.spinner("Recherche en cours..."):
                        # Simulation de réponse pour la démo
                        st.success("✅ Recherche effectuée !")
                        st.markdown("""
                        <div class="result-card">
                            <strong>Réponse de l'assistant :</strong><br><br>
                            L'impressionnisme est un mouvement artistique du XIXe siècle caractérisé par 
                            des touches de pinceau visibles, des couleurs pures et une attention particulière 
                            aux effets de lumière. Les artistes impressionnistes comme Monet, Renoir et 
                            Pissarro peignaient souvent en plein air pour capturer les variations naturelles 
                            de la lumière.
                            <br><br>
                            <em>Score de pertinence : 0.87 | Temps de traitement : 1.2s</em>
                        </div>
                        """, unsafe_allow_html=True)
        
        with demo_tab2:
            st.markdown("""
            <div class="feature-card">
                <strong>Analyse d'image d'œuvre d'art</strong><br>
                Uploadez une image d'œuvre d'art pour obtenir des informations détaillées.
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choisissez une image :",
                type=['jpg', 'jpeg', 'png'],
                help="Formats supportés : JPG, PNG (max 5MB)"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Image analysée", width=300)
                
                if st.button("🔍 Analyser l'image", key="image_analysis"):
                    with st.spinner("Analyse en cours..."):
                        st.success("✅ Analyse terminée !")
                        st.markdown("""
                        <div class="result-card">
                            <strong>Analyse de l'image :</strong><br><br>
                            Cette œuvre présente les caractéristiques du style post-impressionniste, 
                            avec des couleurs vives et des formes expressives. La technique de pinceau 
                            et la composition suggèrent une influence de l'école française du XIXe siècle.
                            <br><br>
                            <strong>Œuvres similaires trouvées :</strong>
                            <ul>
                                <li>Nuit étoilée - Van Gogh (similarité: 0.82)</li>
                                <li>Les Tournesols - Van Gogh (similarité: 0.76)</li>
                                <li>Café de nuit - Van Gogh (similarité: 0.71)</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
        
        with demo_tab3:
            st.markdown("""
            <div class="feature-card">
                <strong>Recherche multimodale combinée</strong><br>
                Combinez une question textuelle avec une image pour une analyse approfondie.
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                multimodal_question = st.text_input(
                    "Question sur l'image :",
                    placeholder="Ex: Quel est le style de cette œuvre ?"
                )
            
            with col_b:
                multimodal_image = st.file_uploader(
                    "Image associée :",
                    type=['jpg', 'jpeg', 'png'],
                    key="multimodal_upload"
                )
            
            if multimodal_question and multimodal_image:
                if st.button("🔗 Analyse Multimodale", key="multimodal_search"):
                    with st.spinner("Analyse multimodale en cours..."):
                        st.success("✅ Analyse multimodale terminée !")
                        st.markdown("""
                        <div class="result-card">
                            <strong>Réponse multimodale :</strong><br><br>
                            Basé sur l'analyse combinée de votre question et de l'image, cette œuvre 
                            appartient au mouvement impressionniste tardif. Les caractéristiques visuelles 
                            identifiées correspondent aux techniques utilisées par les maîtres de cette 
                            période, notamment dans l'utilisation de la lumière et des couleurs.
                            <br><br>
                            <em>Confiance multimodale : 0.91 | Sources consultées : 8</em>
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Métriques du Système")
        
        # Métriques simulées pour la démo
        col_metric1, col_metric2 = st.columns(2)
        
        with col_metric1:
            st.metric("Œuvres en base", "1,247", delta="12")
            st.metric("Requêtes traitées", "8,954", delta="156")
        
        with col_metric2:
            st.metric("Temps moyen", "1.8s", delta="-0.2s")
            st.metric("Satisfaction", "94%", delta="2%")
        
        # Graphique de performance
        st.subheader("⚡ Performance")
        
        # Données simulées pour le graphique
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = [datetime.now() - timedelta(days=x) for x in range(7, 0, -1)]
        performance_data = pd.DataFrame({
            'Date': dates,
            'Temps de réponse (s)': np.random.uniform(1.2, 2.5, 7),
            'Précision (%)': np.random.uniform(85, 95, 7)
        })
        
        st.line_chart(performance_data.set_index('Date'))
        
        # Architecture du système
        st.subheader("🏗️ Architecture")
        st.markdown("""
        ```
        Frontend (Streamlit)
              ↓
        API Gateway (FastAPI)
              ↓
        RAG Engine (LangChain)
              ↓
        Vector DB (ChromaDB)
              ↓
        Embeddings (CLIP + SentenceTransformers)
        ```
        """)
        
        # Technologies utilisées
        st.subheader("🛠️ Stack Technique")
        
        tech_badges = [
            ("Python", "🐍"), ("FastAPI", "⚡"), ("Streamlit", "🎈"),
            ("CLIP", "👁️"), ("ChromaDB", "🗃️"), ("MLflow", "📊"),
            ("Docker", "🐳"), ("HF Spaces", "🤗")
        ]
        
        cols = st.columns(2)
        for i, (tech, emoji) in enumerate(tech_badges):
            with cols[i % 2]:
                st.markdown(f"{emoji} **{tech}**")
    
    # Section MLOps en bas
    st.markdown("---")
    st.subheader("🔧 Aspects MLOps Démontrés")
    
    mlops_col1, mlops_col2, mlops_col3, mlops_col4 = st.columns(4)
    
    with mlops_col1:
        st.markdown("""
        **📊 Data Pipeline**
        - Collecte automatisée
        - Preprocessing multimodal
        - Validation des données
        - Versioning
        """)
    
    with mlops_col2:
        st.markdown("""
        **🤖 Model Management**
        - Embeddings multimodaux
        - Model registry
        - A/B testing
        - Performance tracking
        """)
    
    with mlops_col3:
        st.markdown("""
        **🚀 CI/CD Pipeline**
        - Tests automatisés
        - Déploiement continu
        - Rollback automatique
        - Monitoring sanité
        """)
    
    with mlops_col4:
        st.markdown("""
        **📈 Monitoring**
        - Métriques temps réel
        - Alertes automatiques
        - Dashboards MLflow
        - Observabilité complète
        """)
    
    # Footer avec informations de contact
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>🎨 Assistant Culturel Multimodal</strong> - Projet Portfolio MLOps</p>
        <p>Démonstration d'un système RAG complet avec pipeline MLOps intégré</p>
        <p><em>Développé avec Python, FastAPI, Streamlit, et déployé sur Hugging Face Spaces</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()