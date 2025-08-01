# multimodal-rag-museum# 🎨 Assistant Culturel Multimodal - Projet RAG MLOps

[![CI/CD Pipeline](https://github.com/votre-username/multimodal-rag-museum/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/votre-username/multimodal-rag-museum/actions)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/votre-username/multimodal-rag-museum)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Vue d'ensemble

**Assistant Culturel Multimodal** est un projet de démonstration complet d'un système RAG (Retrieval-Augmented Generation) multimodal implémenté avec les meilleures pratiques MLOps. Le système combine analyse d'images et recherche textuelle pour créer un chatbot intelligent spécialisé dans le domaine culturel et artistique.

### ✨ Fonctionnalités principales

- 🖼️ **Analyse d'images d'œuvres d'art** avec CLIP (OpenAI)
- 📚 **Recherche textuelle sémantique** avec SentenceTransformers  
- 🔗 **Recherche multimodale combinée** (texte + image)
- 🤖 **Génération de réponses contextuelles** avec LLM
- 📊 **Monitoring MLOps complet** avec MLflow
- 🚀 **API REST performante** avec FastAPI
- 💬 **Interface utilisateur interactive** avec Streamlit
- 🐳 **Déploiement containerisé** avec Docker
- ☁️ **Déploiement cloud gratuit** sur Hugging Face Spaces

## 🏗️ Architecture du système

```mermaid
graph TB
    A[Frontend Streamlit] --> B[API FastAPI]
    B --> C[RAG Engine]
    C --> D[Vector Database ChromaDB]
    C --> E[Multimodal Embeddings]
    E --> F[CLIP Vision Model]
    E --> G[SentenceTransformer Text Model]
    B --> H[MLflow Monitoring]
    H --> I[Metrics Dashboard]
    B --> J[LLM Generation]


    # 1. Cloner le repository
git clone https://github.com/votre-username/multimodal-rag-museum.git
cd multimodal-rag-museum

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r deployment/requirements.txt

# 4. Configurer les variables d'environnement
export HF_API_TOKEN="votre_token_huggingface"

# 5. Générer les données de démonstration
python -m src.data.data_generator

# 6. Lancer l'API
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 7. Lancer l'interface (nouveau terminal)
streamlit run src/frontend/app.py --server.port 8501