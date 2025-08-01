# Assistant Culturel Multimodal - Projet RAG MLOps

[![CI/CD Pipeline](https://github.com/votre-username/multimodal-rag-museum/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/votre-username/multimodal-rag-museum/actions)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/votre-username/multimodal-rag-museum)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Un système RAG (Retrieval-Augmented Generation) multimodal qui combine vision par ordinateur et traitement du langage naturel pour créer un assistant intelligent spécialisé dans l'analyse d'œuvres d'art.

## Description du projet

Ce projet démontre l'implémentation complète d'un système d'intelligence artificielle capable d'analyser simultanément des images d'œuvres d'art et des requêtes textuelles pour fournir des réponses contextualisées sur l'art et l'histoire culturelle.

**Fonctionnalités principales :**
- Analyse d'images d'œuvres d'art avec CLIP (OpenAI)
- Recherche sémantique dans une base de connaissances culturelles
- Génération de réponses contextuelles avec Mistral-7B
- Interface web interactive pour démonstrations
- Pipeline MLOps complet avec monitoring intégré

**Technologies utilisées :**
- Vision : CLIP (OpenAI) pour l'analyse multimodale
- NLP : SentenceTransformers pour les embeddings textuels
- LLM : Mistral-7B-Instruct-v0.2 via API Hugging Face
- Base vectorielle : ChromaDB pour la recherche de similarité
- API : FastAPI pour le backend REST
- Frontend : Streamlit pour l'interface utilisateur
- MLOps : MLflow pour le tracking et monitoring
- Déploiement : Docker et Hugging Face Spaces

## Installation et utilisation

### Prérequis
- Python 3.9 ou supérieur
- Token Hugging Face gratuit (https://huggingface.co/settings/tokens)
- Git

### Instructions d'installation

```bash
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
```

### Configuration des secrets

**Étape importante :** Ne jamais commiter vos tokens dans le code.

1. Créez un fichier `.env` à la racine du projet :
```bash
HF_TOKEN=votre_token_huggingface_ici
HF_API_TOKEN=votre_token_huggingface_ici
```

2. Le fichier `.env` est automatiquement ignoré par Git grâce au `.gitignore`.

3. Pour les autres développeurs, copiez `.env.example` vers `.env` et configurez vos propres tokens.

### Accès aux services

Une fois les services lancés :
- **API REST** : http://localhost:8000
- **Interface web** : http://localhost:8501
- **Documentation API** : http://localhost:8000/docs
- **Dashboard MLflow** : http://localhost:5000 (si configuré)

### Utilisation

1. **Recherche textuelle** : Posez des questions sur l'art dans l'interface web
2. **Analyse d'image** : Uploadez une photo d'œuvre d'art pour identification
3. **Recherche multimodale** : Combinez question textuelle et image uploadée
4. **API REST** : Intégrez les fonctionnalités dans vos propres applications

## Structure du projet

```
multimodal-rag-museum/
├── src/
│   ├── data/           # Pipeline de données et génération
│   ├── models/         # Modèles ML et moteur RAG
│   ├── api/            # API FastAPI
│   ├── frontend/       # Interface Streamlit
│   └── utils/          # Utilitaires et configuration
├── tests/              # Tests automatisés
├── deployment/         # Configuration déploiement
├── data/               # Données et base vectorielle
└── notebooks/          # Expérimentations Jupyter
```

## Aspects MLOps

Ce projet implémente un pipeline MLOps complet :

**Données** : Collecte automatisée depuis APIs publiques, preprocessing multimodal, validation qualité

**Modèles** : Gestion des embeddings multimodaux, évaluation continue, versioning des modèles

**CI/CD** : Tests automatisés, pipeline GitHub Actions, déploiement containerisé

**Monitoring** : Métriques temps réel avec MLflow, dashboards de performance, alertes automatiques

## Déploiement

### Local avec Docker
```bash
docker-compose -f deployment/docker-compose.yml up --build
```

### Cloud (Hugging Face Spaces)
Le projet est configuré pour déploiement automatique sur Hugging Face Spaces. Consultez le guide de déploiement dans `deployment/README.md`.

## Tests

```bash
# Tests unitaires
pytest tests/ -v

# Tests avec couverture
pytest tests/ --cov=src --cov-report=html

# Tests d'intégration API
pytest tests/test_api.py -v
```

## Contribution

1. Fork le projet
2. Créez une branche feature (`git checkout -b feature/amelioration`)
3. Committez vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrez une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Support

Pour toute question ou problème :
- Ouvrir une issue GitHub
- Consulter la documentation dans `/docs`
- Vérifier les logs d'application pour le debugging

---

**Note** : Ce projet est conçu à des fins éducatives et de démonstration des capacités MLOps modernes appliquées à l'intelligence artificielle multimodale.