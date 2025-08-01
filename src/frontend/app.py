# src/frontend/app.py
import streamlit as st
import requests
from PIL import Image
import io
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import logging

# Configuration de la page
st.set_page_config(
    page_title="Assistant Culturel Multimodal",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de l'API
API_BASE_URL = "http://localhost:8000"

# CSS personnalisé pour l'interface
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #153d5a;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #315805;
        border-left: 4px solid #4caf50;
    }
    .metrics-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialise les variables de session"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'api_stats' not in st.session_state:
        st.session_state.api_stats = {}

def check_api_health() -> bool:
    """Vérifie la santé de l'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_api_stats() -> Dict[str, Any]:
    """Récupère les statistiques de l'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats: {e}")
        return {}

def send_query(question: str, image: Image.Image = None) -> Dict[str, Any]:
    """Envoie une requête à l'API"""
    try:
        files = {}
        data = {"question": question}
        
        # Ajout de l'image si présente
        if image is not None:
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            files["image"] = ("image.jpg", img_buffer, "image/jpeg")
        
        # Envoi de la requête
        response = requests.post(
            f"{API_BASE_URL}/ask",
            data=data,
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Erreur API: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de la requête: {e}")
        return {"error": f"Erreur de connexion: {str(e)}"}

def display_message(message: Dict[str, Any], is_user: bool = False):
    """Affiche un message dans le chat"""
    css_class = "user-message" if is_user else "assistant-message"
    icon = "🧑" if is_user else "🎨"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{icon} {'Vous' if is_user else 'Assistant Culturel'}:</strong><br>
            {message.get('content', '')}
        </div>
        """, unsafe_allow_html=True)

def display_context_results(context: List[Dict[str, Any]]):
    """Affiche les résultats de contexte trouvés"""
    if not context:
        return
    
    st.subheader("Œuvres trouvées dans la recherche")
    
    for i, result in enumerate(context):
        with st.expander(f"Résultat {i+1}: {result.get('title', 'Titre inconnu')} - Score: {result.get('similarity_score', 0):.3f}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write(f"**Artiste:** {result.get('artist', 'Inconnu')}")
                st.write(f"**Similarité:** {result.get('similarity_score', 0):.3f}")
            
            with col2:
                st.write("**Extrait:**")
                st.write(result.get('excerpt', ''))

def create_metrics_dashboard(stats: Dict[str, Any]):
    """Crée un tableau de bord avec les métriques"""
    if not stats:
        return
    
    st.subheader("Statistiques du système")
    
    # Métriques principales
    col1, col2, col3 = st.columns(3)
    
    database_stats = stats.get('database', {})
    
    with col1:
        st.metric(
            label="Œuvres d'art",
            value=database_stats.get('total_artworks', 0)
        )
    
    with col2:
        st.metric(
            label="Embeddings",
            value=database_stats.get('embedding_dimension', 0)
        )
    
    with col3:
        st.metric(
            label="Version API",
            value=stats.get('api_version', '1.0.0')
        )
    
    # Informations sur les modèles
    models_info = stats.get('models', {})
    if models_info:
        st.subheader("Modèles utilisés")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Vision & Multimodal:** {models_info.get('clip_model', 'N/A')}
            
            **Embeddings Texte:** {models_info.get('text_model', 'N/A')}
            """)
        
        with col2:
            st.info(f"""
            **Modèle de Langage:** {models_info.get('llm_model', 'N/A')}
            
            **Base Vectorielle:** ChromaDB
            """)

def main():
    """Interface principale de l'application"""
    
    # Initialisation
    init_session_state()
    
    # En-tête principal
    st.markdown('<h1 class="main-header">Assistant Culturel Multimodal</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Bienvenue dans votre assistant culturel intelligent ! Posez vos questions sur l'art, 
    uploadez des images d'œuvres pour les analyser, ou combinez les deux pour une expérience multimodale.
    """)
    
    # Sidebar avec statistiques et contrôles
    with st.sidebar:
        st.header("Contrôles")
        
        # Vérification de l'état de l'API
        if st.button("🔄 Actualiser les stats", help="Recharge les statistiques du système"):
            st.session_state.api_stats = get_api_stats()
            st.experimental_rerun()
        
        # Affichage des statistiques
        if not st.session_state.api_stats:
            st.session_state.api_stats = get_api_stats()
        
        create_metrics_dashboard(st.session_state.api_stats)
        
        st.markdown("---")
        
        # Bouton pour vider l'historique
        if st.button("🗑️ Vider l'historique", help="Efface tout l'historique des conversations"):
            st.session_state.messages = []
            st.experimental_rerun()  # Utilisation de st.rerun() au lieu de st.experimental_rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Conseils d'utilisation
        - Posez des questions spécifiques sur les œuvres d'art
        - Uploadez une image pour une analyse visuelle
        - Combinez texte et image pour des requêtes multimodales
        - Explorez différents styles artistiques
        """)
    
    # Interface principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Conversation")
        
        # Affichage de l'historique des messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_message(message, is_user=message.get('is_user', False))
    
    with col2:
        st.subheader("Upload d'image")
        uploaded_image = st.file_uploader(
            "Choisissez une image d'œuvre d'art",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Formats supportés: JPG, PNG, WebP"
        )
        
        if uploaded_image is not None:
            # Affichage de l'image uploadée
            image = Image.open(uploaded_image)
            st.image(image, caption="Image uploadée", use_column_width=True)
            
            # Informations sur l'image
            st.write(f"**Taille:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
    
    # Zone de saisie pour les questions
    st.markdown("---")
    
    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_input(
            "Posez votre question sur l'art:",
            placeholder="Ex: Qui a peint La Joconde ? Quel est le style de cette œuvre ?",
            key="question_input"
        )
        
        # Bouton de soumission requis pour les formulaires Streamlit
        submit_button = st.form_submit_button(
            "Envoyer la question"
        )
    
    # Traitement de la soumission
    if submit_button and user_question.strip():
        
        # Vérification de l'état de l'API
        if not check_api_health():
            st.error("L'API n'est pas accessible. Vérifiez que le serveur est démarré.")
            return
        
        # Ajout du message utilisateur à l'historique
        user_message = {
            "content": user_question,
            "is_user": True,
            "timestamp": time.time()
        }
        st.session_state.messages.append(user_message)
        
        # Traitement de l'image uploadée
        image_to_send = None
        if uploaded_image is not None:
            image_to_send = Image.open(uploaded_image)
        
        # Envoi de la requête avec indicateur de progression
        with st.spinner("Recherche en cours..."):
            response = send_query(user_question, image_to_send)
        
        # Traitement de la réponse
        if "error" in response:
            st.error(f"Erreur: {response['error']}")
        else:
            # Ajout de la réponse à l'historique
            assistant_message = {
                "content": response.get("answer", "Pas de réponse disponible"),
                "is_user": False,
                "timestamp": time.time(),
                "metadata": response.get("metadata", {})
            }
            st.session_state.messages.append(assistant_message)
            
            # Affichage des résultats de contexte
            context = response.get("context", [])
            if context:
                display_context_results(context)
            
            # Affichage des métriques de performance
            metadata = response.get("metadata", {})
            if metadata:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Temps de traitement",
                        f"{metadata.get('processing_time', 0):.2f}s"
                    )
                
                with col2:
                    st.metric(
                        "Résultats trouvés",
                        metadata.get('results_count', 0)
                    )
                
                with col3:
                    st.metric(
                        "Image incluse",
                        "Oui" if metadata.get('has_image', False) else "Non"
                    )
        
        # Actualisation de l'affichage
        st.experimental_rerun()  # Remplacement de st.experimental_rerun() par st.rerun()

if __name__ == "__main__":
    main()