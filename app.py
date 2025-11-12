# chatbot_app.py

# Importiere die notwendigen Module
import os
import sys

# *****************************************************************
# 1. Zum Projekt-Root-Verzeichnis navigieren (eine Ebene hoch)
# um 'src' als Top-Level-Paket zu finden.
# Annahme: 'notebooks' liegt direkt unter dem Root.
current_dir = os.path.dirname(os.path.abspath(__file__)) # Aktuelles Verzeichnis (notebooks/)
project_root = os.path.join(current_dir, os.pardir)     # Eine Ebene höher (Root-Verzeichnis)

# RR current_dirstreamlit run app.py = os.getcwd()
# RR project_root = os.path.join(current_dir, os.pardir)

# 2. Den Root-Pfad zum Python-Suchpfad hinzufügen
# Dadurch kann Python 'src' als Top-Level-Paket finden
if project_root not in sys.path:
    sys.path.append(project_root)
# *****************************************************************

# 3. Jetzt funktioniert der Import wie in der app.py
# *****************************************************************
# 1. Projekt-Root ermitteln (das Verzeichnis, in dem app.py liegt)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Root zum Python-Suchpfad hinzufügen (nur falls nicht schon vorhanden)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 3. Import der Pipeline
# from src.rag_core.pipeline import get_rag_chain_response
from src.rag_core.pipeline import stream_rag_chain_response
# *****************************************************************

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# RR auskommentiert am 11.11 sinnvoll zum Debugen
# VORÜBERGEHENDE DEBUG-FUNKTION:
# st.sidebar.markdown("---")
# st.sidebar.write(f"OPENAI KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'NICHT gesetzt'}")
# st.sidebar.write(f"PINECONE KEY: {'Set' if os.getenv('PINECONE_API_KEY') else 'NICHT gesetzt'}")
# st.sidebar.write(f"PINECONE ENV: {os.getenv('PINECONE_ENVIRONMENT')}")
# st.sidebar.markdown("---")
# ENTFERNE DIESEN CODE NACH DEM TEST

# --- Angenommen, du importierst deine rag_chain (oder definierst sie hier neu) ---
# Importiere die notwendigen LangChain-Komponenten
# from deine_rag_modul import rag_chain 


# rr def get_rag_response(question):
# rr    # ANPASSEN: Rufe hier deine definierte rag_chain auf
# rr    response = rag_chain.invoke(question) 
    

# --------------------------------------------------------------------------

# Setze den OpenAI API Key als Umgebungsvariable (LangChain/OpenAI erwartet dies)
import os
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Fehler: Der OPENAI_API_KEY fehlt in .streamlit/secrets.toml!")
    st.stop()


## Streamlit UI Konfiguration
st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📚 Digitaler Chatbot - Ask me anything")
st.caption("Verwendet gpt-5 (LLM) und text-embedding-3-small (Embeddings) mit Pinecone.")


## 1. Initialisierung des Chat-Verlaufs
# Verwende st.session_state, um den Chat-Verlauf zu speichern (wichtig bei Streamlit)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        AIMessage(content="Hallo! Ich bin Robert's digitaler Zwilling. Frag mich etwas zu meinem beruflichen Werdegang."),
    ]


## 2. Anzeige des Chat-Verlaufs
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("User"):
            st.write(message.content)


## 3. Eingabe und Verarbeitung
user_query = st.chat_input("Deine Frage...")

if user_query is not None and user_query != "":
    # 1. Füge die Benutzeranfrage zum Verlauf hinzu
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Zeige die Benutzeranfrage sofort an
    with st.chat_message("User"):
        st.write(user_query)
        
    # 2. Generiere die Antwort
    with st.chat_message("AI"):
        with st.spinner("Antwort wird generiert..."):
            # Rufe deine tatsächliche RAG-Funktion auf
            response_generator = stream_rag_chain_response(question = user_query, chat_history = st.session_state.chat_history)
            full_response = st.write_stream(response_generator)
            # st.write(ai_response)
            
    # 3. Füge die AI-Antwort zum Verlauf hinzu
    st.session_state.chat_history.append(AIMessage(content=full_response))

# Hinweis: Das Speichern des Verlaufs *nach* der Verarbeitung ist entscheidend,
# damit die neue AI-Antwort beim nächsten Rerun angezeigt wird.