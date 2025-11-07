# rag_core/pipeline.py

import os
# Stelle sicher, dass python-dotenv installiert ist (steht in requirements.txt)
from dotenv import load_dotenv 

# *******************************
# Konfigurationslogik für Umgebungsvariablen
# *******************************

# Prüfen, ob eine der kritischen Umgebungsvariablen bereits gesetzt ist (z.B. in Streamlit Cloud)
# Wenn der OPENAI_API_KEY NICHT gesetzt ist, nehmen wir an, dass wir lokal laufen
# und versuchen, die .env-Datei zu laden.
if not os.getenv("OPENAI_API_KEY"):
    # Lade die Umgebungsvariablen aus der .env-Datei im Stammverzeichnis 
    # (Diese Zeile wird in Streamlit Cloud übersprungen)
    load_dotenv() 

# Stelle sicher, dass die Variablen nun gesetzt sind, BEVOR der Code weiterläuft.
# (Dies hilft, Laufzeitfehler zu vermeiden)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # Oder PINECONE_HOST

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
    # Dies ist hilfreich für Debugging
    print("FEHLER: Nicht alle kritischen API-Schlüssel sind als Umgebungsvariable gesetzt.")
    # Du könntest hier auch raise ValueError(...) aufrufen
# *******************************


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import create_retrieval_chain
# Du benötigst den Pinecone Client und den Vektor-Speicher-Adapter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough


# --- 1. Initialisierung der Komponenten ---

# API Keys werden aus den Umgebungsvariablen gelesen (diese müssen im Streamlit Code gesetzt werden!)
# Die Dimension war 1536
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5", temperature=0)
INDEX_NAME = "ki-master"

# Pinecone Initialisierung und Retriever-Erstellung
# Stelle sicher, dass PINECONE_API_KEY und PINECONE_ENVIRONMENT (oder URL) in den Umgebungsvariablen sind!
# Wichtig: Die Pinecone Bibliothek hat sich kürzlich geändert. Wir verwenden hier die aktuelle Methode.
try:
    # 1. Pinecone Client initialisieren
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # 2. Den Vektor-Store laden
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )
    
    # 3. Den Retriever erstellen
    retriever = vectorstore.as_retriever()
    
except Exception as e:
    # Das ist wichtig für das Streamlit Deployment
    print(f"FEHLER beim Initialisieren von Pinecone/Retriever: {e}")
    retriever = None # Setze den Retriever auf None, falls die Initialisierung fehlschlägt

# --- 2. Prompt Template und Formatierungsfunktion ---

template = """Du bist ein hilfsbereiter Assistent und beantwortest die Frage 
basierend nur auf dem bereitgestellten Kontext. Wenn die Antwort nicht im Kontext enthalten ist, 
sage höflich, dass du die Antwort nicht finden kannst.

KONTEXT:
{context}

FRAGE:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    """Formatierte die Liste der Document-Objekte in einen einzigen String."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- 3. Die RAG-Kette final erstellen ---

if retriever:
    # Die LangChain Expression Language (LCEL) Kette
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    # Ersatz-Kette, falls die Datenbankverbindung fehlschlägt
    rag_chain = RunnablePassthrough() | (lambda x: "Entschuldigung, die RAG-Pipeline konnte nicht geladen werden (Datenbankverbindung fehlgeschlagen).")
    
    
# --- 4. Exponierte Funktion für Streamlit ---


def get_rag_chain_response(question: str) -> str:
    """Führt die RAG-Kette mit einer Benutzerfrage aus."""
    # Der .invoke() Aufruf in der LCEL Kette
    if retriever:
        return rag_chain.invoke(question)
    else:
        return rag_chain.invoke(None) # Ruft die Ersatz-Funktion auf, wenn der Retriever fehlt


# Beispiel für direkten Test (wird bei Import ignoriert, aber funktioniert beim direkten Ausführen)
if __name__ == "__main__":
    # Testen Sie hier Ihre Kette mit einem Environment-Setup (API Keys)
    # Beachten Sie, dass Sie hier die API-Keys manuell setzen müssten, falls Sie außerhalb von Streamlit testen
    print("Starte lokalen Test der RAG-Pipeline...")
    
    if retriever:
        test_query = "Welche Hauptthemen werden im Dokument behandelt?"
        answer = get_rag_chain_response(test_query)
        print(f"Frage: {test_query}")
        print(f"Antwort: {answer}")
    else:
        print("Test fehlgeschlagen: Retriever nicht initialisiert.")