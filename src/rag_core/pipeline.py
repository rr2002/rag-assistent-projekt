# rag_core/pipeline.py

import os
# Stelle sicher, dass python-dotenv installiert ist (steht in requirements.txt)

# *******************************
# Konfigurationslogik für Umgebungsvariablen
# *******************************

# *******************************
# Bedingter Import und Laden der Umgebungsvariablen
# *******************************

# Versuche, dotenv zu importieren und zu laden (nur für lokale Entwicklung)
try:
    from dotenv import load_dotenv
    # Da die Keys im Terminal gesetzt sein könnten (höhere Prio), 
    # rufen wir load_dotenv() trotzdem auf, um alle anderen lokalen Keys zu laden.
    load_dotenv() 
    print("INFO: .env Datei lokal geladen.") # Optional: Nur zum Debuggen
except ImportError:
    # Dies ist der Pfad in Streamlit Cloud, wo dotenv nicht installiert ist.
    print("INFO: python-dotenv nicht installiert. Nutze Umgebungsvariablen/Secrets.") # Optional
    pass

# *******************************
# Abrufen und Überprüfen (funktioniert sowohl lokal als auch in der Cloud)
# *******************************
    

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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import create_retrieval_chain
# Du benötigst den Pinecone Client und den Vektor-Speicher-Adapter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import AIMessage, HumanMessage

from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever


# --- 1. Initialisierung der Komponenten ---

# API Keys werden aus den Umgebungsvariablen gelesen (diese müssen im Streamlit Code gesetzt werden!)
# Die Dimension war 1536
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5", temperature=0)
INDEX_NAME = "ki-master"

# Pinecone Initialisierung und Retriever-Erstellung
# Wir nutzen die LangChain-Methode, die oft einfacher ist und die ENV-Variable braucht.

try:
    # 1. Pinecone Initialisierung (stellt die Umgebungsvariablen sicher)
    #    LangChain's Pinecone integration liest diese Keys direkt, 
    #    wenn sie als Umgebungsvariable gesetzt sind.
    
    # Sicherstellen, dass die LangChain-Komponente die Environment erkennt:
    # Dies ist die sauberste Methode, wenn man PINECONE_API_KEY und PINECONE_ENVIRONMENT nutzt.
    
    # Es ist nicht notwendig, den Client 'pc' separat zu initialisieren, wenn LangChain es tut.
    # Entferne die Zeile: pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # **KORREKTUR:** Wir verwenden PineconeVectorStore.from_existing_index 
    # und lassen ihn die Umgebungsvariablen PINECONE_API_KEY und PINECONE_ENVIRONMENT lesen.
    
    # 2. Den Vektor-Store laden über die LangChain-spezifische Methode
    from langchain_pinecone import PineconeVectorStore # Stelle sicher, dass du dies importierst
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, 
        embedding=embeddings
    )
    
    # 3. Den Retriever erstellen
    retriever = vectorstore.as_retriever()
    
except Exception as e:
    # Gebe den genauen Fehler aus, falls die Verbindung fehlschlägt.
    print(f"FEHLER beim Initialisieren von Pinecone/Retriever: {e}")
    retriever = None



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


# RR auskommentiert 10.11.2025
# def get_rag_chain_response(question: str) -> str:
#     """Führt die RAG-Kette mit einer Benutzerfrage aus."""
#     # Der .invoke() Aufruf in der LCEL Kette
#     if retriever:
#         return rag_chain.invoke(question)
#     else:
#         return rag_chain.invoke(None) # Ruft die Ersatz-Funktion auf, wenn der Retriever fehlt

# src/rag_core/pipeline.py

# Annahme: Du hast bereits 'llm' (z.B. ChatOpenAI) und 'vectorstore' (Pinecone) initialisiert.
# Beispielhafte Initialisierungen (deine sind wahrscheinlich woanders)
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# embeddings = OpenAIEmbeddings()
# vectorstore = PineconeVectorStore(index_name="dein-index", embedding=embeddings)


def get_rag_chain_response(question: str, chat_history: list):

    # Generiert eine Antwort unter Berücksichtigung des Chat-Verlaufs.
    # Args:
    #     question (str): Die aktuelle Frage des Benutzers.
    #     chat_history (list): Eine Liste von HumanMessage und AIMessage Objekten.
    # Returns:
    #     str: Die generierte Antwort.
   
    # RR auskommentiert 10.11.2025 weil oben bereits zugewiesen
    # retriever = vectorstore.as_retriever()

    #  Generiert eine Antwort mit Reranking zur Verbesserung der Kontextqualität.

    # 1a. Cohere Reranker initialisieren
    # Der API-Schlüssel wird automatisch aus der Umgebungsvariable COHERE_API_KEY gelesen
    reranker = CohereRerank(top_n=3) # Gibt die Top 3 Dokumente nach dem Reranking zurück

    # 1b. Base Retriever konfigurieren, um MEHR Dokumente abzurufen (wichtig!)
    # Wir geben dem Reranker eine größere Auswahl, aus der er die besten auswählen kann.
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 1c. Contextual Compression Retriever erstellen
    # Dieser "wickelt" sich um den Base Retriever und den Reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=base_retriever
    )




    # 2. Prompt für die Umformulierung der Frage (Contextualizing question)
    contextualize_q_system_prompt = (
        "Angesichts eines Chat-Verlaufs und der neuesten Benutzerfrage, "
        "die sich auf den Kontext des Chat-Verlaufs beziehen könnte, "
        "formuliere eine eigenständige Frage, die ohne den Chat-Verlauf "
        "verstanden werden kann. Beantworte die Frage NICHT, "
        "sondern formuliere sie bei Bedarf nur um, andernfalls gib sie unverändert zurück."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
        # LCEL-Kette, um die Frage neu zu formulieren
    rephrase_question_chain = (
        contextualize_q_prompt
        | llm
        | StrOutputParser()
    )

    # Hilfsfunktion, um die Dokumente für das finale Prompt zu formatieren
    # RR: um überflüssige Infos aus der Rückantwort zu entfernen
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Prompt für die finale Antwortgenerierung
    qa_system_prompt = (
    "Du bist ein Assistent für Fragen-Antworten-Aufgaben. Verwende die folgenden "
    "abgerufenen Kontextinformationen, um die Frage zu beantworten.  "
    "Wenn du die Antwort nicht kennst, sage einfach, dass du es nicht weißt. "
    "Verwende maximal drei Sätze und halte die Antwort prägnant."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 4. Komposition der gesamten RAG-Kette mit LCEL
    rag_chain = (
        # Das Eingabedictionary wird durchgereicht
        RunnablePassthrough.assign(
            # Neuer Schlüssel 'context' wird hinzugefügt.
            # Er wird gefüllt, indem die Eingabe ('input', 'chat_history')
            # an die 'rephrase_question_chain' geht, deren Ergebnis (die neue Frage)
            # dann an den 'retriever' übergeben wird.
            context=rephrase_question_chain | compression_retriever  | RunnableLambda(format_docs)
        )
        # RR 11.11.25 Zeile vorher: context=rephrase_question_chain | retriever | RunnableLambda(format_docs)
        # Das erweiterte Dictionary ('input', 'chat_history', 'context')
        # wird an das finale Prompt übergeben.
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # 5. Kette aufrufen
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    
     # RR Temporärer Debug-Schritt:
    rephrased_question = rephrase_question_chain.invoke({"input": question, "chat_history": chat_history})
    print(f"Version:1 DEBUG: Original='{question}' | Rephrased='{rephrased_question}'")
    
   
    return response

    # RR auskommentiert 10.11.2025 weil oben bereits zugewiesen
    # Der .invoke() Aufruf in der LCEL Kette
    # if retriever:
    #     return response["answer"]
    # else:
    #     return rag_chain.invoke(None) # Ruft die Ersatz-Funktion auf, wenn der Retriever fehlt




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