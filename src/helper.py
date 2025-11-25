import os
from PyPDF2 import PdfReader
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# 🔑 GOOGLE API KEY (Use env variable in production)
# ---------------------------------------------------------
GOOGLE_API_KEY = "AIzaSyCljuy1qHYxMoG8KaM17-KZ4G10qxGm7A8"


# ---------------------------------------------------------
# 📌 READ PDF FILES
# ---------------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            print(f"Error reading PDF {pdf.name}: {e}")
    return text


# ---------------------------------------------------------
# 📌 SPLIT INTO SMALL CHUNKS → Best for Gemini Embeddings
# ---------------------------------------------------------
def get_txt_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,     # small chunks = no timeout
        chunk_overlap=50
    )
    return splitter.split_text(text)


# ---------------------------------------------------------
# 📌 CREATE FAISS STORE (NO BATCH EMBEDDING!)
# ---------------------------------------------------------
def get_vector_store(text_chunks, api_key=GOOGLE_API_KEY):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
        request_timeout=60
    )

    # ---- ❗ CRITICAL: manual embedding to avoid batching ----
    embedded_pairs = []
    for chunk in text_chunks:
        try:
            vector = embeddings.embed_query(chunk)   # safe + small
            embedded_pairs.append((chunk, vector))
        except Exception as e:
            print(f"Embedding failed for chunk: {e}")

    # Build FAISS vector store safely
    vector_store = FAISS.from_embeddings(
        embedded_pairs,
        embeddings
    )

    return vector_store


# ---------------------------------------------------------
# 📌 CREATE CONVERSATIONAL RAG CHAIN
# ---------------------------------------------------------
def get_conversational_chain(vector_store, api_key=GOOGLE_API_KEY):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing.")

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",   # Fast + cheap + accurate
        google_api_key=api_key,
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory
    )

    return chain


# ---------------------------------------------------------
# Optional: Debug check
# ---------------------------------------------------------
if not GOOGLE_API_KEY:
    print("⚠️ Warning: GOOGLE_API_KEY not found!")
