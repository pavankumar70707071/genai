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

GOOGLE_API_KEY = "AIzaSyCljuy1qHYxMoG8KaM17-KZ4G10qxGm7A8"


# ---------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------
def get_pdf_text(pdf_docs):
    full_text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        except Exception as e:
            print(f"Error reading {pdf.name}: {e}")
    return full_text



# ---------------------------------------
# TEXT CHUNKING
# Optimized for Gemini embeddings
# ---------------------------------------
def get_txt_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # best for Gemini
        chunk_overlap=50
    )
    return splitter.split_text(text)



# ---------------------------------------
# VECTOR STORE (FAISS)
# IMPORTANT FIX: manual embedding to avoid API batch timeout
# ---------------------------------------
def get_vector_store(text_chunks, api_key=GOOGLE_API_KEY):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
        request_timeout=60
    )

    # Manual embedding → avoids 504 errors
    embedded = []
    for chunk in text_chunks:
        vector = embeddings.embed_query(chunk)
        embedded.append((chunk, vector))

    vector_store = FAISS.from_embeddings(
        embedded,
        embeddings
    )

    return vector_store



# ---------------------------------------
# RAG CONVERSATIONAL CHAIN
# ---------------------------------------
def get_conversational_chain(vector_store, api_key=GOOGLE_API_KEY):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing.")

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
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
