import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- PDF PROCESSING ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            print(f"Error reading PDF file {pdf.name}: {e}")
            continue
    return text

def get_txt_chunks(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

# --- VECTOR STORE CREATION ---

def get_vector_store(text_chunks, api_key=GOOGLE_API_KEY):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key     # ✅ fixed
    )

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# --- RAG CHAT MODEL ---

def get_conversational_chain(vector_store, api_key=GOOGLE_API_KEY):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is required.")

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",   # ✅ fixed
        google_api_key=api_key,           # ❗fix here too
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return chain

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found.")
