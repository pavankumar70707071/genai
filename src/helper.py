import os
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

GOOGLE_API_KEY = "AIzaSyDQXJ9QVm79XR5a9Ol9Bdo4D30EVwD-dEg"

# ---------------------------------------------------------
# PDF READER
# ---------------------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            print(f"❌ Error reading PDF {pdf.name}: {e}")
            continue

    print("📄 PDF Raw Text Length:", len(text))
    return text

# ---------------------------------------------------------
# CHUNKING
# ---------------------------------------------------------
def get_txt_chunks(text, chunk_size=500, chunk_overlap=100):
    if not text or len(text.strip()) == 0:
        raise ValueError("PDF contains no extractable text.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    print("🔹 Total Chunks:", len(chunks))

    if len(chunks) == 0:
        raise ValueError("Text splitter returned no chunks.")

    return chunks

# ---------------------------------------------------------
# VECTOR STORE (FAISS)
# ---------------------------------------------------------
def get_vector_store(text_chunks, api_key=GOOGLE_API_KEY):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing.")
    if not text_chunks or len(text_chunks) == 0:
        raise ValueError("Text chunks are empty — cannot create vector DB.")

    # Ensure API key is set
    os.environ["GOOGLE_API_KEY"] = api_key

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    print("🔍 Generating embeddings...")
    try:
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        print("✅ Vector Store Created Successfully")
    except Exception as e:
        raise RuntimeError(f"Error creating vector store: {e}")

    return vector_store

# ---------------------------------------------------------
# RAG CONVERSATIONAL CHAIN
# ---------------------------------------------------------
def get_conversational_chain(vector_store, api_key=GOOGLE_API_KEY):
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing.")

    os.environ["GOOGLE_API_KEY"] = api_key

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
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

    print("🤖 Conversational Chain Ready")
    return chain
