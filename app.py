import streamlit as st
import os

from src.helper import (
    get_pdf_text,
    get_txt_chunks,
    get_vector_store,
    get_conversational_chain
)

# ----- Set API key directly -----
GOOGLE_API_KEY = "AIzaSyDQXJ9QVm79XR5a9Ol9Bdo4D30EVwD-dEg"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  
# ---------------------------------

def user_input(user_question):
    if st.session_state.conversation is not None:
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chatHistory = response['chat_history']
            for i, message in enumerate(st.session_state.chatHistory):
                if i % 2 == 0:
                    st.write("User: ", message.content)
                else:
                    st.write("Reply: ", message.content)
        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.error("Please upload and process PDFs first.")


def main():
    st.set_page_config(page_title="PDF Chatbot")
    st.title("📘 PDF Chatbot - Ask questions from your PDFs")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    with st.sidebar:
        st.header("Upload PDF Files")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_txt_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks, GOOGLE_API_KEY)
                        st.session_state.conversation = get_conversational_chain(vector_store, GOOGLE_API_KEY)
                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")
            else:
                st.warning("Please upload at least one PDF file.")

    user_question = st.text_input("Ask something from your PDFs:")
    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
