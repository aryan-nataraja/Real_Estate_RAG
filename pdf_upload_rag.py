import streamlit as st
import subprocess
from rag_helper_functions import read_document_and_embedd, get_chunks, rag_openai

st.title("Document Upload and Query System")

# Upload document
uploaded_file = st.file_uploader("Upload a Document", type=["pdf"])

if uploaded_file:
    embedded_document = read_document_and_embedd(uploaded_file)
    st.write("Document successfully uploaded and read.")
    
    # Query input from the user
    query = st.text_input("Ask a question about the document:")

    chunks = get_chunks(embedded_document, query)
    
    if query:
        # Get answer from OpenAI
        answer = rag_openai(query, chunks)
        st.write(f"Answer: {answer}")
