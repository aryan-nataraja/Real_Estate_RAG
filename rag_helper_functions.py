from openai import OpenAI
from dotenv import load_dotenv
import os
import pdfplumber
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.vectorstores import Qdrant

load_dotenv()

def rag_openai(user_query, chunks):

    openai_api_key = os.getenv("OPENAI_API_KEY")
    MODEL="gpt-4o-mini"
    client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful HR assistant. Provide accurate information about company policies and employee benefits. Only use the chunks and context provided below."},
        {"role": "user", "content": chunks}, 
        {"role": "user", "content": user_query} 
    ]
    )

    return completion.choices[0].message.content

def read_document_and_embedd(file):
    with pdfplumber.open(file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()

    documents = [Document(page_content=full_text)]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    qdrant_client = QdrantClient(":memory:")

    collection_name = "document_chunks"
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,  # Size of the embedding vector (depends on your embedding model)
            distance=Distance.COSINE,  # Set the distance metric
        ),
    )

    qdrant = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embedding_model.embed_query,  # Embedding function
    )


    qdrant.add_texts(
        texts=[doc.page_content for doc in docs],
        metadatas=[doc.metadata for doc in docs]
    )

    return qdrant


def get_chunks(vector_store,user_query):

    results = vector_store.similarity_search_with_score(user_query, k=5)

    chunks = ""
    for i, (doc, score) in enumerate(results):
        chunks += f"Result {i + 1}\n"
        chunks += f"Text Chunk: {doc.page_content}\n"
        chunks += "-" * 50 + "\n\n" 

    return chunks

    

    