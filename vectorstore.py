# Import FAISS vector store from LangChain community package
# FAISS is used to store embeddings and perform fast similarity search
from langchain_community.vectorstores import FAISS

# Import HuggingFace embedding model wrapper
# This converts text into numerical vectors (embeddings)
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vectorstore(chunks):
    """
    This function takes a list of text chunks
    and converts them into a FAISS vector database.
    
    Each chunk is converted into an embedding (vector),
    then stored for similarity search (used in RAG systems).
    """

    # Load embedding model from HuggingFace
    # "all-MiniLM-L6-v2" is a lightweight model that converts text → vectors
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Create FAISS vector store
    # Steps happening internally:
    # 1. Each text chunk → converted into embedding vector
    # 2. Vectors are stored in FAISS index
    # 3. FAISS allows fast similarity search later
    vectorstore = FAISS.from_texts(chunks, embedding_model)

    # Return the vector database for later retrieval
    return vectorstore


# Why FAISS?
    # Instead of --> looping over all chunks 
    # We --> search similar vectors fast 