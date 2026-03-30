# ==========================================
# IMPORTS
# ==========================================

# Your custom RAG modules
from loader import load_pdf
from chunker import chunk_text
from vectorstore import create_vectorstore
from rag import build_prompt

# LLaMA (Hugging Face)
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# ==========================================
# STEP 1: LOAD CV (PDF → TEXT)
# ==========================================
# Extract raw text from the CV PDF file

cv_path = "cv.pdf"
text = load_pdf(cv_path)


# ==========================================
# STEP 2: CHUNKING
# ==========================================
# Split long text into smaller chunks
# This improves embedding quality and retrieval accuracy

chunks = chunk_text(text)



# ==========================================
# STEP 3: CREATE VECTOR DATABASE
# ==========================================
# Convert chunks → embeddings → store in FAISS

vectorstore = create_vectorstore(chunks)


# ==========================================
# STEP 4: DEFINE JOB DESCRIPTION
# ==========================================

job_description = "Looking for a Python data analyst with ML experience"


# ==========================================
# STEP 5: RETRIEVAL (SIMILARITY SEARCH)
# ==========================================
# Find top-k relevant chunks from CV

docs = vectorstore.similarity_search(job_description, k=5)

# Extract only text content from retrieved docs
context = "\n".join(
    [doc.page_content for doc in docs]
    )


# ==========================================
# STEP 6: BUILD PROMPT
# ==========================================
# IMPORTANT: Use LLaMA instruction format for better results
prompt = build_prompt(context,job_description)