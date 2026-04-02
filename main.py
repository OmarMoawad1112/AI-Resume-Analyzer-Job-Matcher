# ==========================================
# IMPORTS
# ==========================================

from loader import load_pdf
from chunker import chunk_text
from vectorstore import create_vectorstore
from rag import build_prompt

from huggingface_hub import InferenceClient
import json


# ==========================================
# STEP 1: LOAD CV
# ==========================================

cv_path = "cv.pdf"
text = load_pdf(cv_path)


# ==========================================
# STEP 2: CHUNKING
# ==========================================

chunks = chunk_text(text)

print(f"[INFO] Number of chunks: {len(chunks)}")


# ==========================================
# STEP 3: VECTOR DATABASE
# ==========================================

vectorstore = create_vectorstore(chunks)


# ==========================================
# STEP 4: JOB DESCRIPTION
# ==========================================

job_description = "Looking for a Python data analyst with ML experience"


# ==========================================
# STEP 5: RETRIEVAL
# ==========================================

docs = vectorstore.similarity_search(job_description, k=5)

seen = set()
clean_chunks = []

for doc in docs:
    txt = doc.page_content.strip()
    if txt not in seen:
        clean_chunks.append(txt)
        seen.add(txt)

context = "\n".join(clean_chunks)


# ==========================================
# STEP 6: PROMPT BUILDING
# ==========================================

prompt = build_prompt(context, job_description)


# ==========================================
# STEP 7: Loading Access token from .env
# ==========================================

from dotenv import load_dotenv
import os
load_dotenv()
token = os.getenv("ACCESS_TOKEN")



# ==========================================
# STEP 8: LLM CLIENT (FIXED + SAFE)
# ==========================================
from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key=os.environ["ACCESS_TOKEN"],
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct:groq", 
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=500,        # control output
    temperature=0.2
)

response_text = completion.choices[0].message.content


print(response_text)


print(completion.usage['prompt_tokens'])
print(completion.usage['completion_tokens'])
print(completion.usage['total_tokens'])