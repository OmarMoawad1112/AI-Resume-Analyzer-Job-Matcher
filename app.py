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
# STEP 7: LLM CLIENT (FIXED + SAFE)
# ==========================================

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token="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",
)


# ==========================================
# STEP 8: LLM CALL (ROBUST)
# ==========================================

def ask_llama(prompt):
    try:
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict HR AI assistant. "
                        "Follow instructions exactly. "
                        "Return ONLY valid JSON when requested."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=400,
            temperature=0.2  # lower = less hallucination
        )

        return response.choices[0].message["content"]

    except Exception as e:
        print("[ERROR] LLM call failed:", str(e))
        return None


# ==========================================
# STEP 9: RUN MODEL
# ==========================================

import json

raw_response = ask_llama(prompt)

print("\n========== RAW LLM OUTPUT ==========\n")
print(raw_response)

try:
    result = json.loads(raw_response)
except Exception as e:
    print("\n[ERROR] Invalid JSON output from LLM")
    result = None