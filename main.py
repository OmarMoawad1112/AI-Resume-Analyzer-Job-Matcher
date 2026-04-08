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

cv_path = input("enter file path: ")
text = load_pdf(cv_path)


# ==========================================
# STEP 2: CHUNKING
# ==========================================

chunks = chunk_text(text)

# ==========================================
# STEP 3: VECTOR DATABASE
# ==========================================

vectorstore = create_vectorstore(chunks)


# ==========================================
# STEP 4: JOB DESCRIPTION
# ==========================================

job_description = """
Job Title: Machine Learning Engineer
Job Summary

A Machine Learning Engineer is responsible for designing, building, and deploying machine learning models that solve real-world problems. They work with large datasets, develop algorithms, and integrate models into production systems to improve business decisions and automate processes.

Key Responsibilities
Design and develop machine learning models and algorithms
Collect, clean, and preprocess large datasets
Train, test, and evaluate model performance
Deploy models into production environments
Optimize models for accuracy, scalability, and efficiency
Collaborate with data scientists, software engineers, and stakeholders
Monitor and maintain models after deployment
Stay updated with the latest trends in AI and machine learning
Required Skills

Technical Skills:

Programming languages: Python, R, or Java
Machine learning libraries: TensorFlow, PyTorch, Scikit-learn
Data manipulation: Pandas, NumPy
Knowledge of algorithms (regression, classification, clustering)
Experience with databases (SQL, NoSQL)
Understanding of data structures and algorithms

Tools & Technologies:

Cloud platforms (AWS, Azure, Google Cloud)
Big Data tools (Hadoop, Spark)
Version control (Git)
Education & Qualifications
Bachelor’s degree in Computer Science, Data Science, AI, or related field
(Preferred) Master’s degree in Machine Learning, AI, or Data Science
Strong background in mathematics, statistics, and probability
Soft Skills
Problem-solving and analytical thinking
Communication and teamwork
Attention to detail
Ability to handle large and complex data
Example Use Cases
Recommendation systems (like Netflix or Amazon)
Fraud detection systems
Image recognition and NLP applications
Predictive analytics for business decisions
Career Path
Junior ML Engineer → ML Engineer → Senior ML Engineer → AI Architect / ML Lead
"""


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
    temperature=0.5
)

response_text = completion.choices[0].message.content


print(response_text)


print(f"Prompt Tokens: {completion.usage['prompt_tokens']}")
print(f"Output Tokens: {completion.usage['completion_tokens']}")
print(f"Totak Tokens: {completion.usage['total_tokens']}")