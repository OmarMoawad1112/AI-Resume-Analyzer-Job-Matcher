# ==========================================
# STREAMLIT APP - AI RESUME ANALYZER
# ==========================================

import streamlit as st
import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from loader import load_pdf
from chunker import chunk_text
from vectorstore import create_vectorstore
from rag import build_prompt


# ==========================================
# ENV
# ==========================================
load_dotenv()
token = os.getenv("ACCESS_TOKEN")


# ==========================================
# STREAMLIT UI
# ==========================================
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer & Job Matcher")
st.write("Upload a CV and compare it with a job description using RAG + LLM")


# ==========================================
# INPUTS
# ==========================================
cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
job_description = st.text_area("Enter Job Description")


run_btn = st.button("Analyze")


# ==========================================
# MAIN LOGIC
# ==========================================
if run_btn:

    if cv_file is None:
        st.error("Please upload a CV PDF")
        st.stop()

    if not job_description:
        st.error("Please enter a job description")
        st.stop()


    # STEP 1: SAVE TEMP FILE
    temp_path = "temp_cv.pdf"
    with open(temp_path, "wb") as f:
        f.write(cv_file.read())


    # STEP 2: LOAD CV
    with st.spinner("Reading CV..."):
        text = load_pdf(temp_path)


    # STEP 3: CHUNKING
    with st.spinner("Chunking text..."):
        chunks = chunk_text(text)


    # STEP 4: VECTOR STORE
    with st.spinner("Building vector database..."):
        vectorstore = create_vectorstore(chunks)


    # STEP 5: RETRIEVAL
    with st.spinner("Retrieving relevant sections..."):
        docs = vectorstore.similarity_search(job_description, k=5)

        seen = set()
        clean_chunks = []

        for doc in docs:
            txt = doc.page_content.strip()
            if txt not in seen:
                clean_chunks.append(txt)
                seen.add(txt)

        context = "\n".join(clean_chunks)


    # STEP 6: PROMPT
    prompt = build_prompt(context, job_description)


    # STEP 7: LLM CLIENT
    client = InferenceClient(
        api_key=token
    )


    with st.spinner("Analyzing with LLM..."):
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct:groq",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )

    response_text = completion.choices[0].message.content


    # ==========================================
    # OUTPUT (BEAUTIFUL UI)
    # ==========================================

    data = json.loads(response_text)

    st.subheader("📊 Resume Match Report")

    # ------------------------------------------
    # MATCH SCORE (PROGRESS BAR)
    # ------------------------------------------
    score = data.get("match_score", 0)

    st.metric(label="Match Score", value=f"{score}%")
    st.progress(score / 100)

    st.divider()


    # ------------------------------------------
    # SKILLS SECTION
    # ------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Matching Skills")
        for skill in data.get("matching_skills", []):
            st.success(f"✔ {skill}")

    with col2:
        st.markdown("### ❌ Missing Skills")
        for skill in data.get("missing_skills", []):
            st.error(f"✖ {skill}")

    st.divider()


    # ------------------------------------------
    # STRENGTHS & WEAKNESSES
    # ------------------------------------------
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 💪 Strengths")
        for s in data.get("strengths", []):
            st.info(f"💡 {s}")

    with col4:
        st.markdown("### ⚠️ Weaknesses")
        for w in data.get("weaknesses", []):
            st.warning(f"⚠ {w}")

    st.divider()


    # ------------------------------------------
    # RECOMMENDATIONS
    # ------------------------------------------
    st.markdown("### 🚀 Recommendations")

    for i, r in enumerate(data.get("recommendations", []), 1):
        st.markdown(
            f"""
            <div style="
                background-color: #0f172a;
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 10px;
                border-left: 5px solid #22c55e;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            ">
                <h4 style="margin:0; color:#22c55e;">💡 Recommendation {i}</h4>
                <p style="margin:5px 0 0 0; color:#e2e8f0;">{r}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ------------------------------------------
    # TOKEN USAGE
    # ------------------------------------------
    usage = completion.usage
    print(f"Prompt Tokens: {usage["prompt_tokens"]}")
    print(f"Output Tokens: {usage["completion_tokens"]}")
    print(f"Total Tokens: {usage["total_tokens"]}")