# 🚀 AI Resume Analyzer & Job Matcher

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)
![LLM](https://img.shields.io/badge/LLM-LLaMA%203.3-purple.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📌 Overview

**AI Resume Analyzer & Job Matcher** is an AI-powered web application that evaluates how well a candidate’s resume (CV) matches a job description using **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**.

It extracts structured insights from resumes and generates an intelligent match report to help both **recruiters** and **job seekers** make faster, data-driven decisions.

---

## ❗ Problem Statement

Recruiters often spend significant time manually reviewing resumes against job descriptions, while candidates struggle to understand how well they fit a role.

This project solves that by:
- Automating CV parsing and analysis
- Extracting relevant skills and experience
- Computing semantic similarity between CV and job descriptions
- Generating structured AI-powered feedback

---

## ✨ Key Features

- 📄 Upload and parse CVs in PDF format  
- 🧠 Extract text using LangChain `PyPDFLoader`  
- ✂️ Smart text chunking with `RecursiveCharacterTextSplitter`  
- 🔎 Vector-based semantic search for relevant content retrieval  
- 🤖 AI-powered analysis using **LLaMA 3.3 (Hugging Face Inference API)**  
- 📊 Match score calculation between CV and job description  
- 🧩 Identification of matching & missing skills  
- 💪 Strengths and weaknesses analysis  
- 🚀 Actionable recommendations for improvement  
- 🌐 Interactive web UI built with Streamlit  

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|------------|
| Frontend | Streamlit |
| Backend | Python |
| LLM | LLaMA 3.3 (via Hugging Face InferenceClient) |
| Framework | LangChain (RAG pipeline) |
| Vector Store | Custom semantic similarity search |
| Document Processing | PyPDFLoader |
| Environment Management | python-dotenv |

---

## ⚙️ How It Works

1. User uploads a CV (PDF) and enters a job description  
2. Resume is parsed using `PyPDFLoader`  
3. Text is split into chunks using `RecursiveCharacterTextSplitter`  
4. A vector store is built from embeddings  
5. Relevant sections are retrieved via similarity search  
6. A structured prompt is created with CV + job context  
7. Prompt is sent to **LLaMA 3.3 via Hugging Face API**  
8. Model returns a structured JSON response  
9. The app displays:

   - 📊 Match Score  
   - 🎯 Matching Skills  
   - ⚠️ Missing Skills  
   - 💪 Strengths  
   - 🔍 Weaknesses  
   - 🚀 Recommendations  

---

## 🖥️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/ai-resume-analyzer.git
cd ai-resume-analyzer
