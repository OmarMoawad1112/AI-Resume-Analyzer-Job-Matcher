<div align="center">

# 🤖 AI Resume Analyzer & Job Matcher

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![LLaMA](https://img.shields.io/badge/LLaMA_3.3-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**An AI-powered web application that analyzes resumes and matches them against job descriptions using RAG and LLMs — giving recruiters and job seekers instant, structured insights.**

[Features](#-key-features) · [How It Works](#how-it-works) · [Installation](#-installation) · [Usage](#-usage) · [Roadmap](#-roadmap)

---

<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/PRs-Welcome-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red?style=flat-square" />

</div>

---

## 🧩 Problem Statement

Recruiters and job seekers often struggle to quickly evaluate how well a resume matches a job description. Manually reviewing CVs is time-consuming, subjective, and error-prone — especially at scale.

**AI Resume Analyzer & Job Matcher** automates this process by leveraging Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to produce a detailed, structured match report in seconds — saving time and improving hiring decisions.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **PDF Resume Upload** | Upload and parse CVs directly in PDF format |
| 🔍 **Intelligent Text Extraction** | Powered by LangChain's `PyPDFLoader` |
| ✂️ **Smart Chunking** | `RecursiveCharacterTextSplitter` for optimal context segmentation |
| 🗄️ **Vector Store** | Custom semantic vector database for similarity search |
| 🎯 **RAG Pipeline** | Retrieves the most relevant CV sections for each job description |
| 🦙 **LLM Analysis** | LLaMA 3.3 (via Hugging Face InferenceClient) for deep analysis |
| 📊 **Match Score** | Quantitative compatibility score between resume and job |
| ✅ **Skill Gap Analysis** | Identifies matching skills and missing skills at a glance |
| 💡 **Strengths & Weaknesses** | Detailed breakdown of candidate's profile vs. requirements |
| 🚀 **Actionable Recommendations** | Personalized improvement suggestions for job seekers |
| 🖥️ **Interactive UI** | Clean, user-friendly interface built with Streamlit |

---

## 🏗️ Architecture & Tech Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│              (PDF Upload + Job Description Input)               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                     Document Processing                         │
│         PyPDFLoader  →  RecursiveCharacterTextSplitter          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                       Vector Store                              │
│              Embeddings  →  Semantic Similarity Search          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      RAG Pipeline                               │
│           Context Retrieval  →  Prompt Construction             │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                LLaMA 3.3 (Hugging Face API)                     │
│                  Structured JSON Response                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                     Results Dashboard                           │
│   Match Score · Skills · Strengths · Weaknesses · Suggestions   │
└─────────────────────────────────────────────────────────────────┘
```

### 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend** | Python 3.9+ |
| **AI / LLM** | LLaMA 3.3 (`meta-llama` via Hugging Face InferenceClient) |
| **RAG Framework** | LangChain |
| **Vector Store** | Custom (semantic similarity search) |
| **Document Processing** | PyPDFLoader (LangChain) |
| **Environment Management** | `python-dotenv` |

---

<a name="how-it-works"></a>
## ⚙️ How It Works

```
1. 📤  User uploads a CV (PDF) and pastes a job description
         │
2. 📖  PyPDFLoader extracts raw text from the PDF
         │
3. ✂️   RecursiveCharacterTextSplitter segments text into chunks
         │
4. 🗄️   A vector store is built from the chunks using embeddings
         │
5. 🔎  Similarity search retrieves the most relevant CV sections
         │
6. 🧠  A structured prompt is built combining context + job description
         │
7. 🦙  Prompt is sent to LLaMA 3.3 via Hugging Face InferenceClient
         │
8. 📦  LLM returns structured JSON output
         │
9. 📊  Results are rendered in the Streamlit dashboard
```

### 📤 Output Structure

The LLM returns a structured JSON report containing:

```json
{
  "match_score": 85,
  "matching_skills": ["Python", "Machine Learning", "REST APIs"],
  "missing_skills": ["Docker", "Kubernetes"],
  "strengths": ["Strong ML background", "Relevant project experience"],
  "weaknesses": ["No cloud deployment experience"],
  "recommendations": ["Add Docker to your skill set", "Highlight API projects"]
}
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- A [Hugging Face](https://huggingface.co/) account and API token

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/your-username/ai-resume-analyzer.git
cd ai-resume-analyzer
```

**2. Create and activate a virtual environment** *(recommended)*

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the root directory:

```env
ACCESS_TOKEN=your_huggingface_api_token_here
```

> 💡 Get your free API token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**5. Launch the application**

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📖 Usage

1. **Upload Resume** — Click the file uploader and select a PDF resume
2. **Enter Job Description** — Paste the full job description into the text area
3. **Analyze** — Click the **"Analyze"** button to start the process
4. **View Results** — Review your detailed match report:
   - 📊 Match Score
   - ✅ Matching Skills
   - ❌ Missing Skills
   - 💪 Strengths
   - ⚠️ Weaknesses
   - 🚀 Recommendations

---

## 📁 Project Structure

```
ai-resume-analyzer/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
├── .env.example            # Example environment configuration
├── .gitignore              # Git ignore rules
│
├── src/
│   ├── pdf_processor.py    # PDF loading and text chunking
│   ├── vector_store.py     # Vector database creation and search
│   ├── rag_pipeline.py     # RAG retrieval and prompt construction
│   └── llm_analyzer.py     # LLM interaction and response parsing
│
└── README.md               # Project documentation
```

---

## 🔮 Roadmap

- [ ] Support for additional file formats (DOCX, TXT)
- [ ] Advanced ranking algorithms for candidate scoring
- [ ] Integration with job platforms (LinkedIn, Indeed)
- [ ] Resume improvement suggestions with auto-editing
- [ ] User authentication and history tracking
- [ ] Recruiter dashboard with batch analysis & analytics
- [ ] Multi-language resume support
- [ ] Export reports as PDF

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please make sure your changes follow the existing code style and include relevant documentation.

---

## 👤 Author

<table>
  <tr>
    <td align="center">
      <strong>Omar Mohamed</strong><br/>
      <a href="https://github.com/your-username">GitHub</a> ·
      <a href="https://linkedin.com/in/your-profile">LinkedIn</a>
    </td>
  </tr>
</table>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If you find this project useful, please give it a star!** ⭐

*Built with ❤️ using LangChain, LLaMA 3.3, and Streamlit*

</div>
