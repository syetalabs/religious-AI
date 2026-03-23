# Multi Religious-AI Chatbot

**A multi-religious, multilingual AI chatbot grounded in scripture**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-red)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB)](https://react.dev/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Sponsor](https://img.shields.io/badge/Sponsor-❤️-blue)](https://fund.syetalabs.org/en/explore/multi-religious-ai-platform)

[**Live Demo**](https://religious-ai.syetalabs.org/) · [**Report a Bug**](https://github.com/syetalabs/religious-AI/issues/new?template=bug_report.md) · [**Request a Feature**](https://github.com/syetalabs/religious-AI/issues/new?template=feature_request.md)

---

##  Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Environment Variables](#environment-variables)
- [Data Pipeline](#-data-pipeline)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

##  About the Project

**Religious-AI** is an open-source AI chatbot that answers questions grounded strictly in scripture across multiple spiritual traditions. It is designed to be respectful, non-comparative, and theologically accurate — powered by a **Retrieval-Augmented Generation (RAG)** pipeline that retrieves relevant scripture passages before generating any answer.

The project was born from a need for accessible, trustworthy religious guidance in multiple languages — particularly for Sinhala and Tamil speaking communities in Sri Lanka. It is built and maintained by [Syetalabs](https://github.com/syetalabs).

> **Principle:** The AI never mixes religious teachings, never offers personal opinions, and never fabricates scripture. If it cannot find a reliable answer in the retrieved context, it says so.

---

##  Features

-  **Multi-religion support** — Buddhism and Christianity live; Hinduism and Islam coming soon
-  **Multilingual** — English, Sinhala (සිංහල), and Tamil (தமிழ்)
-  **Scripture-grounded answers** — RAG pipeline retrieves real verses before generating responses
-  **Moderation layer** — Input is moderated before and after translation to catch unsafe queries in any language
-  **Optimised for low-resource deployment** — ONNX embeddings + SQLite keep RAM under 350MB (Render free tier compatible)
-  **On-demand religion loading** — Religions load lazily with `/prepare` + `/status/{religion}` endpoints

---

##  Architecture

<img width="925" height="956" alt="Image" src="https://github.com/user-attachments/assets/a8b6e67e-bcb1-4268-9ac3-0761666418a2" />

---

##  Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI |
| **Embeddings** | ONNX Runtime (`all-MiniLM-L6-v2` exported to ONNX) |
| **Vector Search** | FAISS (`IndexFlatIP` with L2 normalisation) |
| **Database** | SQLite (`chunks.db` per religion) |
| **LLM (English)** | Groq API — `llama-3.1-8b-instant` |
| **LLM (Sinhala/Tamil)** | Groq API — `qwen/qwen3-32b` |
| **Translation** | Google Translate (via `deep-translator`) |
| **Frontend** | React 18, Vite |
| **Data Hosting** | Hugging Face Datasets (`sdevr/religious-ai-data`) |
| **Deployment** | Render (backend + frontend static site) |

---

##  Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com/) (free tier available)
- A [Notion integration token](https://www.notion.so/my-integrations)
- Git

### Backend Setup

```bash
# 1. Clone the repository
git clone https://github.com/syetalabs/religious-AI.git
cd religious-AI/backend

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the environment template and fill in your keys
cp .env.example .env

# 5. Run the development server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Frontend Setup

```bash
cd ../frontend

# 1. Install dependencies
npm install

# 2. Copy environment template
cp .env.example .env.local
# Set VITE_API_URL=http://localhost:8000 in .env.local

# 3. Start the dev server
npm run dev
```

The frontend will be available at `http://localhost:5173`.

### Environment Variables

#### Backend (`.env`)

```env
# LLM
GROQ_API_KEY=your_groq_api_key_here

# Notion
NOTION_API_KEY=your_notion_integration_token_here
NOTION_DATABASE_ID=your_notion_database_id_here

# Hugging Face
HF_DATASET_REPO=sdevr/religious-ai-data
```

> **How to get your Notion credentials:**
> 1. Go to [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations) and create a new integration to get `NOTION_API_KEY`
> 2. Open your Notion database, click **Share → Invite** your integration, then copy the database ID from the page URL: `notion.so/YOUR_WORKSPACE/<DATABASE_ID>?v=...`

#### Frontend (`.env.local`)

```env
VITE_API_URL=http://localhost:8000
```

---

##  Data Pipeline

Pre-built data files for each religion are hosted on Hugging Face at [`sdevr/religious-ai-data`](https://huggingface.co/datasets/sdevr/religious-ai-data). Only two files are stored per religion:

- `chunks.db` — SQLite database with pre-chunked, embedded scripture text
- `faiss_index.bin` — pre-built FAISS vector index

These files are **downloaded automatically at runtime** via the `/prepare/{religion}` endpoint. You do not need to run anything manually just to use the chatbot.

### Rebuilding Data from Scratch

If you are adding a new religion or regenerating the index for an existing one, the process is two steps: **fetch** the raw scripture, then **chunk and embed** it.

**Step 1 — Fetch raw scripture data**

Run `data_loader.py` from inside the relevant religion folder under `multi-religion/`:

```bash
# Buddhism
cd multi-religion/buddhism
python data_loader.py
```

This fetches and stores the raw scripture text locally inside the religion's `data/` folder.

**Step 2 — Chunk and embed**

Once the raw data is ready, run `chunk_and_embed.py` from the same folder to produce the `chunks.db` and `faiss_index.bin` files:

```bash
# Buddhism
cd multi-religion/buddhism
python chunk_and_embed.py
```

This chunks the scripture into verse-group segments, generates ONNX embeddings, writes them to `chunks.db`, and builds the FAISS index saved as `faiss_index.bin`.

Once both files are generated, upload them to the Hugging Face dataset repo to make them available at runtime.

> **Note:** Do not commit `chunks.db`, `faiss_index.bin`, or the `data/` folder to Git — these are generated locally and do not belong in the repository. `chunks.db` and `faiss_index.bin` should be uploaded to Hugging Face; `data/` can be regenerated by running `data_loader.py`.

---

##  Project Structure

```
religious-AI/
├── backend/
│   ├── main.py               # FastAPI app, route definitions
│   ├── rag_answer.py         # Answer generation (RAG + LLM)
│   ├── retrieve.py           # FAISS retrieval, per-religion state
│   ├── data_fetcher.py       # Fetches raw scripture from external APIs
│   ├── translator.py         # Google Translate integration
│   └── requirements.txt
├── frontend/
│   └── religious-chatbot/
│          ├── src/ 
│          │     ├── main.jsx       
│          │     ├── App.jsx       
│          │     ├── Chatbot.jsx       # Chat UI component
│          │     └── Landingpage.jsx   # Landing page
│          ├── public/
│          └── package.json
└── multi-religion/
    ├── buddhism/
    │   ├── data/             # Local data cache (gitignored)
    │   ├── chunk_and_embed.py
    │   └── data_loader.py    # ← Run this to rebuild Buddhism data
    ├── christianity/
    │   ├── data/
    │   ├── chunk_and_embed.py
    │   └── data_loader.py    # ← Run this to rebuild Christianity data
    └── Hindu/
        ├── data/
        ├── chunk_and_embed.py
        ├── data_loader.py    # ← Run this to rebuild Hinduism data
        ├── patch_gita.py
        └── patch_rigveda.py

```

---

##  Roadmap

| Status | Feature |
|---|---|
| ✅ Done | Buddhism — English, Sinhala, Tamil |
| ✅ Done | Christianity — English, Sinhala, Tamil |
| ✅ Done | Moderation layer (pre + post translation) |
| ✅ Done | On-demand religion loading (`/prepare` + `/status`) |
| ✅ Done | Native-language chunk retrieval for Sinhala/Tamil |
| 🔄 In Progress | Hinduism — English, Sinhala, Tamil |
| 📋 Planned | Islam — English, Sinhala, Tamil |
| 📋 Planned | Voice input/output (STT + TTS) |
| 📋 Planned | Community Q&A archive |
| 📋 Planned | Admin dashboard for religious leaders |
| 💡 Idea | Mobile app (React Native) |

---

##  Contributing

Contributions are what make open-source thrive. Whether you're fixing a typo, adding a new religion's data, improving retrieval quality, or building a new feature — all contributions are welcome.

Please read [**CONTRIBUTING.md**](CONTRIBUTING.md) for the full guide, including:
- How to set up your development environment
- How to submit a Pull Request
- How to add a new religion
- Coding standards and commit message format

Please also read our [**Code of Conduct**](CODE_OF_CONDUCT.md) before participating.

---

##  License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full text.

---

##  Acknowledgements

- [SuttaCentral](https://suttacentral.net/) — Buddhist scripture data
- [GetBible / bible-api.com](https://getbible.net/) — Bible corpus API
- [Sacred-texts.com](https://sacred-texts.com/) — Hindu scriptures
- [Groq](https://groq.com/) — LLM inference API
- [Hugging Face](https://huggingface.co/) — Dataset hosting
- [Render](https://render.com/) — Hosting

Built with ❤️ by [Syetalabs](https://github.com/syetalabs) in Sri Lanka.
