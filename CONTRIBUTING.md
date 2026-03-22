# Contributing to Religious-AI

Thank you for your interest in contributing to Religious-AI! This document explains everything you need to know to get started as a contributor.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating. By contributing, you agree to abide by it.

---

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Your First Contribution](#your-first-contribution)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Adding a New Religion](#adding-a-new-religion)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)

---

## Ways to Contribute

There are many ways to help, beyond writing code:

-  **Report bugs** via [GitHub Issues](https://github.com/syetalabs/religious-AI/issues/new?template=bug_report.md)
-  **Suggest features** via [GitHub Issues](https://github.com/syetalabs/religious-AI/issues/new?template=feature_request.md)
-  **Improve documentation** — fix typos, clarify instructions, add examples
-  **Improve translations** — help with Sinhala or Tamil output quality
-  **Add or clean scripture data** — help expand or verify the data pipeline for a religion
-  **Add a new religion** — see the [Adding a New Religion](#adding-a-new-religion) section
-  **Review open Pull Requests** — give feedback on others' contributions

---

## Reporting Bugs

Before opening a bug report, please search [existing issues](https://github.com/syetalabs/religious-AI/issues) to avoid duplicates.

When filing a bug, use the [Bug Report template](https://github.com/syetalabs/religious-AI/issues/new?template=bug_report.md) and include:

- A clear description of what went wrong
- Steps to reproduce it
- What you expected vs what actually happened
- Your OS, browser, and which religion/language was selected
- Any error logs or screenshots

---

## Requesting Features

Use the [Feature Request template](https://github.com/syetalabs/religious-AI/issues/new?template=feature_request.md) to suggest improvements. Good feature requests include:

- The problem you are trying to solve
- Your proposed solution
- Which part of the project it affects (frontend, backend, data pipeline, etc.)

> **For large changes**, please open an Issue and discuss the approach with maintainers *before* starting implementation. This avoids wasted effort if the direction needs adjustment.

---

## Your First Contribution

Not sure where to start? Look for issues tagged:

- [`good first issue`](https://github.com/syetalabs/religious-AI/labels/good%20first%20issue) — small, well-scoped tasks ideal for newcomers
- [`help wanted`](https://github.com/syetalabs/religious-AI/labels/help%20wanted) — tasks where extra hands are needed

Feel free to comment on an issue to say you're working on it so others don't duplicate the effort.

---

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com/)
- A [Notion integration token](https://www.notion.so/my-integrations)
- Git

### Backend

```bash
git clone https://github.com/syetalabs/religious-AI.git
cd religious-AI/backend

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env           # Fill in your API keys

uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd ../frontend
npm install
cp .env.example .env.local     # Set VITE_API_URL=http://localhost:8000
npm run dev
```

---

## Making Changes

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/religious-AI.git
   ```
3. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/your-bug-description
   ```
4. **Make your changes** following the [Coding Standards](#coding-standards)
5. **Commit** your changes following the [Commit Message Guidelines](#commit-message-guidelines)
6. **Push** to your fork and open a Pull Request

---

## Pull Request Process

1. Ensure your branch is up to date with `main` before opening a PR:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Open a Pull Request against the `main` branch of this repository.

3. Fill in the PR description with:
   - **What** the change does
   - **Why** it is needed
   - **How** you tested it
   - Screenshots or logs if relevant

4. A maintainer will review your PR. Be ready to make revisions based on feedback.

5. Once approved, a maintainer will merge your PR.

### PR Checklist

Before submitting, confirm the following:

- [ ] My code follows the coding standards in this document
- [ ] I have tested my changes locally
- [ ] I have not committed `.env` files, API keys, `chunks.db`, `faiss_index.bin`, or `data/` folders
- [ ] My PR addresses a single concern (one bug fix or one feature per PR)
- [ ] I have updated documentation if my change affects how the project works

---

## Adding a New Religion

Adding a new religion is one of the most impactful contributions you can make. Here is the process:

### 1. Create the religion folder

```
multi-religion/
└── your_religion/
    ├── data/                  # gitignored — raw scripture goes here
    ├── data_loader.py         # fetches raw scripture from the source API
    └── chunk_and_embed.py     # chunks, embeds, writes chunks.db + faiss_index.bin
```

### 2. Fetch and process the data

```bash
cd multi-religion/your_religion
python data_loader.py          # Step 1: fetch raw scripture into data/
python chunk_and_embed.py      # Step 2: chunk, embed, write chunks.db + faiss_index.bin
```

### 3. Upload to Hugging Face

Upload the generated `chunks.db` and `faiss_index.bin` to the Hugging Face dataset repo at `sdevr/religious-ai-data` under a subfolder named after your religion.

> Do **not** commit `chunks.db`, `faiss_index.bin`, or the `data/` folder to Git.

### 4. Register the religion in the backend

- Add routing logic in `backend/main.py`
- Add a per-religion system prompt in `backend/rag_answer.py`
- Add religion metadata in `frontend/src/religions.js`

### 5. Verify data licensing

Before submitting, confirm that the scripture source you used allows redistribution. Include a note about the data source and its license in your PR description.

---

## Coding Standards

### Python (backend)

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints on all function signatures
- Keep functions focused — one responsibility per function
- Apply bug fixes symmetrically across all equivalent code paths (e.g. a moderation fix must be applied to all language/religion branches, not just one)
- Prefer `async`/`await` for all I/O-bound operations in FastAPI routes

### JavaScript / React (frontend)

- Use functional components with hooks
- Keep components small and single-purpose; extract data into separate `.js` files
- Avoid inline styles — use CSS modules or Tailwind utility classes
- Prefer `position: fixed` with explicit pixel offsets for mobile overlays (avoid `dvh` / `overflow: hidden` which break on iOS Safari with the virtual keyboard)

### General

- Do not commit API keys, `.env` files, `chunks.db`, `faiss_index.bin`, or `data/` folders
- Keep PRs focused — one concern per PR
- Update the README or relevant docs if your change affects how the project is set up or used

---

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>: <short description>
```

| Type | When to use |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `data` | Scripture data or pipeline changes |
| `refactor` | Code restructuring without behaviour change |
| `style` | Formatting, whitespace (no logic change) |
| `chore` | Dependency updates, config changes |

**Examples:**

```bash
git commit -m "feat: add Islam data loader and chunk pipeline"
git commit -m "fix: re-moderate translated query before answer generation"
git commit -m "docs: add setup instructions for Notion API"
git commit -m "data: add Rigveda chunks for Hinduism"
```

---

If you have any questions, feel free to open an Issue.

We appreciate every contribution, big or small. 🙏