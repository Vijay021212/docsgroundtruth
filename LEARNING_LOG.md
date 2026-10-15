# DocsGroundTruth Learning Log

A record of what was built, why, and the concepts behind it.

---

## Phase 0: Setup & Understanding

**Goal:** Install and understand the local AI stack (Ollama, embeddings, LLMs).

**What we did:**
1. **Installed Ollama** — runs open-source AI models locally without sending data to the cloud.
2. **Downloaded two models:**
   - `qwen3:4b-instruct` — a 4-billion-parameter chat model for generation (answering questions).
   - `nomic-embed-text` — turns text into 768-dimensional vectors (embeddings) for semantic search.
3. **Tested embeddings:** proved that similar meanings produce similar vectors:
   - "How do I get approval before surgery?" vs "Prior authorization requirements" → 0.648 similarity (high)
   - "How do I get approval before surgery?" vs "Best pizza in Chicago" → 0.324 similarity (low)
4. **Installed uv** — a fast Python package manager that locks exact versions for reproducibility.

**Key concepts learned:**

### RAG = Retrieval + Generation
- **Retrieval (R):** embed the question, find similar document chunks using embeddings, pick the top matches.
- **Generation (G):** send the question + matched chunks to the LLM, which reads them and answers.
- **Why it works:** the LLM only answers based on the documents you give it, not its training data, so it won't hallucinate.

### Embeddings (Vectors)
- An embedding model turns text into 768 numbers.
- Similar texts get similar vectors (point in similar directions).
- **Cosine similarity** (0 to 1) measures how similar two vectors are.
- This is how RAG finds the right chunks without keyword matching.

### Model Sizes
- `qwen3:4b` = 4 billion parameters, ~2.5 GB (compressed), runs on CPU, answers in ~19 tokens/sec.
- Smaller models are faster, good for RAG where the retrieved chunks already contain the answer.

### Thinking vs Direct Answering
- The first Qwen3 model included "reasoning" (showing its thinking before answering).
- For RAG, we want direct answers from chunks, so we use `qwen3:4b-instruct` (no thinking).
- Thinking is 8× slower and unnecessary when documents do the reasoning.

**Why local models matter:**
- Free (no API costs)
- Privacy (documents stay on your laptop)
- Works offline
- Fully reproducible (same model, same hardware = same answers)

---

## Phase 1: Project Skeleton

**Goal:** Set up a professional Python project structure, ready for code.

**What we did:**
1. **Initialized git** to track changes.
2. **Created `pyproject.toml`:**
   - Project metadata (name, version, description, author)
   - Dependencies (118 packages: ollama, qdrant-client, docling, fastapi, pydantic, numpy, torch, transformers)
   - Build system configuration (hatchling + src/ layout)
3. **Ran `uv sync`** to create a virtual environment (`.venv/`) with exact, locked versions.
4. **Created folder structure:**
   ```
   docsgroundtruth/
   ├── src/docsgroundtruth/    ← source code
   ├── tests/                  ← test files
   ├── data/raw                ← PDFs to ingest
   ├── data/processed          ← chunks after parsing
   ├── data/index              ← vector database
   └── docs/                   ← documentation
   ```
5. **Created `.gitignore`** to exclude cache, `.venv`, data, and environment files from git.
6. **Created `LEARNING_LOG.md`** (this file) for reflection and documentation.

**Key concepts learned:**

### Build systems
- `pyproject.toml` tells the build tool (hatchling) what to package and where code is.
- `[project]` section defines the project.
- `[build-system]` section defines which tool builds it.
- `[tool.hatch.build.targets.wheel]` tells where to find the source code.

### src/ layout (modern standard)
- Code goes in `src/projectname/`.
- Tests go separate in `tests/`.
- When someone `pip install docsgroundtruth`, only the code gets installed, not tests.

### Virtual environments
- `.venv/` is a sandboxed Python with your dependencies, isolated from the system.
- One `.venv` per project means projects don't interfere with each other.

### Lock files
- `uv.lock` records exact versions of all 118 packages.
- When someone clones this repo, `uv sync` installs the exact same versions.
- This is how we guarantee reproducibility (essential for AI work).

**Design decisions:**
- **Healthcare domain:** chose medical/policy documents for evaluation, because they have clear right/wrong answers.
- **Project name:** `docsgroundtruth` signals "answers grounded in truth—your documents."
- **Local models:** Qwen3 4B + Nomic Embed Text, both free and run on CPU.

---

## Next Steps

**Phase 2:** Ingestion — parse PDFs, split into chunks, embed and index them.
**Phase 3:** Retrieval — hybrid search (vectors + BM25), reranking.
**Phase 4:** Generation — LLM answers with citations.
**Phase 5:** Evaluation — measure retrieval quality and answer accuracy.
**Phase 6:** API & UI — FastAPI web server and simple frontend.

---

## References

- **Ollama:** https://ollama.ai
- **Qwen3:** https://github.com/QwenLM/Qwen3
- **Nomic Embed Text:** https://huggingface.co/nomic-ai/nomic-embed-text-v1
- **uv:** https://github.com/astral-sh/uv
- **Qdrant:** https://qdrant.tech
- **Docling:** https://ds4sd.github.io/docling/
