# DocsGroundTruth: Learning Log

A record of what we built, why we built it that way, what went wrong, and what
we learned. One section per phase, newest last.

---

## Phase 0: Local AI setup

**Goal:** install the local AI tools and understand what each one does.

### What we did

1. **Installed Ollama** (`winget install Ollama.Ollama`). It downloads open-source
   models, runs them on the laptop, and exposes a local API at `http://localhost:11434`.
2. **Downloaded the models:**

   | Model | Role | Size |
   |---|---|---|
   | `qwen3:4b-instruct-2507-q4_K_M` | Chat model: writes answers (the "G" in RAG) | 2.5 GB |
   | `nomic-embed-text` | Embedding model: turns text into 768 numbers (the "R" in RAG) | 274 MB |
   | `qwen3:4b` | First chat model we tried; replaced (see *Problems & fixes*) | 2.5 GB |

3. **Measured the chat model** and **tested embeddings by hand** with PowerShell.
4. **Installed uv** (`winget install astral-sh.uv`), the Python project and package manager.

**Machine:** 24 GB RAM, AMD Ryzen AI 7 with a Radeon 860M (no NVIDIA GPU), so models run on the CPU.

### Key concepts

- **RAG = Retrieval + Generation.** Like an open-book exam: first find the relevant
  pages (retrieval), then answer using only those pages (generation), with citations.
  The model answers from *our documents*, not from its training data.
- **Embeddings.** An embedding model turns text into a vector (768 numbers). Texts with
  similar meaning get similar vectors, even with no words in common. **Cosine similarity**
  measures how similar two vectors are. Our experiment:

  | Compared with "How do I get approval before surgery?" | Similarity |
  |---|---|
  | "Prior authorization requirements for procedures" | **0.648** |
  | "Best pizza places in Chicago" | 0.324 |

- **Tokens and speed.** A token is a piece of a word (about ¾ of a word).
  Response time ≈ load time + input tokens ÷ read speed + output tokens ÷ write speed.
- **Parameters and quantization.** "4b" = 4 billion learned numbers. At 2 bytes each
  that's about 8 GB; quantized to about 4 bits (`q4_K_M`) it's about 2.5 GB.
- **Model tags.** `qwen3:4b-instruct-2507-q4_K_M` = family : size - type - release (2025-07) - quantization.
  **Pin full tags**: a short tag like `qwen3:4b` can be re-pointed to a different model.
- **Thinking vs instruct models.** For RAG we want direct answers:

  | Model | Output tokens | Total time | Speed |
  |---|---|---|---|
  | `qwen3:4b` (thinking) | 934 | 64 s | 15.6 tokens/s |
  | `qwen3:4b-instruct-2507-q4_K_M` | 73 | 7.6 s | 19.1 tokens/s |

- **Why local models:** free, private (documents never leave the laptop), work offline.
  Trade-off: smaller and slower than cloud AI models.

### Problems & fixes

| Problem | Cause | Fix |
|---|---|---|
| `ollama` not recognized right after installing | A terminal reads the PATH only when it starts | Open a new terminal (or restart VS Code) |
| `qwen3:4b` took 64 s and leaked `</think>` text | That tag behaves as a thinking-only model; `--think=false` can't turn thinking off | Switched to `qwen3:4b-instruct-2507-q4_K_M` |

---

## Phase 1: Project skeleton

**Goal:** a professional Python project structure, ready for code.

### What we did

1. `git init` to track every change.
2. Created `pyproject.toml`: project metadata, dependencies, build settings.
3. `uv sync`: created the virtual environment (`.venv/`) and the lock file (`uv.lock`).
4. Created the folders: `src/docsgroundtruth/`, `tests/`, `data/raw|processed|index/`, `docs/`.
5. Added `.gitignore`, `README.md` and this log, then made the first commit.

### Key concepts

- **Build system:** `pyproject.toml` tells the build tool (hatchling) what the project
  is and where the code lives (`src/` layout: code in `src/`, tests separate).
- **Virtual environment (`.venv/`):** a private Python installation per project, so
  projects never break each other's packages.
- **Lock file (`uv.lock`):** the exact version of every package. Anyone who clones the
  repo and runs `uv sync` gets an identical setup. Our 7 direct dependencies pulled in
  118 packages in total.
- **`.gitignore`:** keeps `.venv/`, caches, data and PDFs out of git (and off GitHub).
- **`.env` vs `.env.example`:** `.env` holds this machine's settings and secrets and is
  never committed; `.env.example` is the committed template. Our variables use the `DGT_` prefix.

### Problems & fixes

| Problem | Cause | Fix |
|---|---|---|
| `unknown field python-version` in `[tool.uv]` | Not a valid uv setting | Removed it; `requires-python` sets the Python version |
| `qdrant-client>=2.7.0` could not be found | That version doesn't exist (latest is 1.x) | Used real, installed versions |
| `Readme file does not exist` | `pyproject.toml` referenced a missing README.md | Created README.md |
| Hatchling: "Unable to determine which files to ship" | The `src/` layout wasn't declared | Added `[tool.hatch.build.targets.wheel] packages = ["src/docsgroundtruth"]` |
| `head` not recognized | Unix command; PowerShell is different | `Select-Object -First N` |

**Lesson:** never guess versions or config fields. Check them against the real tools. Since
then, we read the installed libraries' source code before writing code against them.

---

## Phase 2: Ingestion and vector search

**Goal:** turn the PDF into searchable chunks, and search them by meaning.

### What we built

```
cli.py  (ingest / search / stats commands)
   │
factory.py  ◄── config.py  (settings from .env)
   │ builds and connects everything
   ▼
IngestionPipeline  (pipeline.py: orchestration only)
   │ talks only to the interfaces in domain.py
   ├── DoclingChunker   (chunking.py)       PDF → chunks
   ├── OllamaEmbedder   (embedding.py)      chunks → vectors
   └── QdrantStore      (vector_store.py)   vectors → disk; search; read back

stats.py    report card for the stored chunks
timing.py   Stopwatch for per-step timings
```

```powershell
uv run docsgroundtruth ingest          # parse → chunk → embed → store
uv run docsgroundtruth search "..."    # find the chunks most similar to a question
uv run docsgroundtruth stats           # chunk counts, sizes, page coverage
uv run pytest -v                       # run the tests
```

### Key concepts

1. **Parsing with layout understanding (Docling).** It finds headings, paragraphs,
   tables, reading order and page numbers, and uses OCR for text inside images.
   Only the main *body* is chunked: page headers, footers and page numbers are
   skipped on purpose, because they would repeat in every chunk.
2. **Structure-aware chunking (HybridChunker).** Split along the document's sections,
   split anything over the limit, merge small neighbours under the same heading.
   **The limit (800 tokens) is a maximum, not a target.**
3. **The tokenizer must match the embedding model**, so "800 tokens" is measured with the right ruler.
4. **Headings are embedded with each chunk** ("Prior Authorization" + sentence beats the bare sentence).
5. **Task prefixes:** nomic-embed-text expects `search_document:` before chunks and
   `search_query:` before questions.
6. **Idempotent ingestion:** chunk ID = UUID made from file name + position, and a file's
   old chunks are deleted before its new ones are stored. Re-running never creates
   duplicates or leaves stale chunks.
7. **Qdrant (local mode):** a *collection* holds vectors of one size; each *point* is
   id + vector + payload (text and metadata); similarity = cosine. On disk:
   `.lock` (folder in use), `meta.json` (collection settings), `collection/docs/storage.sqlite`
   (the chunks). `scroll` reads all points page by page. Always close the store (it locks the folder).
8. **Architecture:** interfaces (`Protocol`s) in `domain.py`; dependencies passed in
   ("dependency injection"); one place that wires everything (`factory.py`); Docling
   imported lazily so `search` stays fast.
9. **Tests with fakes:** `FakeEmbedder` and an in-memory Qdrant replace Ollama and the disk.
   Each test is *prepare → run → check*. 18 tests run in under a second, with no Ollama needed.
10. **Fail loudly:** clear messages when Ollama is down, a model is missing, or nothing is
    indexed; a warning when Docling converts only part of a file (`PARTIAL_SUCCESS`),
    so pages are never lost silently.
11. **Measure each step:** the log shows parse / chunk / embed / store times separately.

### Results on our PDF (`AHM_250_Transcript.pdf`)

| Metric | Value |
|---|---|
| Pages | 472, every page produced text (no gaps) |
| Chunks | 1,007 |
| First ingest | 471 s (about 1 page per second), including first-time model downloads |
| Second ingest, per step | parse 433.0 s (76%) · chunk 6.3 s · embed 120.4 s (21%) · store 7.7 s · total 567.4 s |
| Re-ingest result | Still 1,007 chunks, not 2,014: idempotency confirmed on real data |
| Chunk size | smallest 15, median 132, largest 808 tokens |
| Distribution | 66% of chunks ≤ 200 tokens; 7% reached the 800 limit; 2 slightly over (probably split tables that repeat their header row) |

**Search experiments:**

| Question | Type | Top scores | What came back |
|---|---|---|---|
| "Why would a health plan combine data from many sources into one database?" | Specific, in the document | 0.74 – 0.78 | Relevant: "High Volume of Data from Multiple Sources" (p. 303) |
| "What is this document about?" | Whole-document | 0.59 – 0.65 | Loosely related chunks about documents and records |
| "best pizza in Chicago" | Not in the document | 0.45 – 0.46 | Still 3 results, all irrelevant |

### Lessons

- Chunk-based RAG is **strong at specific questions, weak at whole-document questions**
  ("what is this about?", "summarize").
- **Vector search always returns its top matches**, even when nothing is relevant. A fixed
  score cut-off is fragile; a reranker (Phase 3) and "not in the documents" answers (Phase 4) fix it properly.
- **Small chunks can depend on their neighbours:** the chunk on p. 311 starts with
  "To address these problems...", and the problems are described on p. 303.
  So the LLM should receive several chunks, and "small-to-big" retrieval is worth trying.
- Reference lists ("Additional Readings", "Notes") get indexed too and can show up as noise.
- Docling dropped 4 cells from one table: **PDF parsing is never perfect, and tables are the hardest part.**
- **Parsing dominates ingestion time** (76%, about 0.9 s per page on the CPU). Speed work
  should start there, not with embedding or storage.
- **Laptop timings vary between runs** (567 s vs 471 s for the same PDF). Measure more than
  once before drawing conclusions.

### Problems & fixes

| Problem | Cause | Fix |
|---|---|---|
| `uv run pytest` → "program not found" | Ran from the home folder; uv finds the project through `pyproject.toml` in the current folder | `cd` into `docsgroundtruth` first |
| Hugging Face "symlinks" warnings | Windows without Developer Mode | Harmless: downloads are cached as copies |

### Ideas for later

- Save parsed documents in `data/processed/`, so changing chunk settings doesn't mean
  parsing the PDF again (parsing is 76% of ingestion time: 433 of 567 s).
- "Small-to-big" retrieval: find a small chunk, then give the LLM its neighbours too.
- A stored summary of each document, to answer overview questions.
- Possibly skip reference-list sections during ingestion.

---

## Phase 2.5: Retriever and testing playground

**Goal:** ask questions in a browser instead of the terminal, and prepare the code for Phase 3.

### What we built

```
CLI  ──┐
       ├──► factory.py ──► Retriever ──► embedder + vector store
UI   ──┘                  (Phase 6: the FastAPI service joins here too)
```

- `retrieval.py`: the **Retriever**, the one place that turns a question into chunks
  (embed the question → vector search), with timings per step. The CLI and the playground
  both use it, and Phase 3 adds hybrid search and the reranker here.
- `ui/playground.py`: a **Streamlit** page with a question form, result cards (score,
  source, pages, section path, preview, full chunk on demand) and a sidebar with index info.
- `docsgroundtruth ui` command to start it, with Streamlit's email prompt and usage statistics turned off.
- A clear "index in use by another program" message, shared display helpers on `Chunk`
  (`pages_label`, `preview()`), 26 tests, and the first run of the ruff code checker.

### Key concepts

1. **Streamlit:** a free, open-source Python library plus a command. It starts a small web
   server on the laptop, and the page opens in the browser at `localhost:8501`. It's not a
   website, so documents stay local. Good for internal tools, prototypes and testing; public,
   large-scale apps use a proper frontend (e.g. React) talking to an API.
2. **The rerun model:** Streamlit re-runs the whole script on every click. So inputs go in a
   *form* (it only runs when "Search" is pressed), and slow work is *cached* (the sidebar
   summary is reused for 30 s with `st.cache_data`).
3. **One program at a time:** Qdrant's local mode locks the index folder. The playground opens
   the index only while it searches, so `ingest` can run between searches; if the index is
   busy, a clear message appears. Production fix: run Qdrant as a server (Phase 6).
4. **One retrieval path for every interface:** interfaces only *display* results; the
   Retriever does the work. No logic is duplicated.
5. **Smoke test:** Streamlit's `AppTest` loads the page without a browser and checks that it
   renders and explains an empty index. It checks the basics, not every detail.
6. **Linter (ruff):** reads the code *without running it* and flags likely bugs, unused
   imports, messy import order, outdated syntax and needless complexity (rule groups
   F, E, I, UP, B, SIM). Ruff checks how code is *written*; pytest checks what it *does*.
   **Run both before every commit.**
7. **`uv add <package>`:** adds the package to `pyproject.toml`, updates `uv.lock` and installs it, in one step.

### Results

- The playground works: questions return the relevant chunks in the browser (retrieval only;
  generated answers come in Phase 4).
- 26 tests pass in about 2 s; ruff reports *All checks passed!*

### Problems & fixes

| Problem | Cause | Fix |
|---|---|---|
| ruff `E501` in `tests/conftest.py` (101 > 100 characters) | A docstring one character too long | Shortened it |

---

## Next phases

- **Phase 3, retrieval quality:** hybrid search (BM25 keywords + vectors) and a reranker, with
  a side-by-side comparison in the playground.
- **Phase 4, generation:** answers from the chat model, with citations and "not in the documents",
  shown in the playground.
- **Phase 5, evaluation:** test questions with known answers; measure retrieval and answer
  quality; save parsed documents first, then tune chunking.
- **Phase 6, production hardening:** FastAPI service, Qdrant server mode, GitHub Actions
  (ruff + pytest on every push), README, demo, and push to GitHub.

## References

- Ollama: https://ollama.com
- Qwen3: https://github.com/QwenLM/Qwen3
- nomic-embed-text v1.5: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- Docling: https://docling-project.github.io/docling/
- Qdrant: https://qdrant.tech/documentation/
- uv: https://docs.astral.sh/uv/
- pytest: https://docs.pytest.org
- Streamlit: https://docs.streamlit.io
- Ruff: https://docs.astral.sh/ruff/
