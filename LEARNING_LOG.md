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

## Phase 3: Retrieval quality (hybrid search + reranker)

**Goal:** find better chunks by combining two kinds of search, then re-checking the best candidates.

### What we built

```
                 ┌─► vector search  (meaning)      → top 30 ─┐
 question ───────┤                                            ├─► fusion → top 30 ─► reranker → best 5
                 └─► keyword search (exact words)  → top 30 ─┘
```

- `bm25.py`: turns text into BM25 keyword vectors, using fastembed.
- `reranking.py`: runs the reranker model and converts its scores to 0–1.
- `retrieval.py`: four search modes (vector, keyword, hybrid, hybrid + reranker) and our own RRF function.
- `vector_store.py`: each chunk now stores its text and two vectors, and there are two search methods.
- Playground: a search mode choice and a Compare view showing all four modes side by side.
- 41 tests, using fakes for the keyword encoder and the reranker.

### Key concepts

1. **BM25** is a scoring formula, not a library or an AI model. "BM" means Best Matching, and 25
   is the version of the formula. Rare words count more, repeated mentions have diminishing
   returns, and long chunks are adjusted for length.
2. **Sparse vector:** one slot per possible word, almost all empty, so only the filled slots are stored.
   fastembed computes each chunk's word weights, and Qdrant adds the rarity weights (`Modifier.IDF`).
3. **Hybrid search** uses both methods because they fail differently: vector search handles
   different wording, and BM25 handles exact terms.
4. **Reciprocal Rank Fusion (RRF)** merges ranked lists using positions, not scores, because
   the scores are on different scales. Each list gives a chunk `1 / (60 + position)` points.
5. **Reranker (cross-encoder):** reads the question and one chunk together and judges how well
   the chunk answers it. It's more accurate but slower, so it only checks the top 30 candidates.
6. **Model engines:** Ollama runs GGUF models, PyTorch runs Docling's models, and ONNX Runtime
   runs the reranker on the CPU.
7. **Schema change:** adding a second vector changed the index format, so the index had to be
   deleted and rebuilt. An old index now gets a clear message explaining this.

### Results on our PDF

Top score per method:

| Question | Vector | Keyword | Hybrid | Reranker |
|---|---|---|---|---|
| "What is a HIN?" | 0.694 | 9.6 | 0.033 | 0.999 |
| "Why would a health plan combine data from many sources into one database?" | 0.782 | 28.3 | 0.032 | 0.995 |
| "best pizza in Chicago" (not in the PDF) | 0.457 | 6.1 | 0.032 | 0.000 |

Scores from different methods are on different scales, so compare them only within a column.
Reranking added about 1.3 s per question on the CPU.

### Lessons

- **Hybrid + reranker worked best** on these three questions, so it stays the default. This is a
  first check, not proof; Phase 5 measures it with many more questions.
- **Only the reranker clearly showed that nothing relevant was found** (pizza scored 0.000).
  That could power "this isn't in the documents" in Phase 4, after testing what cut-off works best.
- **The reranker can rescue a good chunk:** "High Volume of Data" was keyword #19 and hybrid #5,
  and the reranker moved it to #2.
- **Hybrid removed keyword search's noise:** off-topic chunks such as Unbundling and XML dropped
  out of the top 5.
- **Test predictions, don't assume them:** we expected vector search to struggle with the acronym
  "HIN", but it did fine, probably because the PDF uses the term so often.
- **Similar results can score alike:** all five "HIN" results scored 0.947–0.999, because all are about HINs.

### Problems & fixes

| Problem | Cause | Fix |
|---|---|---|
| The playground was reachable from other computers on the network | Streamlit listens on all network addresses by default, and the page has no login | Added `--server.address=localhost` to the `ui` command |
| fastembed's default model folder could be emptied | It defaults to the system temp folder | Set a permanent folder: `.cache/fastembed` in the home folder |

### Ideas for later

- Check whether the retrieved "HIN" chunks actually define the term (the previews were cut off).
- Measure the real average chunk length for BM25 (`DGT_BM25_AVG_LEN`, currently fastembed's default of 256).
- Compare a bigger reranker model, such as `BAAI/bge-reranker-base`, in Phase 5.

---

## Phase 4: Generation (answers with citations)

**Goal:** have the chat model read the best chunks and write an answer with page sources.

### What we built

```
question → retriever (hybrid + reranker) → best 5 chunks → chat model → answer with [1] citations
```

- `generation.py`: builds the prompt, gets the answer, and checks its citations.
- `chat.py`: connects to the local chat model in Ollama and streams the answer word by word.
- `ollama_errors.py`: one place for clear messages when Ollama is down or a model is missing.
- `docsgroundtruth ask "question"`: a new command for getting answers in the terminal.
- Playground: a **Get answer** button, with the answer above its source cards.
- 57 tests, using a fake chat model.

### Key concepts

1. **The three rules:** answer only from the sources, cite them, and admit when the answer isn't there.
2. **Prompt:** the instructions (system message) plus the numbered sources and the question
   (user message). The prompt also tells the model to ignore instructions found inside documents.
3. **Two-level refusal:** if the reranker scores every chunk below `min_relevance`, we refuse
   without calling the model. Otherwise the model is told to reply with a fixed sentence.
4. **Citation check:** the code confirms that each cited number matches a source that was sent.
   It can't confirm that the source really says what the sentence claims.
5. **Context size (`num_ctx`):** Ollama silently cuts off anything over the limit, so we set it
   to 8192 tokens, enough for 5 chunks, the instructions and the answer.
6. **Temperature 0:** the model picks the most likely next word, for factual, more repeatable answers.
7. **Streaming:** words appear as they're written, so a slow CPU answer doesn't look frozen.

### Results on our PDF

| Question | Result | Writing time |
|---|---|---|
| "What is a HIN?" | Answer citing p. 315. **Later review found it wrong:** it describes HIE as if it were a HIN (see Phase 5) | 17.4 s |
| "Why would a health plan combine data from many sources into one database?" | Answer citing pp. 311–312; a person compared it with the source and it matched | 19.4 s |
| "best pizza in Chicago" | "The documents don't contain an answer to this question." The model never ran. | 0.0 s |

Finding the chunks added 3–4 s per question, about 2 s of it embedding the question.

### Lessons

- **The model can use too little of what it's given:** it had 5 chunks but cited only one, and
  left out useful points. The answer was correct, but incomplete.
- **It cites less often than asked:** one `[1]` at the end of a paragraph instead of one per claim.
- **A valid citation isn't proof of a correct answer:** a person still had to compare the answer
  with the source. Phase 5 measures this across many questions.
- **PDF parsing can break sentences:** one sentence in the chunk on pp. 311–312 splits across two bullets.
- **Read the library's code:** when streaming, the Ollama library lets `httpx.ConnectError`
  through, unlike normal requests, so we handle both.

### Problems & fixes

| Problem | Cause | Fix |
|---|---|---|
| `uv run` failed with "failed to remove file ... docsgroundtruth.exe" | The playground was still running, and Windows can't replace a running program's file | Stop the playground before running `uv` |
| The playground got stuck on "Stopping..." | Not investigated | Close the terminal; force-stop the process if needed |
| The page said "1 results" | The word was fixed in the text | Choose "result" or "results" based on the count |

### Ideas for later

- Improve answers that use only one source and cite too rarely, then measure the effect.
- Test a question on a healthcare topic the PDF doesn't cover, where the reranker may not score
  near 0 and the model itself has to refuse.

---

## Phase 5: Evaluation

**Goal:** replace "it looks good" with numbers, so we can tell whether a change really helps.

### What we built

- `eval/ahm250_questions.yaml`: a test paper of 30 questions: 20 answerable from the PDF, 5 healthcare
  questions it can't answer, and 5 off-topic. Each expected answer was checked against the PDF's text.
- `evaluation.py` and `evaluation_report.py`: run every question through the real system,
  calculate the scores, and save a review sheet for a person (`eval/results/`, ignored by git).
- `docsgroundtruth evaluate`: the command that runs it all.

### Key concepts

1. **Mean reciprocal rank (MRR):** how high the right page ranks. 1 means always first, 0.5 means always second.
2. **What code can and can't score:** code can check pages, refusals and citations, but not
   whether an answer is true to its source. That still needs a person.
3. **Baseline:** a saved score to compare every later change against.

### Baseline results (first run)

| Check | Score |
|---|---|
| Expected page in the top 5: all four search modes | 20/20 |
| MRR: vector / keyword / hybrid / hybrid + reranker | 0.85 / 0.89 / 0.97 / 0.93 |
| Answered when it should | 19/20 |
| Contains the key words | 19/20 |
| Cites an expected page | 17/20 |
| Sentences with a citation | 19/35 (54%) |
| Refused: healthcare not in the PDF / off-topic | 5/5 / 5/5 |
| Typical time to write an answer | 11.4 s |

Reading the 20 answers against their sources (not by a domain expert): 16 match their sources.
The other 4: q15 (HIN) is wrong, q03 was wrongly refused, and q04 and q17 add small unsupported claims.

### Lessons

- **Key-word checks aren't enough:** the wrong HIN answer passed its key-word check.
- **Three questions weren't enough:** the Phase 3 result that the reranker ranks best didn't hold
  across 20 questions, where hybrid ranked slightly better. The reranker is still useful for
  refusals: refusal questions scored 0.074 or below, and real questions 0.576 or above.
- **A test paper written from the pages it tests is easy:** the 100% search score flatters the system.
- **Some chunks mix unrelated topics:** the COBRA chunk also contains the HMO Act and HIPAA.

---

## Next phases

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
