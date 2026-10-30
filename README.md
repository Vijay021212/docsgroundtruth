# DocsGroundTruth

A RAG system that answers questions from your own documents, with page citations. It runs entirely on your own computer, with free local models and no GPU, and it's tested with evaluation scores, including its failures.

```
$ uv run docsgroundtruth ask "How long do I have to return from Medicare Advantage and still buy Medigap?"

You have a "trial right" under federal law to buy a Medigap policy if you return to
Original Medicare within 12 months of joining a Medicare Advantage Plan [1].

Sources:
  [1] 12026-understanding-medicare-advantage-plans.pdf, p. 7
```

*An illustrative example, based on the answer to a similar question in the public test (p08).*

> **Not for medical, legal or insurance decisions.** It can give wrong answers on hard questions, even with a citation. Always check the cited page.

## How it works

```
INGEST   PDF ─► Docling (layout, tables, OCR) ─► chunks ─┬─► dense vectors (meaning)
                                                         └─► BM25 sparse vectors (keywords)
                                                                     │
                                                              Qdrant (embedded)
                                                                     │
ASK      question ─► vector search ─┐                                │
                                    ├─► fusion (RRF) ─► reranker ─► best 5 chunks
                 ─► keyword search ─┘                                │
                                                                     ▼
                                     local chat model ─► answer with [1] citations
                                                                     │
                                                       code checks each citation
```

| Layer | Tool | Why |
|---|---|---|
| Parsing and chunking | [Docling](https://docling-project.github.io/docling/) | Understands layout, so chunks follow the document's sections |
| Embeddings and chat | [Ollama](https://ollama.com): `nomic-embed-text`, `qwen3:4b-instruct` | Free and private, and runs on a CPU |
| Vector and keyword search | [Qdrant](https://qdrant.tech) local mode, BM25 through [fastembed](https://github.com/qdrant/fastembed) | One database with no server to run |
| Reranking | `ms-marco-MiniLM-L-6-v2` (ONNX, CPU) | Ranks hard questions much better, and makes many off-topic questions easy to refuse |
| Interfaces | CLI, [Streamlit](https://streamlit.io) playground, [FastAPI](https://fastapi.tiangolo.com) | All three call the same core code |

**Design choices**
- **Answers are tied to their sources.** The model is told to use only the retrieved chunks, and the code checks every citation against what was actually sent.
- **It can refuse.** A very low reranker score skips the model entirely, and otherwise the model is told to say the answer isn't there. A higher score cut-off wouldn't work: in testing, real questions scored as low as 0.03, while an unanswerable one scored 0.93.
- **Ingestion is safe to repeat.** Chunk IDs are deterministic, and a file's old chunks are replaced.
- **Components are replaceable.** The pipeline and retriever depend on small interfaces (`domain.py`), and all the wiring lives in `factory.py`. That also lets the tests run in seconds with fakes.

## Evaluation

Each test paper states what should happen for each question, and `docsgroundtruth evaluate` runs them through the real system. There are three papers:

| Check | Public booklet (32 pages) | Course PDF: standard (472 pages) | Course PDF: hard (472 pages) |
|---|---|---|---|
| Right page in the top 5 | 12/12 | 20/20 | 11/12 |
| Ranking (MRR, 1.00 = always first) | 0.96 | 0.95 | 0.83 |
| Answered when it should | 12/12 | 19/20 | 11/12 |
| Contains the expected key words | 12/12 | 19/20 | 10/12 |
| Cites an expected page | 11/12 | 19/20 | 10/12 |
| Sentences with a citation | 67% | 70% | 89% |
| Refused when it should | 5/5 | 10/10 | 5/5 |
| Typical time to write an answer (laptop CPU) | 9.9 s | 9.2 s | 9.8 s |

On the course PDF's hard paper, the reranker clearly earned its place, with an MRR of 0.83 against about 0.60 for each other search mode. On the other two papers, the search modes were close.

### What didn't work

Key words can't prove an answer is true, so the answers were also read against their sources. That review wasn't done by an independent domain expert.

- **Wrong answer:** "What is a HIN?" describes health information exchange as if it were a health information network. It still passes the key-word check.
- **Wrong answer:** "Can someone with a long-term illness use a specialist as their main doctor?" gets "No", though the PDF says some plans allow it. Parsing filed that text under the wrong heading, and its words differ from the question's, so search misses it.
- **Wrong refusals:** two questions were refused even though the sources help.
- **Weaker details:** a few answers on the hard paper are muddled or add a nearby detail.
- **Citations:** only 67–70% of sentences have one on the two easier papers, and citations sometimes point to the wrong chunk.

The test papers are small (47 questions on the course PDF and 17 on the booklet), and they were written after reading the documents, so they probably flatter the system. The [learning log](LEARNING_LOG.md) has the full history, including a change that raised the scores but made the answers worse.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com). The commands below use Windows PowerShell.

```powershell
git clone https://github.com/Vijay021212/docsgroundtruth.git
cd docsgroundtruth
uv sync
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct-2507-q4_K_M

# Put your PDFs in data\raw, then:
uv run docsgroundtruth ingest                        # read, chunk and index them
uv run docsgroundtruth ask "your question"           # get an answer with sources
uv run docsgroundtruth ui                            # testing playground in your browser
uv run docsgroundtruth api                           # web API, docs at http://127.0.0.1:8000/docs
```

Other commands: `search` (see the chunks, with `-m vector|keyword|hybrid|rerank`), `stats` (a report on the chunks) and `evaluate` (run a test paper). All settings are optional and listed in [.env.example](.env.example).

The playground and API have no login, so both only accept connections from your own computer. Local storage also allows only one program at a time on the index, so stop the playground before you ingest.

### Reproduce the public scores

```powershell
New-Item -ItemType Directory -Force data\public\raw
Invoke-WebRequest https://www.medicare.gov/publications/12026-understanding-medicare-advantage-plans.pdf -OutFile data\public\raw\12026-understanding-medicare-advantage-plans.pdf

$env:DGT_QDRANT_PATH = "data/public/index"
$env:DGT_PARSED_DOCS_DIR = "data/public/processed"
uv run docsgroundtruth ingest data\public\raw
uv run docsgroundtruth evaluate eval\public_questions.yaml
```

Scores used the April 2026 edition of the booklet ([its checksum](eval/public_questions.yaml) is in the test paper). A new edition may change them, and timings depend on your computer. The course PDF isn't in this repository, so its scores can't be reproduced from here.

## Development

```powershell
uv run ruff check .      # code checker
uv run pytest            # tests; no models or Ollama needed
```

GitHub Actions runs both on every push.

```
src/docsgroundtruth/
  domain.py                types and interfaces; depends on nothing else
  chunking.py, bm25.py, embedding.py, reranking.py, chat.py     one wrapper per tool
  vector_store.py          the only module that talks to Qdrant
  pipeline.py              ingestion: chunk → encode → store
  retrieval.py             the four search modes and RRF
  generation.py            prompt, answer and citation checks
  evaluation.py            test papers and scores
  factory.py               wires everything together
  cli.py, api.py, ui/      the three interfaces
eval/                      the test papers
tests/                     fast tests with fakes
```

## Limitations and next steps

- **Single program, small scale:** Qdrant's local mode searches every chunk and allows one connection, which suits thousands of chunks. The next step for real use is a Qdrant server.
- **No login:** add authentication before exposing the API.
- **Wrong headings from parsing** can hide text from search.
- **Whole-document questions** like "summarize this" work poorly, because search looks for specific passages.
- **Ideas to test next:** a bigger chat model, other chunk sizes, and larger test papers reviewed by a domain expert.

## License

The code is under the [MIT License](LICENSE). Documents you ingest aren't part of this repository.
