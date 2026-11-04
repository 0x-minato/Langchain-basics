# LangSmith

This folder is **not a new RAG or agent**. It is **observability** for the pipelines you already built.

```text
LangChain   = bricks (ChatCohere, retriever, AgentExecutor, LCEL |)
LangGraph   = stateful graphs (nodes, edges, thread_id)
LangSmith   = the dashboard that records every invoke as a tree of spans
```

`ragv1.py` is the PDF RAG from `rag/notes.md` with tracing **on**.  
`ragv2.py` is the same RAG with **named** `@traceable` steps.  
`agent_rag.py` is the ReAct agent from `agents/agent.py` with a named run + tags.

You need RAG + agents first. Come here after `yt-chatbot/chatbot.py` and `agents/agent.py`.

Dashboard for this org: **[https://apac.smith.langchain.com](https://apac.smith.langchain.com)** (not the US site). Run from the **repo root** with `venv312`.

---

## How to restudy this folder

| Order | File | What it proves |
| --- | --- | --- |
| 1 | this `notes.md` | vocabulary: trace, span, tags, p50 |
| 2 | `ragv1.py` | **auto-trace**: env vars + `chain.invoke` is enough |
| 3 | `ragv2.py` | **manual spans**: `@traceable`, nesting, tags, metadata |
| 4 | `agent_rag.py` | AgentExecutor tree: Thought → tool → Observation → Final Answer |

After each run, open the matching **project** in LangSmith and expand the run. If you only see one box, you are on the wrong run or a child was skipped.

| File | `LANGCHAIN_PROJECT` | Run name to click |
| --- | --- | --- |
| `ragv1.py` | `RAG Chatbot` | the LCEL chain |
| `ragv2.py` | `pdf_rag_demo` | **`pdf_rag_query`** (parent) |
| `agent_rag.py` | `agent_rag` | **`agent_rag_query`** |

---

## What LangSmith is

**LangSmith** is LangChain’s hosted **tracing, monitoring, and eval** product.

Every `invoke` can become a **trace**: a tree of timed steps (LLM call, retriever, tool, your Python function). You inspect inputs/outputs, latency, tokens, and errors **after** the fact — like browser DevTools for the backend.

It does **not** retrieve documents, call Cohere, or run your graph. If tracing is off, the app still works. If tracing is on and LangSmith 403s, the app **still works**; you only lose the dashboard (see the 403 we hit on `sequencial-chain.py`).

Three jobs, one product:

| Job | What you look at |
| --- | --- |
| **Debug** | this one question: which child failed? |
| **Monitor** | many runs: p50 / p95 latency, error rate |
| **Eval** | a dataset of questions + a scorer (you have not coded this yet) |

**One line:** LangSmith records *what happened inside* an `invoke`. LangChain/LangGraph *are* the invoke.

---

## Trace vocabulary

```text
TRACE  = one user question (root run)
  SPAN = one child step (retriever, LLM, tool, @traceable function)
    SPAN = nested child (e.g. load_pdf inside setup_pipeline)
```

| Word | Meaning |
| --- | --- |
| **Run / span** | one timed box in the tree |
| **Trace** | the whole tree for one request |
| **Project** | folder of traces (`LANGCHAIN_PROJECT`) |
| **Tags** | short labels you **filter** on (`rag`, `setup`, `weather`) |
| **Metadata** | extra JSON on the run (`k: 4`, model id, city) |
| **Run name** | title in the UI (`pdf_rag_query`) |

LCEL already emits spans (`ChatCohere`, `RunnableParallel`, parser). `@traceable` adds spans for **your** functions (`load_pdf`, `split_documents`), which LangChain would otherwise treat as invisible Python.

---

## How tracing turns on

Nothing in the script “calls LangSmith” for the default path. The **SDK reads env vars**.

In `.env` (gitignored):

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://apac.api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=langchain-demo
```

Newer names (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT`) are aliases. This repo still uses `LANGCHAIN_*`.

`load_dotenv()` must run **before** the first `invoke`. Scripts also override the project:

```python
os.environ["LANGCHAIN_PROJECT"] = "pdf_rag_demo"
```

Do that **after** `load_dotenv()` if you want the script to win; `load_dotenv()` does **not** override vars already in the process. `ragv1.py` sets the project *before* `load_dotenv()`, so a project already in `.env` can win. `ragv2.py` / `agent_rag.py` set it *after*, so the script wins.

### Region (this is why ingest 403’d)

There is **no auto-detect**. A key from an APAC org is rejected by the US API with a generic `403 Forbidden` on `/runs/multipart`.

| Region | UI | API |
| --- | --- | --- |
| GCP US | `https://smith.langchain.com` | `https://api.smith.langchain.com` |
| GCP APAC | `https://apac.smith.langchain.com` | `https://apac.api.smith.langchain.com` |
| GCP EU | `https://eu.smith.langchain.com` | `https://eu.api.smith.langchain.com` |

This workspace is **APAC**. `.env` must use the APAC endpoint. The chain still ran when it was wrong; only traces failed.

---

## Two ways to get spans

### 1. Auto (LangChain / LangGraph)

If tracing env is set, `chain.invoke`, `model.invoke`, `AgentExecutor.invoke`, retrievers, and tools are recorded **with no extra decorator**.

`ragv1.py` is this. You only see whatever LangChain wrapped. `load()` / `split_documents` / `FAISS.from_documents` are plain Python — **not** in the tree unless you decorate them.

You can still name the LCEL run:

```python
chain.invoke(q, config={"run_name": "pdf_rag_query", "tags": ["rag"]})
```

### 2. Manual (`@traceable`)

```python
from langsmith import traceable

@traceable(name="load_pdf", tags=["setup", "loader"], metadata={"file": "islr.pdf"})
def load_pdf(path: str):
    ...
```

`ragv2.py` is this. Nested `@traceable` calls become **children** of the parent if they run **inside** that parent’s stack.

`@traceable` needs tracing enabled (`LANGCHAIN_TRACING_V2=true`). It does not replace auto-trace; it **adds** named boxes around your functions. The Cohere call inside `chain.invoke` still appears as an LLM span.

---

## Nesting — the bug you actually hit

Child spans appear **only if that function ran inside the parent you clicked**.

```text
WRONG (two traces)
  setup_pipeline()     ← run A, before input()
  chain.invoke(...)    ← run B, name pdf_rag_query
  You open B → no load_pdf / split / build

RIGHT (one tree)
  @traceable pdf_rag_query
      setup_pipeline
          load_pdf
          split_documents
          load_vectorstore  OR  build_vectorstore
      chain.invoke  (retriever, prompt, ChatCohere)
```

`ragv2.py` calls `setup_pipeline` **inside** `run_query` so one click shows setup + answer.

Second trap: **short-circuit**. If you `return load_vectorstore()` before `split_documents()`, those spans never exist. `setup_pipeline` always loads + splits; it only skips **embedding** when `faiss_islr/` already exists.

```text
first run:   load_pdf → split_documents → build_vectorstore
later runs:  load_pdf → split_documents → load_vectorstore
```

`load_vectorstore` and `build_vectorstore` are alternatives, not siblings on the same run.

Third trap: returning a **FAISS object** as the span output. LangSmith tries to JSON-serialize I/O. Huge `Document` lists and FAISS indexes bloat or drop payloads. `process_inputs` / `process_outputs` keep a small summary:

```python
def _n_docs(docs):
    return {"n_docs": len(docs)}

def _vs_out(vs):
    return {"n_vectors": vs.index.ntotal}

@traceable(name="load_pdf", process_outputs=_n_docs)
```

---

## Tags vs metadata

Both hang on a run. They are not the same.

| | Tags | Metadata |
| --- | --- | --- |
| Shape | list of strings | dict |
| UI | chips; **filter** the project | JSON panel |
| Use | `rag` / `setup` / `query` / `agent` | `k`, model id, file name, city |

```python
@traceable(
    name="pdf_rag_query",
    tags=["rag", "query"],
    metadata={"k": 4, "llm": "command-a-03-2025"},
)
```

`AgentExecutor.invoke(..., config={"tags": [...], "metadata": {...}, "run_name": "..."})` attaches the same things to the **auto** root run (`agent_rag.py`).

Filter in the UI: tag `setup` → only index steps; tag `query` → only questions.

---

## p50 and p95

On the project’s monitoring view you will see **p50** and **p95**. They are **percentiles** of a pile of run durations (or tokens, or scores). Not magic LangSmith numbers.

Sort every run’s latency.

| Name | Meaning |
| --- | --- |
| **p50** (median) | half the runs were faster, half slower. Typical request. |
| **p95** | 95% were faster. Only 5% slower. Tail users feel. |
| **p99** | rarer tail (timeouts, 429 retries, extra agent loops) |

Average hides a 30s Cohere retry. p50 stays ~2s; p95 jumps. SLAs are usually on **p95**, not the mean.

Your first `ragv2` question is slow because it **re-parses the whole ISLR PDF** for the `load_pdf` span. That is p95 material, not “embeddings printing.”

---

## `ragv1.py` — auto-traced RAG

Same pipeline as `rag/notes.md` / `yt-chatbot/chatbot.py`, corpus = `islr.pdf` (ISLR textbook).

```text
PDF → split → MiniLM embed → FAISS → retrieve k=4 → prompt → ChatCohere
```

What LangSmith shows (auto):

```text
chain
  RunnableParallel
    retriever + format_docs
    passthrough question
  ChatPromptTemplate
  ChatCohere
  StrOutputParser
```

What it does **not** show: `PyPDFLoader.load`, splitter, `FAISS.from_documents`. Those happen **before** `chain.invoke` as ordinary Python.

Practical bits already in the file:

| Problem | Fix in this file |
| --- | --- |
| `islr.pdf` not found from repo root | `PDF_PATH = join(dirname(__file__), "islr.pdf")` |
| Cohere trial 429 on a 600-page book | local `HuggingFaceEmbeddings` MiniLM |
| Re-embed every run | `vs.save_local` / `load_local` in `faiss_islr/` |
| Font `/Widths` spam looking like vectors | `warnings.filterwarnings` for fontTools |
| Need `pypdf` | extra package; `langchain-community` does not ship it |

`allow_dangerous_deserialization=True` on `FAISS.load_local` is required for the pickle FAISS uses. Only load **your** index.

Chat is Cohere `command-a-03-2025`. Embeddings are **not** Cohere (trial token/min and 100 calls/min).

---

## `ragv2.py` — named setup + one parent query

Same RAG. Extra: `@traceable` around each setup step, plus `run_query` as the root.

Tree you should expand:

```text
pdf_rag_query          tags: rag, query
  setup_pipeline       tags: setup, rag
    load_pdf           tags: setup, loader, pdf
    split_documents    tags: setup, splitter
    load_vectorstore   tags: setup, index, cache     (if faiss_islr exists)
    build_vectorstore  tags: setup, index, embed     (first index only)
  RunnableParallel / ChatCohere / parser
```

`run_query` builds the retriever **after** setup so the chain is inside the same parent.

Cost of nested traces: **every question re-loads and re-splits the PDF** so those children exist. Embedding is still skipped when the index is on disk. First answer after `Q:` can sit on fontTools warnings for a while — that is `load_pdf`, not MiniLM dumping vectors.

Projects: `pdf_rag_demo`. Do not look at `langchain-demo` leftover traces from other scripts.

---

## `agent_rag.py` — traced ReAct agent

Same idea as `agents/agent.py`: `create_react_agent` + `AgentExecutor` + search + weather.

Differences that matter for LangSmith:

- `LANGCHAIN_PROJECT = "agent_rag"`
- `invoke(..., config={run_name, tags, metadata})` so the root is easy to find
- OpenAI / `hub.pull("hwchase17/react")` are **not** used (OpenAI keys failed here; Hub public pull is blocked). Local ReAct `PromptTemplate` + `ChatCohere`.

Auto-trace of `AgentExecutor` is the interesting tree:

```text
agent_rag_query
  LLM (Thought + Action)
  get_weather_data          ← tool span
  LLM
  duckduckgo_search         ← fallback when weather API 615’d
  LLM
  Final Answer
```

That is why agents belong in LangSmith: you see **which tool ran**, the observation, and why it looped. Verbose stdout (`> Entering new agent_rag_query chain...`) is the same story as the UI.

`handle_parsing_errors=True` keeps a bad ReAct parse from killing the trace mid-loop. `max_iterations=5` caps runaway tools (and p95).

Weatherstack may fail (`success: False`). The agent is allowed to search instead. That is a **good** trace to keep: failed tool → recovery.

---

## What the terminal spam actually is

| You see | It is | Not |
| --- | --- | --- |
| `[500, 500, 513, 525, ...]` + `fontTools is required` | PDF font **glyph widths** (LaTeX CMR/CMTT) | embeddings |
| `Retrying ... TooManyRequestsError` 429 | Cohere **trial** rate limit | LangSmith |
| `Failed to POST .../runs/multipart` 403 | **wrong LangSmith region** or bad key | the LLM call itself |
| long pause after `Q:` in `ragv2.py` | `load_pdf()` walking ~600 pages | hung Python |

`pypdf` warns because ISLR uses Type1 fonts and `fontTools` is optional. Install it only if encoding looks wrong; filtering the warning is enough for this demo.

---

## LangSmith vs print vs verbose

| | `print` / `verbose=True` | LangSmith |
| --- | --- | --- |
| Lives | this terminal, then gone | dashboard, searchable |
| Structure | text log | tree with timings |
| Compare two prompts | scrollback | two traces side by side |
| Production | noisy | tags + p95 |

You still want `verbose=True` while learning ReAct. LangSmith is what you keep when the terminal is gone.

LangGraph `thread_id` is **not** a LangSmith concept. `thread_id` is conversation memory in a checkpointer (`persistence/notes.md`). LangSmith’s “thread” in some UIs groups traces; do not mix them in an interview.

---

## Evals (you have not built these yet)

Beyond traces, LangSmith can store a **dataset** of `{input, expected}` and run your chain over it with a **scorer** (exact match, LLM-as-judge, embedding distance). That is how you prove a prompt change helped.

This folder only does **tracing**. If an interview asks “how do you know RAG got better?”, the answer is: hold-out questions + evals, not staring at one lucky trace.

---

## Interviews

LangSmith is **rarely** a named round. They ask how you **debug** a bad RAG/agent answer. Tracing is the tool, not the syllabus.

Worth saying:

> Each invoke is a trace. Children are retrieve / tools / LLM. Tags filter. p50 is typical latency; p95 is the tail. 403 on ingest is usually the wrong regional endpoint.

Not worth a week of study. RAG failure modes and agents weigh more. See also the interview note in the chat: know the vocabulary, do not memorize the UI.

---

## Run commands

From repo root, `venv312` active:

```powershell
python .\langgraph\langsmith\ragv1.py
python .\langgraph\langsmith\ragv2.py
python .\langgraph\langsmith\agent_rag.py
```

Needs: `COHERE_API_KEY`, LangSmith key + **APAC** endpoint, `pypdf`, `faiss-cpu`, MiniLM weights (downloaded once). `islr.pdf` sits next to the scripts.

UI: [https://apac.smith.langchain.com](https://apac.smith.langchain.com) → project `pdf_rag_demo` or `agent_rag` → expand the named run.

---

## If traces are missing

| Symptom | Cause |
| --- | --- |
| App works, 403 ingest | `LANGCHAIN_ENDPOINT` is US; use APAC |
| No runs at all | `LANGCHAIN_TRACING_V2` not true, or `load_dotenv()` after first invoke, or wrong project in the UI |
| Only one span | you clicked `pdf_rag_query` while setup ran as a **separate** earlier run, or cache skipped `split`/`build` |
| `load_pdf` huge / slow | expected: ISLR re-parse for the span |
| Empty output on a FAISS span | serialize the count via `process_outputs`, not the index |
| OpenAI / Hub errors | this folder uses Cohere + a **local** ReAct prompt |

---

## Map back to the rest of the repo

| Idea | Already learned in | Observed here |
| --- | --- | --- |
| RAG chain | `rag/notes.md`, `yt-chatbot/` | `ragv1.py`, `ragv2.py` |
| FAISS + MiniLM | `vector_store/`, `embeddedModels/` | index cache `faiss_islr/` |
| ReAct agent | `agents/agent.py` | `agent_rag.py` |
| LCEL parallel | `runnables/parallel.py` | retriever ∥ question |
| Wrong relative path | `doc_loaders/` “run from root” | `__file__` so `islr.pdf` resolves |
| Persistence of **conversation** | `langgraph/persistence/` | not this folder — this persists the **vector index** only |
