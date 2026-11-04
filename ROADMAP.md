# Study roadmap

This repo grew in **topic order**, not random files. Walk it **top to bottom**. Each file adds one idea. Do not jump to LangGraph until the LangChain spine is clear.

How to use this:

1. Open the **code file** (or notebook). Run it from the **repo root** with `venv312`.
2. Then read that folder’s `notes.md` if you forget *why* it exists.
3. Only then go to the next numbered file.

Folders that are **notes only** (`text_splitting/`, `retrievers/`, `rag/`, some LangGraph theory) have no script — read the notes, then continue. Instrumented RAG/agent scripts live in `langgraph/langsmith/`.

```text
Python types → call a model → chat + parsers → LCEL chains
    → load → split → embed → store → retrieve → RAG app
    → tools → agents → LangSmith traces
    → LangGraph graphs → persistence → Streamlit chatbot
```

---

## Fast path (if you only re-read)

If you already ran most of this once, this is the **spine**. Skip the rest unless a concept is fuzzy.

| # | File | Idea |
| --- | --- | --- |
| 1 | `LLMs/llm_demo.py` | `invoke` |
| 2 | `chatModels/messages.py` | system / human / ai |
| 3 | `chatModels/stoutparser1.py` | `prompt \| model \| parser` |
| 4 | `masti-with-chains/sequencial-chain.py` | two LLM calls in a row |
| 5 | `runnables/branch.py` | routing |
| 6 | `doc_loaders/text_loader.py` | `Document` |
| 7 | `embeddedModels/embedding_model.py` | vectors by hand |
| 8 | `vector_store/chroma_db.py` | persist + search |
| 9 | `yt-chatbot/chatbot.py` | full RAG |
| 10 | `tools-calling/currency_converter.py` | tool loop |
| 11 | `agents/agent.py` | ReAct agent |
| 12 | `langgraph/langsmith/ragv2.py` | `@traceable` tree in LangSmith |
| 13 | `langgraph/workflows/iterative.ipynb` | graph cycle |
| 14 | `langgraph/persistence/chatbot.ipynb` | `thread_id` memory |
| 15 | `langgraph/chatbot/frontend.py` | Streamlit UI |

---

## Part 0 — Python the parsers need

Not LangChain. TypedDict vs Pydantic, because later files ask models for JSON.

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 0.1 | [`py-basics/type-dict.py`](py-basics/type-dict.py) | Named dict keys, **no** runtime check | 0.2 |
| 0.2 | [`py-basics/pydantic-demo.py`](py-basics/pydantic-demo.py) | `BaseModel` **does** validate at runtime | Part 1 |

Companion: [`py-basics/notes.md`](py-basics/notes.md)

---

## Part 1 — Call a model

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 1.1 | [`LLMs/llm_demo.py`](LLMs/llm_demo.py) | `load_dotenv` → chat model → `invoke` → `AIMessage` | Part 2 |

Print `result.content` when you only want the text. Companion: [`LLMs/notes.md`](LLMs/notes.md)

---

## Part 2 — Chat models, messages, parsers

Largest folder. Stay in this order; parsers only make sense after messages.

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 2.1 | [`chatModels/chat_model.py`](chatModels/chat_model.py) | OpenAI wrapper, `temperature`, `.content` | 2.2 |
| 2.2 | [`chatModels/gemini_model.py`](chatModels/gemini_model.py) | Same pattern, Gemini (import may be stale — working one is in `LLMs/llm_demo.py`) | 2.3 |
| 2.3 | [`chatModels/anthropic_model.py`](chatModels/anthropic_model.py) | Same pattern; model id is a placeholder | 2.4 |
| 2.4 | [`chatModels/huggingface_model.py`](chatModels/huggingface_model.py) | Local Qwen via LangChain | 2.5 |
| 2.5 | [`chatModels/quen_model.py`](chatModels/quen_model.py) | Same model **without** LangChain (`generate`) | 2.6 |
| 2.6 | [`chatModels/messages.py`](chatModels/messages.py) | `SystemMessage` / `HumanMessage` / append `AIMessage` | 2.7 |
| 2.7 | [`chatModels/chatbot.py`](chatModels/chatbot.py) | Real chat loop + sliding window history | 2.8 |
| 2.8 | [`chatModels/stoutparse.py`](chatModels/stoutparse.py) | Two `invoke`s by hand (report → summary) | 2.9 |
| 2.9 | [`chatModels/stoutparser1.py`](chatModels/stoutparser1.py) | Same job as **LCEL pipe** | 2.10 |
| 2.10 | [`chatModels/jsonoutputparser.py`](chatModels/jsonoutputparser.py) | JSON string → dict | 2.11 |
| 2.11 | [`chatModels/structuredoutputparser.py`](chatModels/structuredoutputparser.py) | Named fields via `ResponseSchema` | 2.12 |
| 2.12 | [`chatModels/pydanticoutputparser.py`](chatModels/pydanticoutputparser.py) | JSON → Pydantic object | 2.13 |
| 2.13 | [`chatModels/structured_bot.py`](chatModels/structured_bot.py) | `with_structured_output` (model emits the schema) | Part 3 |

Skip until you care: [`chatModels/chat_templete.py`](chatModels/chat_templete.py) (unfinished sketch). Working prompt templates are in `chatbot.py`.

Companion: [`chatModels/notes.md`](chatModels/notes.md)

---

## Part 3 — LCEL chains (`|`)

First time you **pipe** prompt → model → parser. `|` is `RunnableSequence`.

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 3.1 | [`masti-with-chains/simple-chain.py`](masti-with-chains/simple-chain.py) | One pipe, `StrOutputParser`, `print_ascii` | 3.2 |
| 3.2 | [`masti-with-chains/sequencial-chain.py`](masti-with-chains/sequencial-chain.py) | Two model calls: report → summary | 3.3 |
| 3.3 | [`masti-with-chains/parallel-chain.py`](masti-with-chains/parallel-chain.py) | Dict of runnables = fan-out | 3.4 |
| 3.4 | [`masti-with-chains/conditional-chain.py`](masti-with-chains/conditional-chain.py) | Branch on classifier output | Part 4 |

Companion: [`masti-with-chains/notes.md`](masti-with-chains/notes.md)

---

## Part 4 — Same graphs, class names spelled out

Same shapes as Part 3, but `RunnableSequence(...)` instead of `|`. Read this if `|` still feels like magic.

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 4.1 | [`runnables/sequence.py`](runnables/sequence.py) | Sequential, class form | 4.2 |
| 4.2 | [`runnables/parallel.py`](runnables/parallel.py) | `RunnableParallel` | 4.3 |
| 4.3 | [`runnables/passthrough.py`](runnables/passthrough.py) | Keep original input while a branch runs | 4.4 |
| 4.4 | [`runnables/lambda.py`](runnables/lambda.py) | Custom Python in the pipe | 4.5 |
| 4.5 | [`runnables/branch.py`](runnables/branch.py) | `RunnableBranch` routing | Part 5 |

Companion: [`runnables/notes.md`](runnables/notes.md)

---

## Part 5 — RAG pipeline (one stage at a time)

This is one product split across folders:

```text
loader → splitter → embed → vector store → retriever → LLM
```

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 5.1 | [`doc_loaders/text_loader.py`](doc_loaders/text_loader.py) | File → `Document` (`page_content` + metadata) | 5.2 |
| 5.2 | [`doc_loaders/py_pdf_loader.py`](doc_loaders/py_pdf_loader.py) | PDF pages | 5.3 |
| 5.3 | [`doc_loaders/dir_loader.py`](doc_loaders/dir_loader.py) | Whole folder | 5.4 |
| 5.4 | [`doc_loaders/web_base_loader.py`](doc_loaders/web_base_loader.py) | URL → documents | 5.5 |
| 5.5 | [`text_splitting/notes.md`](text_splitting/notes.md) | Why chunk (`chunk_size` / overlap). **No script yet.** | 5.6 |
| 5.6 | [`embeddedModels/embedding_model.py`](embeddedModels/embedding_model.py) | `embed_documents` vs `embed_query`, cosine search by hand | 5.7 |
| 5.7 | [`vector_store/chroma_db.py`](vector_store/chroma_db.py) | Persist vectors, `similarity_search` | 5.8 |
| 5.8 | [`retrievers/notes.md`](retrievers/notes.md) | Retriever = `invoke(query) → list[Document]`. **No script yet.** | 5.9 |
| 5.9 | [`rag/notes.md`](rag/notes.md) | Name the whole pattern: retrieve → stuff prompt → generate | 5.10 |
| 5.10 | [`yt-chatbot/chatbot.py`](yt-chatbot/chatbot.py) | **All stages in one file** (YouTube captions → FAISS → Cohere) | Part 6 |

PDF RAG with tracing is later: [`langgraph/langsmith/ragv1.py`](langgraph/langsmith/ragv1.py). Companion idea: [`rag/notes.md`](rag/notes.md).

Companions: [`doc_loaders/notes.md`](doc_loaders/notes.md), [`embeddedModels/notes.md`](embeddedModels/notes.md), [`vector_store/notes.md`](vector_store/notes.md), [`yt-chatbot/notes.md`](yt-chatbot/notes.md)

---

## Part 6 — Tools, then agents

RAG **shows** the model extra text. Tools let it **do** something. An agent is that loop with a stop rule.

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 6.1 | [`tools-calling/tools.py`](tools-calling/tools.py) | `@tool` / schemas — define only, no loop | 6.2 |
| 6.2 | [`tools-calling/calling.py`](tools-calling/calling.py) | `bind_tools` → `tool_calls` → run Python → second invoke | 6.3 |
| 6.3 | [`tools-calling/currency_converter.py`](tools-calling/currency_converter.py) | Loop until the model stops calling tools | 6.4 |
| 6.4 | [`agents/agent.py`](agents/agent.py) | Same loop as a **ReAct** `AgentExecutor` | Part 7 |

Companions: [`tools-calling/notes.md`](tools-calling/notes.md), [`agents/notes.md`](agents/notes.md)

`currency_converter.py` is already an agent you wrote by hand. `agent.py` is the library’s named loop.

---

## Part 6b — LangSmith (see what an `invoke` actually did)

Not a new chain. **Traces** of the RAG and agent you already have. Do this **after** Parts 5–6 so the tree means something.

Dashboard: [https://apac.smith.langchain.com](https://apac.smith.langchain.com) (this org is APAC, not US).

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 6.5 | [`langgraph/langsmith/notes.md`](langgraph/langsmith/notes.md) | Trace vs span, tags vs metadata, p50/p95, APAC endpoint | 6.6 |
| 6.6 | [`langgraph/langsmith/ragv1.py`](langgraph/langsmith/ragv1.py) | Auto-trace: env on + `chain.invoke` (ISLR PDF RAG) | 6.7 |
| 6.7 | [`langgraph/langsmith/ragv2.py`](langgraph/langsmith/ragv2.py) | `@traceable` nested setup under **one** `pdf_rag_query` | 6.8 |
| 6.8 | [`langgraph/langsmith/agent_rag.py`](langgraph/langsmith/agent_rag.py) | `AgentExecutor` tree: Thought → tool → Final Answer | Part 7 |

Open the **named** run (`pdf_rag_query` / `agent_rag_query`) and **expand** it. One box usually means you clicked the LCEL child, not the parent.

Companion: [`langgraph/langsmith/notes.md`](langgraph/langsmith/notes.md)

---

## Part 7 — LangGraph (graphs, then memory, then UI)

LangChain is still the bricks (Cohere, prompts, tools). LangGraph is **state + edges + loops**.

### 7a — Why a graph (notes only)

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 7.1 | [`langgraph/langchain_vs_langgraph/notes.md`](langgraph/langchain_vs_langgraph/notes.md) | LangChain composes calls; LangGraph orchestrates **stateful** flow | 7.2 |
| 7.2 | [`langgraph/core_concepts/notes.md`](langgraph/core_concepts/notes.md) | Node, edge, state, reducer, Pregel | 7.3 |

### 7b — Four graph shapes (code)

Same four lines every time: **State** → **nodes** → **edges** → `compile().invoke(...)`.

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 7.3 | [`langgraph/workflows/sequencial.ipynb`](langgraph/workflows/sequencial.ipynb) | Graph with **no LLM** (BMI line) | 7.4 |
| 7.4 | [`langgraph/workflows/basic_seq.ipynb`](langgraph/workflows/basic_seq.ipynb) | One LLM node | 7.5 |
| 7.5 | [`langgraph/workflows/blog.py`](langgraph/workflows/blog.py) | Sequential LLM chain on shared state | 7.6 |
| 7.6 | [`langgraph/workflows/parallel.py`](langgraph/workflows/parallel.py) | Fan-out / fan-in, partial updates | 7.7 |
| 7.7 | [`langgraph/workflows/essay_workflow.ipynb`](langgraph/workflows/essay_workflow.ipynb) | Parallel judges + `operator.add` | 7.8 |
| 7.8 | [`langgraph/workflows/conditional.ipynb`](langgraph/workflows/conditional.ipynb) | `add_conditional_edges` (routing) | 7.9 |
| 7.9 | [`langgraph/workflows/iterative.ipynb`](langgraph/workflows/iterative.ipynb) | **Cycle** (generate ⇄ evaluate) until stop | 7.10 |

Companion: [`langgraph/workflows/notes.md`](langgraph/workflows/notes.md)

After each notebook, look at the mermaid / PNG. If the picture disagrees with the edges you think you wrote, trust the picture.

### 7c — Persistence (same graphs, saved)

Not a new shape. After each node, LangGraph writes a **checkpoint**.

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 7.10 | [`langgraph/persistence/chatbot.ipynb`](langgraph/persistence/chatbot.ipynb) | `thread_id` = conversation; time travel via `checkpoint_id` | 7.11 |
| 7.11 | [`langgraph/persistence/fault_tolerance.ipynb`](langgraph/persistence/fault_tolerance.ipynb) | Crash mid-graph, `get_state`, resume `invoke(None, config)` | 7.12 |

Companion: [`langgraph/persistence/notes.md`](langgraph/persistence/notes.md)

### 7d — Product: Streamlit chatbot

Same chat graph as the persistence notebook, plus SQLite + a UI.

| Order | File | What to take away | Then |
| --- | --- | --- | --- |
| 7.12 | [`langgraph/chatbot/chatbot.py`](langgraph/chatbot/chatbot.py) | Graph + `SqliteSaver` + `thread_id` | 7.13 |
| 7.13 | [`langgraph/chatbot/frontend.py`](langgraph/chatbot/frontend.py) | Streamlit sidebar threads, `stream` tokens | done |

LangSmith can trace these graphs the same way (env vars). Learn graphs here first; traces are Part 6b.

Run from repo root:

```powershell
streamlit run .\langgraph\chatbot\frontend.py
```

---

## What not to study

| Path | Why skip |
| --- | --- |
| `venv/` `venv312/` | Installed packages, not your code |
| `test.py` | Scratch |
| `chatbot.db` | SQLite data for the Streamlit bot |
| `.streamlit/config.toml` | Streamlit server flag, not a lesson |
| `chatModels/chat_templete.py` | Unfinished |
| `langgraph/langsmith/islr.pdf` | Textbook corpus, not a lesson |
| `langgraph/langsmith/faiss_islr/` | Saved MiniLM index from `ragv1` / `ragv2` |

---

## If you get lost

Ask: **which part of the pipe am I in?**

| Symptom | Go back to |
| --- | --- |
| “What even is `invoke`?” | `LLMs/llm_demo.py` |
| “Why messages not a string?” | `chatModels/messages.py` |
| “What does `\|` do?” | `masti-with-chains/simple-chain.py` then `runnables/sequence.py` |
| “Where do PDFs become text?” | `doc_loaders/` |
| “Why split / embed / Chroma?” | Parts 5.5–5.7 |
| “How does RAG actually answer?” | `yt-chatbot/chatbot.py` |
| “When does the model call Python?” | `tools-calling/calling.py` |
| “Agent vs chain?” | `agents/notes.md` then `agent.py` |
| “Where did my LangSmith spans go?” | `langgraph/langsmith/notes.md` (nesting + APAC 403) |
| “What is p50 / p95?” | same notes, percentiles section |
| “Why LangGraph if I have LCEL?” | `langgraph/langchain_vs_langgraph/notes.md` |
| “What is `thread_id`?” | `langgraph/persistence/notes.md` then `chatbot.ipynb` |

Setup, keys, and how to run scripts: [`README.md`](README.md).
