# langchain_models

Personal LangChain practice repo: call models, parse output, build LCEL chains, load documents, split text, and embed for search.

Each topic folder has a `notes.md`. Read those when you forget *why* a file exists. Run scripts from this repo root, with **`venv312`** active (Python 3.12). System Python 3.14 will not see these packages.

```text
py-basics → LLMs → chatModels → embeddedModels
        → masti-with-chains → runnables
        → doc_loaders → text_splitting
```

---

## Setup

```powershell
python3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1
pip install -r requirements.txt
```

Extra packages used by later scripts (not all are in `requirements.txt` yet):

```powershell
pip install langchain-cohere langchain-community grandalf beautifulsoup4 lxml pypdf bitsandbytes
```

Copy `.env` keys locally (the file is gitignored):

```env
OPENAI_API_KEY=
GOOGLE_API_KEY=
ANTHROPIC_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
CO_API_KEY=
```

Cohere also accepts `COHERE_API_KEY`. Hugging Face local scripts point caches at `D:\huggingface_cache` via `HF_HOME`.

Run anything like:

```powershell
python .\LLMs\llm_demo.py
python .\runnables\lambda.py
python .\doc_loaders\text_loader.py
```

---

## Folders

| Folder | What you practise |
| --- | --- |
| [`py-basics/`](py-basics/notes.md) | `TypedDict` vs Pydantic — schemas before parsers |
| [`LLMs/`](LLMs/notes.md) | First `invoke` (Gemini chat model) |
| [`chatModels/`](chatModels/notes.md) | OpenAI / Anthropic / Gemini / local Qwen, messages, chatbot, output parsers |
| [`embeddedModels/`](embeddedModels/notes.md) | Sentence embeddings + cosine similarity search |
| [`masti-with-chains/`](masti-with-chains/notes.md) | LCEL: simple, sequential, parallel, conditional chains |
| [`runnables/`](runnables/notes.md) | `RunnableSequence`, `Parallel`, `Passthrough`, `Lambda`, `Branch` |
| [`doc_loaders/`](doc_loaders/notes.md) | `TextLoader`, PDF, directory, `WebBaseLoader` |
| [`text_splitting/`](text_splitting/notes.md) | Length, structure, and document-based chunking |

`venv/` and `venv312/` are local environments. Do not commit them (see `.gitignore`).

---

## Models used here

| Kind | Typical id in this repo |
| --- | --- |
| Cloud chat | Gemini flash, OpenAI GPT-4, Cohere `command-a-03-2025` |
| Local chat | `Qwen/Qwen2.5-3B-Instruct` (4-bit BitsAndBytes, RTX 3050 4GB) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |

Small local models are fine for chat and summaries. Strict JSON / Pydantic / `RunnableBranch` classifiers work more reliably on Gemini or Cohere.

---

## Notes

- Always `print(res)` or `result.content` — printing the chain dumps tokenizer internals.
- LCEL sequence is `a | b`. Parallel is a **dict** of runnables, not `a & b`.
- `chain.invoke(...)` needs an input dict. `{'topic', 'x'}` is a set; use `{'topic': 'x'}`.
- Document loaders: `load()` is eager (full list in RAM), `lazy_load()` is a generator.
- Preview any `notes.md` with **Ctrl+Shift+V**.
