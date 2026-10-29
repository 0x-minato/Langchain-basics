# YouTube transcript RAG chatbot

This folder is the first time the **whole RAG pipeline** runs as one script. Earlier folders practised one stage each. `chatbot.py` wires them:

```text
YouTube captions → split → embed → FAISS
                                      ↓
question → retrieve top-k → stuff {context} → ChatCohere → string
```

`rag/notes.md` is the **idea**. This file is the **product**: ask a question about **one video**, get an answer **only from its transcript**.

Corpus is not `cricket.txt`. It is the captions for `video_id = "kHCamT4zfH4"` (a Rome video). No chat loop, no disk persistence — index is rebuilt every run.

Needs `CO_API_KEY` / `COHERE_API_KEY` in `.env`. Run from the **repo root** with `venv312`.

---

## What this app does

**Index time** (once per process start):

1. Fetch English captions with `youtube_transcript_api`.
2. Join snippet text into one long string.
3. Split with `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`).
4. Embed chunks with Cohere `embed-english-v3.0`.
5. Put vectors in **FAISS** (in memory).

**Query time** (one hardcoded question):

1. Embed the question, similarity-search top **4** chunks.
2. Join those chunks into `{context}`.
3. Prompt: answer **only** from context, else say you don’t know.
4. `ChatCohere` (`command-a-03-2025`) generates; `StrOutputParser` unwraps to a string.

That is index → retrieve → augment → generate. Same four stages as `rag/notes.md`.

---

## `chatbot.py` — walkthrough

### 1. Load the transcript (the “document loader”)

YouTube is not a `TextLoader`. Captions come from Google’s timed-text API via `youtube_transcript_api`.

```python
ytt_api = YouTubeTranscriptApi()
transcript_list = ytt_api.fetch(video_id, languages=["en"])
transcript_global = " ".join(chunk.text for chunk in transcript_list)
```

`fetch` returns a list of snippet objects (`FetchedTranscriptSnippet`), **not** dicts. Use `chunk.text`, not `chunk["text"]`.

`video_id` is the 11-character id (`kHCamT4zfH4`). Not the watch URL, not `v=kHCamT4zfH4`.

Two different “no captions” errors — catch both:

| Exception | Meaning |
| --- | --- |
| `TranscriptsDisabled` | owner turned captions off |
| `NoTranscriptFound` | this language list has no track (e.g. Hindi-only auto captions, you asked for `en`) |

`languages=["en"]` is a **preference list**, first match wins. For a Hindi-only video you need `["en", "hi"]` or `"hi"`. A comment in the file still talks about Hindi auto-captions from an earlier video; this Rome id has English.

**Old API:** `YouTubeTranscriptApi.get_transcript(video_id)` was removed in v1. New style is an instance + `fetch` / `list`.

If the except branch prints `"No captions available"`, `transcript_global` stays `''`. Splitting an empty string still “works” and you index nothing useful — the model will just say it doesn’t know.

### 2. Split

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript_global])
```

Captions are one blob of spoken language (no headings). Recursive splitting on paragraphs / sentences is the right default. See `text_splitting/notes.md`.

`create_documents([string])` wraps each chunk as a LangChain `Document`. `FAISS.from_documents` needs that, not raw strings.

Overlap 200 keeps a sentence that was cut at the 1000-char boundary from disappearing from both chunks.

### 3. Embed + FAISS

```python
embedding = CohereEmbeddings(model="embed-english-v3.0")
vector_store = FAISS.from_documents(chunks, embedding)
```

**Chat model ≠ embedding model.** `ChatCohere` generates text. `CohereEmbeddings` turns text into vectors. Using `ChatCohere` here will fail.

`from_documents` embeds **every chunk** (API calls, costs trial quota). FAISS then holds `(vector, text, id)` in **RAM**. Kill the process, the index is gone. Next run embeds again. Chroma in `vector_store/` can persist; this script does not.

`langchain_community.vectorstores.FAISS` still works. You get a sunset warning; it is not a crash. Standalone package is `langchain-community` → later `langchain-community` / FAISS extras.

### 4. Prompt (augmentation)

```python
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer only from the provided transcript context
    If context is insufficient, just say you dont know

    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)
```

Two variables. The RAG chain must produce a **dict** with those keys before this prompt runs.

The “only from context / don’t know” line is what makes this RAG instead of a generic chatbot guessing about Rome from pretrain.

Watch the quotes: `template=""""` (four quotes) accidentally puts a stray `"` in the prompt. Use `"""` to open a triple-quoted string.

### 5. Retriever

```python
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 4},
)
```

| Valid `search_type` | Invalid |
| --- | --- |
| `'similarity'` (default) | `'similarity_search'` — that is the **method name**, not the type |
| `'mmr'` | |

`k: 4` = four chunks into the prompt. Too small → miss the fact. Too big → noise and cost.

A retriever’s `invoke` takes a **string query**. Not a dict. That matters in the next section.

### 6. Format docs

```python
def get_context_text(input):
    return " ".join(doc.page_content for doc in input)
```

Retriever returns `list[Document]`. The prompt wants a **string** `{context}`. The lambda is `format_docs` from the RAG notes. Wrapping it in `RunnableLambda` makes it pipeable.

If you pass the list straight into the prompt, `{context}` becomes a Python repr of Document objects, not the spoken text.

### 7. LCEL RAG chain

```python
parallel_chain = RunnableParallel({
    'question': RunnablePassthrough(),
    'context': retriever | RunnableLambda(get_context_text),
})

last_chain = prompt | model | parser
final_chain = parallel_chain | last_chain
answer = final_chain.invoke(question)   # a string
```

Same graph as `rag/notes.md` / `runnables/passthrough.py`:

```text
                    ┌─ Passthrough ──────────────► question ─┐
  "how beautiful is rome"                                     ├─ prompt ─ model ─ parser
                    └─ retriever ─ join page_content ► context ┘
```

`RunnableParallel` runs both branches on the **same** input, returns `{'question': str, 'context': str}`. That dict is exactly what `PromptTemplate` needs.

`RunnablePassthrough` copies the question forward so it is not lost when the retriever consumes it (same reason as keeping the joke in `runnables/passthrough.py`).

Generation: `ChatCohere` + `StrOutputParser` → plain string. Without the parser you print an `AIMessage`.

---

## The bug that crashed query-time embeddings

Cohere **422**:

```text
invalid type: parameter 'texts.0' is of type object but should be of type string
```

FAISS **index** built fine (chunk strings). The failure was `embed_query` on the **question**.

Cause: this invoke:

```python
final_chain.invoke({'question': question})
```

`parallel_chain` forwarded the **whole dict** into the retriever. Retrievers embed a string. Cohere got an object, retried (those “Retrying in 4 seconds” lines are tenacity on a **422**, not a rate limit), then crashed.

**Fix used in the file:** invoke with the string.

```python
final_chain.invoke(question)
```

Passthrough then stores the string as `question`. Retriever embeds the string.

**If you want dict input** (several keys, later a chat API):

```python
from operator import itemgetter

parallel_chain = RunnableParallel({
    'question': itemgetter('question'),
    'context': itemgetter('question') | retriever | RunnableLambda(get_context_text),
})

final_chain.invoke({'question': question})
```

`itemgetter('question')` pulls the string **before** FAISS / Cohere see it.

Mental rule:

| Invoke with | Retriever receives | Works with this parallel map? |
| --- | --- | --- |
| `"how beautiful is rome"` | that string | **yes** (`Passthrough`) |
| `{'question': '...'}` | the dict | **no** unless you `itemgetter` first |

---

## Why not a normal chatbot?

`chatModels/chatbot.py` is ICL: system prompt + last N messages. Mika’s facts live in **weights + the prompt you typed**.

This script is RAG: Rome facts live in **captions**. If the transcript never says it, the prompt tells the model to refuse. That is the point of `{context}`.

No `MessagesPlaceholder`, no `history[-10:]`. One question per process. Adding a loop is: keep `vector_store` alive, `invoke` each new user string, optionally append Q/A to a history list (history still would **not** be in FAISS unless you index it).

---

## FAISS vs Chroma here

| | This file (FAISS) | `vector_store/chroma_db.py` |
| --- | --- | --- |
| Lifetime | this Python process | folder on disk |
| Re-embed every run | **yes** | no, if you skip wipe |
| Extra server | no | no (embedded Chroma) |
| Filters / CRUD | not used | add / get / update / delete |
| Typical use | notebook, one video, demo | app that starts often |

Both are vector **stores**. `as_retriever()` is the same idea. Production YouTube Q&A would persist the index (or a real DB) so you don’t pay Cohere embed on every `python chatbot.py`.

---

## Models

| Role | Class | Id |
| --- | --- | --- |
| Embed chunks + query | `CohereEmbeddings` | `embed-english-v3.0` |
| Generate answer | `ChatCohere` | `command-a-03-2025` |

Same vendor, two APIs. Trial embed limits (`x-trial-endpoint-call-remaining`) drop on **index** (17 chunks) **and** on each question. A 422 with remaining ~96 is a bad payload, not quota.

English embed model + English captions. Hindi transcript + `embed-english-v3.0` is the wrong pair.

---

## Gotchas this script already hit

| Symptom | Cause | Fix |
| --- | --- | --- |
| `get_transcript` missing | youtube_transcript_api v1 | `YouTubeTranscriptApi().fetch(...)` |
| `chunk["text"]` TypeError | snippets are objects | `chunk.text` |
| Invalid video / empty captions | id included `v=` or wrong id | 11-char id only |
| `NoTranscriptFound` | no track in `languages=` | add `"hi"` or the actual language |
| Can’t embed / weird ChatCohere error at index | used chat model as embedder | `CohereEmbeddings` |
| `search_type` ValueError | `'similarity_search'` | `'similarity'` |
| Cohere 422 `texts.0` is object | retriever got a dict | `invoke(question)` or `itemgetter('question')` |
| Retrying in 4 seconds | tenacity retrying 422 | fix the input type; not a sleep/rate issue |
| `langchain-community` sunset warning | FAISS import path | ignore for now, or migrate later |
| Prompt starts with a quote | `template=""""` | `template="""` |

---

## Map onto the rest of the repo

| Stage | This file | Practised in |
| --- | --- | --- |
| Load | `YouTubeTranscriptApi.fetch` | `doc_loaders/` (different source, same `Document` idea) |
| Split | `RecursiveCharacterTextSplitter` | `text_splitting/` |
| Embed | `CohereEmbeddings` | `embeddedModels/` |
| Index | `FAISS.from_documents` | `vector_store/` |
| Retrieve | `as_retriever` + `k=4` | `retrievers/` |
| Chain | `RunnableParallel` + passthrough + lambda | `masti-with-chains/`, `runnables/` |
| Generate | `ChatCohere` + `StrOutputParser` | `chatModels/` |
| Pattern name | RAG | `rag/notes.md` |

---

## What this is not (yet)

- **Not** multi-turn chat (no history).
- **Not** persistent (FAISS dies with the process).
- **Not** streaming.
- **Not** citations (`Document.metadata` is unused; YouTube snippets could store start time).
- **Not** hybrid / MMR / compression (naive similarity only).
- **Not** a production service (no API, auth, eval). For a production-shaped RAG app see Microsoft’s [azure-search-openai-demo](https://github.com/Azure-Samples/azure-search-openai-demo).

Natural next steps, still in this learning path: `input()` loop, save FAISS with `save_local` / `load_local`, print source snippets, `search_type='mmr'`.

---

## Mental model

```text
captions  = the book
chunks    = pages
FAISS     = the library index
retriever = librarian (needs a spoken question, not a form)
prompt    = open those pages on the desk
LLM       = write the essay from those pages only
```

- Index time = build the library (slow, costs embed calls).
- Query time = ask the librarian, then the student.
- `invoke(dict)` vs `invoke(str)` is the difference between handing the librarian a **form** and a **sentence**.

---

## Run

```powershell
python .\yt-chatbot\chatbot.py
```

Change `question` in the file (or later wrap `input()`). Change `video_id` to another video that **has** captions in `languages`.
