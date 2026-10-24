# RAG — Retrieval-Augmented Generation

This folder is the **whole path you already built**, named as one product pattern.

```text
loader → splitter → embed → index (vector store)
                              ↓
question → retrieve → stuff chunks into the prompt → generate
```

`doc_loaders/` + `text_splitting/` + `embeddedModels/` + `vector_store/` + `retrievers/` **are** RAG. The LLM is only the last step. Without retrieve + stuff, you just have a chatbot guessing from weights.

---

## What is RAG?

**RAG** = at answer time, **look up** relevant text, **add it to the prompt**, then **generate**.

```text
User: who is a good bowler?

1. RETRIEVE   Chroma / Wikipedia / BM25  →  Bumrah chunk, Cummins chunk
2. AUGMENT    prompt = system + those chunks + the question
3. GENERATE   Cohere / Gemini / Qwen writes the answer from that context
```

The model is not “trained on your PDFs.” It is **shown** the right pages **this turn**, like an open-book exam.

Three words:

| Word | Meaning |
| --- | --- |
| **Retrieval** | find top-k `Document`s (`retriever.invoke`) |
| **Augmented** | those docs become `{context}` in the prompt |
| **Generation** | the LLM writes the reply |

Contrast with a bare `model.invoke("who is a good bowler")`: that uses **only** pretrain + RLHF. No `cricket.txt`, no Chroma, no citations.

LangChain shape you already sketched in `retrievers/notes.md`:

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

That **is** a RAG chain.

---

## Why we need RAG

A frozen LLM has four holes. RAG plugs them without a training run.

### 1. The model does not know *your* data

Weights were trained on a public scrape. They do not contain:

- this repo’s `cricket.txt`
- a company’s Confluence / tickets / policies
- last week’s PDF

If you ask anyway, it **hallucinates** a confident answer. RAG injects the real paragraph.

### 2. Knowledge goes stale

A 2024 checkpoint will not know a 2026 product change. Fine-tuning every week is expensive. **Re-index** a new doc in Chroma in seconds.

### 3. Context windows are finite

You cannot paste 400-page PDFs into every prompt. RAG sends **4–8 chunks**, not the library. That is why you split and index.

### 4. You want *evidence*, not vibes

Good RAG returns `page_content` **and** metadata (`source`, page). You can cite “from `cricket.txt`.” Fine-tuning buries the fact in weights — no footnote.

### 5. Cost and control

Updating a vector store ≠ GPU training. You can **delete** a wrong chunk (`store.delete`). You cannot surgically delete one fact from a 70B checkpoint.

**One line:** RAG is how a generic model answers questions about **a corpus you own**, **today**, with **quotes**.

---

## In-context learning (ICL)

**In-context learning** = the model “learns” from **tokens in this prompt only**. Weights do not change.

You already do this:

- **Zero-shot:** just the question.  
- **Few-shot:** examples in the prompt (`chat_history.txt` You:/Mika: lines).  
- **RAG:** the “examples” are **retrieved documents**, not hand-picked demos.

```text
[system] [few-shot or retrieved chunks] [user question]
              ▲
              this is the “context” in ICL
```

The sliding window in `chatbot.py` is also ICL: last 10 messages teach tone for **this** session. Close the process and it is gone.

### What ICL is good at

- Style, format, a short schema (“reply as JSON”).
- Using facts that **fit in the window** (RAG chunks, a pasted email).
- Trying a new task **without** training.

### Limits

- Window cap (8k–200k tokens). More docs ≠ more ICL; they fall off.
- Every request **re-sends** the context (cost).
- The model does not *remember* the PDF tomorrow unless you retrieve again.
- Too many few-shots or noisy chunks → distraction / hallucination.

**One line:** ICL is a **sticky note on the exam paper**. Next exam, blank paper again.

---

## Fine-tuning

**Fine-tuning** = **change the weights** on extra data. After training, the new behaviour is **inside the file**. You do not paste the textbook every time.

Typical loop:

```text
base model (Qwen / Command A / GPT)
    → SFT on (prompt, ideal answer) pairs
    → optional DPO / RLHF
    → new checkpoint
```

Kinds you will hear:

| Kind | What you change |
| --- | --- |
| **SFT** (supervised) | “when user says X, answer like Y” |
| **Instruction tune** | follow orders, chat format |
| **Domain tune** | medical / legal / cricket commentary *style* |
| **Preference (DPO/RLHF)** | which of two answers is better |
| **LoRA / QLoRA** | small adapter, not full 7B update (fits a 3050 better) |

### What fine-tuning is good at

- **Voice and format** that must hold for every user (support bot always terse, always JSON).
- Skills that are *procedures*, not look-ups (extract fields, use a tool, speak like Mika without a 400-token system prompt).
- A dialect / jargon the base model keeps mangling even with ICL.
- Lower **per-request** tokens (shorter system prompt) once the skill is in weights.

### Limits

- **Does not replace a knowledge base.** Tuning on 500 docs does not give reliable citations, and new docs need another train.
- **Catastrophic forgetting** — new tune can hurt old skills (the PhD-forgetting analogy).
- Cost: data labelling, GPUs, eval, versioning checkpoints.
- Hard to unlearn one bad paragraph. Easy to delete one Chroma id.
- Stale the day the policy PDF changes.

**One line:** fine-tuning is **rewriting the textbook into the student’s brain**. RAG is **handing them the page**.

---

## Fine-tuning vs in-context learning

Both change behaviour. **Where** the change lives is the difference.

| | In-context learning | Fine-tuning |
| --- | --- | --- |
| What changes | **this prompt** | **weights** (a new model file) |
| Lasts after process exit | no | yes, until you train again |
| Data size | what fits in the window | thousands–millions of examples |
| Cost | pay tokens every call | pay GPUs once, cheaper/shorter prompts later |
| Update a fact | edit the prompt or the store | new training run |
| Best for | one-off tasks, RAG, few-shot style | stable skill / voice / format |
| Risk | context overflow, ignored middle | forgetting old skills, overfitting |
| Your repo | `chatbot.py` history, RAG `{context}` | you have not done this yet (would be extra training) |

RAG is a **special case of ICL**: the context is **chosen by a retriever**, not typed by you.

```text
                    ┌── ICL ── few-shot / system prompt / RAG chunks
how the model uses extra info
                    └── Fine-tune ── baked into weights
```

You can **combine** them: fine-tune for “answer only from context, cite sources,” **and** RAG to supply that context.

---

## RAG vs fine-tuning vs ICL (one table)

| Need | Prefer |
| --- | --- |
| Private / changing facts | **RAG** |
| “Speak like our brand” on every reply | **Fine-tune** (or a strong system prompt if short) |
| 3 examples of a new format, tonight | **ICL** few-shot |
| Cite page 12 | **RAG** |
| Model keeps refusing a JSON schema | Fine-tune or bigger model; ICL first |
| 2 million internal docs | **RAG** (never fine-tune the whole dump) |

---

## The RAG pipeline (four stages)

Interviews and papers split RAG into **index-time** and **query-time**.

```text
INDEX TIME (once, then when docs change)
  load → split → embed → INDEX

QUERY TIME (every question)
  retrieve → augment → generate
```

### 1. Indexing

**Goal:** turn the corpus into something searchable.

Steps you already know:

1. **Load** — `TextLoader`, `PyPDFLoader`, `WebBaseLoader`, `DirectoryLoader`.
2. **Split** — length / recursive / document / semantic (`text_splitting/notes.md`). Chunks must be embeddable and useful.
3. **Embed** — MiniLM `embed_documents` (`embeddedModels/`).
4. **Index** — write `(vector, text, metadata, id)` into Chroma/FAISS. Build HNSW/IVF so search is not `O(n)`.

This is **not** the user question. No LLM required (unless you use a semantic splitter).

Re-index when files change: add / update / delete in `chroma_db.py`. That **is** how RAG stays current.

Bad index → bad RAG. Garbage chunks, no overlap, huge pages, missing metadata.

### 2. Retrieval

**Goal:** given the question, get the **right** `Document`s.

```text
question → (optional rewrite: multi-query / HyDE)
         → embed_query
         → ANN / BM25 / hybrid
         → top-k  (± MMR, filters, compression)
```

`store.as_retriever(search_kwargs={"k": 4})` is the default. Wrappers (MMR, multi-query, compression) live in `retrievers/notes.md`.

Knobs:

- `k` too small → miss the fact.  
- `k` too large → noise, cost, “lost in the middle.”  
- Hybrid (BM25 + vectors) for SKUs and names.

Retrieval quality **caps** generation quality. The LLM cannot use a paragraph it never received.

### 3. Augmentation

**Goal:** **stuff** retrieved text into the prompt so the model is grounded.

```text
System: Answer ONLY from the context. If missing, say you don't know.

Context:
[1] Bumrah ... yorkers ...   (source: cricket.txt)
[2] Cummins ... Test captain

Question: who is a good bowler?
```

Augmentation is boring and critical:

- `format_docs`: join `page_content` (and metadata for citations).
- Prompt template with `{context}` and `{question}`.
- Optional: “use quotes,” “list sources,” “don’t invent.”

This is **ICL**. The weights still do not contain Bumrah; the **prompt** does, for this call.

Failure modes:

- Context too long → truncate, or compress first.  
- No instruction to stay in context → model ignores the docs and freestyles.  
- Best chunk buried in the middle → reorder (best at start and end).

### 4. Generation

**Goal:** the LLM writes the user-facing answer **conditioned on** the augmented prompt.

```text
augmented prompt → ChatCohere / Gemini / Qwen → string (StrOutputParser)
```

Then you can parse, stream, or add citations from `doc.metadata`.

Generation still **can** hallucinate if:

- retrieval returned the wrong chunks
- the prompt did not say “only from context”
- the model is tiny (Qwen 3B) and ignores instructions

RAG **reduces** hallucination; it does not make it zero.

---

## End-to-end picture

```text
                 INDEX TIME
  cricket.txt ─load─► chunks ─embed─► Chroma index
                                      (vectors + text + ids)

                 QUERY TIME
  "good bowler" ─retrieve─► [Bumrah, Cummins]
                 ─augment─► prompt with those texts
                 ─generate─► "Bumrah is known for yorkers..."
```

| Stage | Folder in this repo | LLM? |
| --- | --- | --- |
| Index | `doc_loaders`, `text_splitting`, `embeddedModels`, `vector_store` | no (usually) |
| Retrieve | `retrievers`, `vector_store` | only if multi-query / HyDE / compression |
| Augment | prompt template | no |
| Generate | `chatModels` / `LLMs` | **yes** |

---

## When a product needs RAG vs fine-tuning

Use this in design interviews.

### Ship **RAG** when

- Answers must come from **a corpus that changes** (docs, policies, tickets, SKUs).
- You need **citations** or “show me the source.”
- The fact set is **large** (thousands of pages). You will never fine-tune on all of it reliably.
- Legal / medical **grounding** — the model should refuse if the chunk is missing.
- A **small team**: no GPU training pipeline, just Chroma + an API model.
- Multi-tenant: each customer’s files stay in **their** index; one shared LLM.

Examples: internal Q&A over Notion, “ask this PDF,” support bot over help center, this cricket demo.

### Ship **fine-tuning** when

- The problem is **behaviour**, not look-up: tone, language, always-JSON, tool-calling pattern.
- The same skill runs on **every** request and a 2k system prompt is too slow/expensive.
- ICL **cannot** teach it (tiny model, or the format is long).
- You have **labelled** (input, output) pairs and an eval set.
- Domain **jargon / reasoning style** (how a junior analyst writes), while facts still come from a DB.

Examples: extract invoice fields in a rigid schema, always-Hinglish support voice, a coding model specialised on *your* framework.

### Ship **both** when

- Fine-tune: “answer only from context, cite `[n]`, never invent SKUs.”  
- RAG: the SKUs and policies live in the index.

Or: fine-tune embeddings / a reranker for *your* docs, keep a generic chat model.

### Ship **neither** (plain ICL) when

- One-off script, 3 examples, no corpus (`pydanticoutputparser.py` with a JSON example).
- Character bot with a system prompt + short history (`chatbot.py`).
- The fact is already in the base model (“capital of India”).

### Decision sketch

```text
Do we need facts from OUR files / a live DB?
  yes → RAG (always). Fine-tune optional for style.
  no  → Is the issue tone/format/skill the base model fails at?
          yes, and it's stable → fine-tune (try ICL first)
          yes, but tiny / one feature → ICL few-shot
          no → just invoke the model
```

| Product question | RAG | Fine-tune |
| --- | --- | --- |
| “What’s our refund policy?” | **yes** | no |
| “Always reply in two bullets, no emoji” | no | maybe (or system prompt) |
| “Summarise *this* uploaded PDF” | **yes** (or stuff the file if short) | no |
| “Sound like our senior SRE” | optional | **yes** if ICL isn’t enough |
| “New price list every Monday” | **re-index** | do **not** re-train weekly |
| “Never mention competitors” | filter / prompt | **yes** if prompt is unreliable |

**Rule of thumb:** if you would put it in a **wiki**, use **RAG**. If you would put it in an **employee handbook on how to talk**, consider **fine-tune**. If it fits in a **sticky note**, use **ICL**.

---

## Mental model

```text
ICL     = notes on the desk for this exam
RAG     = librarian fetches the right notes, then ICL
Fine-tune = you go back to school and rewrite the brain

Index   = build the library
Retrieve = pick the books
Augment = open them on the desk
Generate = write the essay
```

- The LLM **generates**. It does not magically know `cricket.txt`.
- **RAG** is ICL with a retriever in front.
- **Fine-tuning** changes the model; **indexing** changes the library.
- Products that answer from **documents** start with RAG. Add fine-tuning only when style or skill will not stay in a prompt.
