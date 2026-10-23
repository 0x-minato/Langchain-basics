# Text Splitting

This folder sits **after document loaders** and **before embeddings**.

```text
loader → splitter → embed → search / LLM
```

`doc_loaders/` gives you `Document`s. Models cannot swallow a 400-page PDF in one prompt, and one giant embedding is a bad search unit. You cut `page_content` into chunks here, then `embeddedModels/` turns those chunks into vectors.

There are no splitter scripts in this folder yet — only these notes. When you add files, the classes live in `langchain_text_splitters`.

Many embedding models and language models have a **maximum input size**. Splitting a long document into smaller chunks lets you stay under that limit instead of sending the whole file at once.

Smaller chunks also help downstream tasks:

| Use | Why split |
| --- | --- |
| Embedding | Short chunks yield more accurate vectors |
| Semantic search | Results point to focused info, not a whole noisy page |
| Summarization | Less hallucination and topic drift |
| Performance | Less memory, easier to process chunks in parallel |

The two knobs you will see on almost every splitter:

- **`chunk_size`** — maximum size of one chunk (characters or tokens)
- **`chunk_overlap`** — characters/tokens copied from the previous chunk into the next, so a sentence is not cut with zero context

---

## 1. Length-based splitting

The simplest splitter. It cuts the document **purely by size** (usually character count, sometimes token count).

LangChain class: **`CharacterTextSplitter`**

It does not care about sentences, paragraphs, or meaning. If the limit is 500 characters, it cuts at 500. Overlap keeps a few characters from the previous chunk so the next one is not context-free.

Fast and cheap. Good for uniform text. Bad when you need clean sentence boundaries — then use recursive / semantic splitters.

### Advantages

- Simple to configure (`chunk_size` + `chunk_overlap`)
- Fast and cheap — no NLP, no extra model, just counting
- Predictable chunk sizes, so you stay under model limits reliably
- Works on any language or messy text (code, logs, scraped pages)

### Disadvantages

- Cuts in the middle of sentences, words, or topics
- Chunks can mix unrelated ideas, or split one idea across two chunks
- Overlap helps a bit but duplicates text (more embeddings, more cost)
- Worse retrieval quality than recursive or semantic splitting on natural language

### Tiny example

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",          # empty = split by characters
    chunk_size=200,
    chunk_overlap=20,
)
chunks = splitter.split_text(text)
```

---

## 2. Text structure-based splitting

Splits by the **layout** of the document, not a raw character count. It tries larger structure first, then smaller:

1. paragraphs (`\n\n`)
2. lines (`\n`)
3. sentences (`. `)
4. words (` `)
5. characters (last resort)

LangChain class: **`RecursiveCharacterTextSplitter`**

`chunk_size` is still a max length, but the cut prefers a natural boundary so a paragraph or heading stays together when it fits.

This still treats the file as **plain text**. If the file actually has markdown headings, HTML tags, JSON keys, or Python functions, use **document-based** splitting below.

### Advantages

- Keeps paragraphs, headings, and sentences more intact than length-only splitting
- Better chunks for RAG / embedding — related sentences tend to stay in one piece
- Still fast and local — no extra model, just separator rules + size limit
- Works well on articles, blogs, markdown, and most normal documents

### Disadvantages

- Chunk sizes are uneven (short heading vs long paragraph)
- Needs real structure (newlines, punctuation). Messy OCR, minified HTML, or one giant line falls back to length-based cuts
- Does not understand meaning — two unrelated paragraphs can still land in one chunk
- Needs a bit more tuning (`separators`, `chunk_size`) than a plain `CharacterTextSplitter`

### Tiny example

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_text(text)
```

---

## 3. Document-based splitting

Splits using the **file format’s own structure**, not generic newlines. Markdown headings, HTML tags, JSON keys, and code (`def` / `class`) become the cut points.

LangChain classes:

| Format | Class |
| --- | --- |
| Markdown (`#`, `##`) | **`MarkdownHeaderTextSplitter`** |
| HTML (`<h1>`, `<h2>`) | **`HTMLHeaderTextSplitter`** |
| JSON | **`RecursiveJsonSplitter`** |
| Python / JS / etc. | **`RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`** |

This is the heading-**tree** idea. A markdown file is walked as:

```text
# Chapter          → H1
  ## Setup         → H1 / H2
    paragraph A
  ## Usage         → H1 / H2
    paragraph B
```

Each chunk keeps **header metadata** (`Header 1: Chapter`, `Header 2: Setup`) so later retrieval knows *where* the text sat, not just the words.

Unlike recursive splitting, the primary cut is a **tag or heading**, not `\n\n`. If one section is still huge, you usually run a recursive splitter *on that section* as a second pass.

Not merge sort. Closer to parsing a document tree, then emitting one (or a few) chunks per node.

### Advantages

- Chunks follow real sections (a markdown `##` or a Python function), so RAG results are on-topic
- Headers are stored in `metadata` — you can show “from Setup” or filter by chapter
- Best choice when the source is already structured (docs site, README, `.py`, API JSON)
- Avoids gluing two unrelated chapters just because they fit in `chunk_size`

### Disadvantages

- Useless on a wall of text with no headings/tags — falls back to looking like recursive/length splitting, or fails to split at all
- Section sizes are wildly uneven (a 2-line heading vs a 3,000-word chapter). Huge sections still need a second splitter
- You must pick the splitter that matches the format (markdown splitter on HTML will not see `<h1>`)
- Slightly more setup: list which headers to split on, then maybe recursive-split leftovers

### Tiny example

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)

# chunks[i].metadata == {"Header 1": "Chapter", "Header 2": "Setup"}
```

For code:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_text(python_source)
```

---

## 4. Semantic meaning-based splitting

Splits where the **topic changes**, not where a newline or heading happens. Consecutive sentences are embedded; a cut is made when similarity **drops** (the next sentence is about something else).

LangChain class: **`SemanticChunker`** (`langchain_experimental.text_splitter`)

Needs an **embedding model** (the same idea as `embeddedModels/`): MiniLM locally, or OpenAI / Cohere in the cloud.

Flow:

```text
text → sentences → embed each sentence → cosine vs the next one
                                          ↓
                         similar  → keep in the same chunk
                         drop     → start a new chunk
```

Example: a paragraph about Kohli batting, then one about Bumrah yorkers. Recursive splitting might glue them because they fit in `chunk_size`. Semantic splitting cuts between them because the vectors jump.

You tune **how sharp** a drop counts as a boundary (`breakpoint_threshold_type`: percentile, standard deviation, …). There is no hard `chunk_size` in the same way — chunks grow until meaning shifts. Overlap is usually not the main knob; the breakpoint is.

This is **not** a tree and **not** merge sort. It is “split on embedding distance.”

### Advantages

- Chunks follow **ideas**, so RAG hits a full thought instead of half a paragraph plus an unrelated next one
- Works on prose with **no** markdown/HTML (articles, transcripts, PDFs that lost headings)
- Often the best retrieval quality of the four methods when embeddings are decent
- You can reuse the same embedding model you already use for the vector store

### Disadvantages

- Slow and **not free** on large corpora — every sentence is embedded before you even store chunks
- Chunk sizes are unpredictable (a 2-sentence aside vs a long on-topic section). Can overflow model context unless you cap / re-split
- Quality depends on the embedding model. A weak embedder makes bad cuts
- Extra dependency (`langchain-experimental`) and extra config (breakpoint type/threshold)
- Overkill for logs, code, or already well-headed docs — use length / document splitters there

### Tiny example

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
)
chunks = splitter.split_text(text)
```

Install: `pip install langchain-experimental`

---

## Which one to use?

| Situation | Prefer |
| --- | --- |
| Logs, code dumps, unknown messy text | Length-based |
| Must never exceed a hard size | Length-based (or token splitter) |
| Articles, PDFs, blogs, notes | Structure-based (recursive) |
| Markdown / HTML / JSON / source code | Document-based (headers, language, JSON) |
| Prose with topic shifts, best RAG quality | Semantic (embed sentences, cut on similarity drop) |

**Rule of thumb:** headings/code → **document** splitter. Plain articles → `RecursiveCharacterTextSplitter`. Need meaning-aware chunks and you can pay for embeddings → **semantic**. No structure at all → `CharacterTextSplitter`.
