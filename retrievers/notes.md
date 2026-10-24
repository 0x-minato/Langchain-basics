# Retrievers

This folder sits **after the vector store**. The store *holds* vectors. A retriever is the **object you ask a question** and get `Document`s back.

```text
loader → splitter → embed → vector store → retriever → (optional compress) → LLM
```

In `vector_store/chroma_db.py` you called `similarity_search` yourself. A retriever wraps that (or Wikipedia, BM25, …) behind one method:

```python
docs = retriever.invoke("who is a good bowler")
```

LangChain chains and agents talk to a **retriever**, not to FAISS/Chroma directly. Swap the backend; the chain stays `retriever | prompt | model`.

---

## What is a retriever?

A **retriever** is anything that implements:

> given a query string → return a list of `Document`s

Interface: `BaseRetriever`

| Method | Job |
| --- | --- |
| `invoke(query)` | LCEL; what you use now |
| `get_relevant_documents(query)` | older name, same idea |
| `batch([...])` | several queries |

It does **not** have to be a vector store. Wikipedia search, BM25 keyword search, a SQL `WHERE`, or an HTTP API can all be retrievers.

### Retriever vs vector store

| | Vector store | Retriever |
| --- | --- | --- |
| Job | persist + index embeddings | **fetch** docs for a query |
| Extra ops | add / delete / update / filter | usually just “search” |
| LangChain | `Chroma`, `FAISS` | `BaseRetriever` |
| Link | `store.as_retriever()` | wraps the store |

**One line:** store = library shelves. Retriever = the librarian you ask.

### Why wrap a store as a retriever?

- RAG chains expect `retriever.invoke(question)`.
- You can stack **other** retrievers in front/behind (multi-query, compression, ensemble) without changing the LLM code.
- `search_kwargs={"k": 4}` lives on the retriever, not in every `similarity_search` call.

---

## Types of retrievers (map)

Two families:

**1. Where the data lives**

| Type | Source |
| --- | --- |
| Vector store retriever | your Chroma / FAISS / Pinecone |
| Wikipedia / Arxiv / PubMed | live web APIs |
| BM25 | in-memory keyword index over your docs |
| Database / self-query | metadata filters → SQL or store `where` |

**2. How the query is processed** (wrappers on top of another retriever)

| Type | Trick |
| --- | --- |
| Naive similarity | embed query, top-k nearest |
| MMR | top-k that are relevant **and** diverse |
| Multi-query | LLM rewrites the question several ways, merge hits |
| Contextual compression | fetch many, then **shrink / drop** junk before the LLM |
| Ensemble / hybrid | blend BM25 + vectors |
| HyDE | LLM writes a fake answer, embed *that* |
| Parent-document | search small chunks, return the **bigger** parent |
| Self-query | LLM turns English into filters (`country = India`) |

Course list first, then interview extras.

---

## 1. Wikipedia retriever

Not your PDF index. It **calls Wikipedia** at query time and returns page summaries (or full text) as `Document`s.

LangChain: `WikipediaRetriever` (`langchain_community.retrievers`)

Needs `wikipedia` (`pip install wikipedia`).

```python
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=3, lang="en")
docs = retriever.invoke("Jasprit Bumrah")
print(docs[0].metadata["title"], docs[0].page_content[:200])
```

Good for: open-world facts, “who is X”, homework demos with **no** local corpus.

### Advantages

- Zero indexing. No embed, no Chroma folder.
- Always current-ish (live wiki).
- Easy first retriever to demo `invoke`.

### Disadvantages

- Needs network. Rate limits / blocked `User-Agent`.
- You do not control the corpus (not your company docs).
- Pages are long; you often still need a splitter or compressor.
- Bad for private data, code, or cricket bios you wrote yourself.

**Interview line:** “WikipediaRetriever is an *external* retriever. RAG over *my* files uses a vector-store retriever.”

---

## 2. Vector store retriever

The default RAG retriever. Built from the store you already have:

```python
retriever = store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("who is a good bowler")
```

That is `similarity_search` with a uniform API.

```python
retriever = store.as_retriever(
    search_type="similarity",          # or "mmr", "similarity_score_threshold"
    search_kwargs={
        "k": 4,
        "filter": {"country": "India"},  # metadata, same as chroma_db.py
    },
)
```

`search_type="similarity_score_threshold"` drops weak hits (`score_threshold=0.3`). Useful when “nothing relevant” is better than stuffing junk.

Flow:

```text
question → embed_query → ANN / cosine vs stored vectors → top-k Documents
```

Same maths as `embeddedModels/embedding_model.py` and `vector_store/chroma_db.py`.

### Advantages

- Semantic: “yorker specialist” still finds Bumrah.
- Metadata filters on the same call.
- Fast at scale (HNSW etc.).
- Drop-in for LCEL: `retriever | format_docs | prompt | model`.

### Disadvantages

- Only as good as chunking + embedding model.
- Near-duplicate chunks all come back (three Kohli paragraphs).
- Vocabulary mismatch: user says “pace attack”, chunk says “fast bowling” — usually OK; rare jargon can miss.
- No keyword guarantee: exact error codes / SKUs often need BM25 as well.

**Interview line:** “`as_retriever()` is the production starting point. Everything else (MMR, multi-query, compression) is a *wrapper* around this.”

---

## 3. Maximum marginal relevance (MMR) retriever

Similarity search optimises **relevance only**. If the top 5 chunks are all about Bumrah yorkers, the LLM never sees Cummins. **MMR** trades a bit of relevance for **diversity**.

Idea (each pick):

```text
score = λ * similarity(query, doc)  −  (1 − λ) * max similarity(doc, already picked)
```

- High `λ` (e.g. 0.8) → almost normal similarity.
- Low `λ` (e.g. 0.3) → more variety, can get slightly off-topic.

LangChain: same store, different search type.

```python
retriever = store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)
```

| Knob | Meaning |
| --- | --- |
| `k` | how many you **return** |
| `fetch_k` | how many similar docs you **consider**, then pick a diverse `k` |
| `lambda_mult` | relevance vs diversity (`1` = no MMR, `0` = max diversity) |

Also: `store.max_marginal_relevance_search(query, k=4, fetch_k=20)`.

### Advantages

- Less redundant context → better use of the token window.
- Broader coverage (batter + bowler + two countries).
- Cheap: no extra LLM call.

### Disadvantages

- Can drop the 2nd-best *on-topic* chunk for a less relevant one.
- `fetch_k` must be > `k` or MMR has nothing to choose from.
- Still embedding-only; will not invent a missing topic.

**Interview line:** “MMR is diversity-aware nearest neighbour. Use it when top-k is repetitive. `fetch_k` then `k`.”

---

## 4. Multi-query retriever

Users ask badly. One embedding of `"bowler?"` may miss “death overs”, “yorkers”, “pace attack”.

**Multi-query** asks an LLM to rewrite the question into several angles, retrieves for **each**, then unions / dedupes.

```text
"who is a good bowler"
        │
        ▼
   LLM rewrites
        │
        ├─ "famous fast bowlers in cricket"
        ├─ "best death-over yorker specialists"
        └─ "leg-spinners in T20 leagues"
        │
        ▼
   retriever × 3  →  merge unique docs
```

LangChain: `MultiQueryRetriever`

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

base = store.as_retriever(search_kwargs={"k": 3})
retriever = MultiQueryRetriever.from_llm(retriever=base, llm=model)
docs = retriever.invoke("who is a good bowler")
```

Uses your chat model (Cohere / Gemini). Costs **1 LLM call + N searches**.

### Advantages

- Fixes vocabulary gap and vague questions.
- Often the biggest quality jump when users type like humans.
- Still uses your existing vector index.

### Disadvantages

- Slower and **not free** (rewrite call + more embedding queries).
- Bad rewriter → bad extra queries (noise).
- Can pull contradictory chunks; the LLM must reconcile them.

**Interview line:** “Query expansion via an LLM. Related: HyDE (embed a *hypothetical answer* instead of rewrites).”

---

## 5. Contextual compression retriever

Vector search is **recall-heavy**: you fetch 8 chunks so you do not miss the fact. Many of those tokens are useless. The LLM then gets distracted (and you pay for them).

**Contextual compression:** retrieve first, then **keep only the bits that answer this query**.

```text
query → base retriever (k=10) → compressor → k' shorter / fewer docs → LLM
```

LangChain: `ContextualCompressionRetriever`

Compressors you will see:

| Compressor | How it shrinks |
| --- | --- |
| `LLMChainExtractor` | LLM extracts relevant sentences from each chunk |
| `LLMChainFilter` | LLM yes/no: drop the whole chunk |
| `EmbeddingsFilter` | drop chunks whose embedding is far from the query (no LLM) |
| `DocumentCompressorPipeline` | chain several (e.g. splitter → embed filter → extractor) |

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

base = store.as_retriever(search_kwargs={"k": 8})
compressor = LLMChainExtractor.from_llm(model)
retriever = ContextualCompressionRetriever(
    base_retriever=base,
    base_compressor=compressor,
)
docs = retriever.invoke("who is a good bowler")
```

`EmbeddingsFilter` is the cheap version (MiniLM, same as your store).

### Advantages

- Cleaner prompt → fewer hallucinations, lower cost.
- Lets you retrieve **wide** (`k=10`) and send **narrow**.
- Works on Wikipedia hits too (long pages).

### Disadvantages

- Extra LLM (or embed) pass; latency and money.
- Extractor can drop a sentence you actually needed.
- Small local models (Qwen 3B) are weak extractors — use Cohere/Gemini here.

**Interview line:** “Compression is *post-retrieval*. MMR/multi-query change *what* you fetch; compression changes *what you keep*.”

---

## How they stack (typical RAG)

```text
user question
    → MultiQueryRetriever          # optional: expand the question
        → VectorStore (similarity or MMR)
            → ContextualCompression
                → prompt | model | parser
```

Wikipedia can replace the vector store when the corpus is “the public web,” not `cricket.txt`.

| Need | Pick |
| --- | --- |
| My PDFs / Chroma | Vector store retriever |
| Public facts, no index | Wikipedia (or Tavily / Arxiv) |
| Top-k all look the same | MMR |
| Vague user wording | Multi-query (or HyDE) |
| Prompt too long / noisy | Contextual compression |
| Exact IDs, error codes, names | add BM25 / hybrid |

---

## Other retrievers worth knowing (industry + interviews)

These show up in LangChain docs, RAG system design rounds, and production stacks.

### Hybrid: BM25 + vectors (`EnsembleRetriever`)

**BM25** is classic keyword ranking (Elasticsearch / Lucene). Exact tokens win.

**EnsembleRetriever** runs BM25 **and** a vector retriever, then **reciprocal rank fusion (RRF)** merges lists.

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 4
vec = store.as_retriever(search_kwargs={"k": 4})
hybrid = EnsembleRetriever(retrievers=[bm25, vec], weights=[0.4, 0.6])
```

**Why interviews love this:** semantic search misses SKUs (`INV-4412`); BM25 misses paraphrases. Production RAG is usually **hybrid**, not vectors-only. Same idea as Elastic + kNN, Azure AI Search hybrid, Pinecone sparse+dense.

### Self-query retriever

LLM converts the question into a **structured filter + a search string**.

> “Indian bowlers in T20” → query=`bowler`, `where country=India AND format=T20`

Needs a metadata schema (you already have `country`, `role`, `format` in Chroma). More accurate than stuffing filters by hand. Extra LLM call. Breaks if the model invents a field you do not have.

### Parent-document retriever

Embed **small** chunks (good vectors). When a small chunk hits, return the **parent** (whole section / page) so the LLM has context.

Fixes “I retrieved one sentence and the model missed the paragraph.” Uses a docstore for parents + a vector store for children.

### Multi-vector retriever

Several vectors **per** parent: summaries, hypothetical questions, per-sentence embeds. Search those, return the parent. Same family as parent-document. ColBERT / late interaction is the “industry name” for multi-vector token-level search (not always LangChain).

### HyDE (Hypothetical Document Embeddings)

LLM writes a **fake paragraph that would answer** the question. You embed *that*, not the short query. Helps when queries are 4 words and documents are essays. Cost: one generation per question.

### Time-weighted retriever

Score = semantic similarity **+ recency**. Chat memory, news, tickets. “What did we decide last week?” should beat a closer-worded note from 2022.

### Web / search retrievers

`Tavily`, `Brave`, `DuckDuckGo`, `WebResearchRetriever`. Agents that must leave your index. Same *interface*, live HTTP.

### Other built-in sources

`ArxivRetriever`, `PubMedRetriever`, `AzureAISearchRetriever`, `AmazonKendraRetriever`, `GoogleVertexAISearchRetriever`. Enterprise interviews: “we already pay for Kendra / Azure Cognitive Search — wrap it as a retriever.”

### Graph / structured (name-drop)

**GraphRAG** (entities + relations), **SQL / text-to-SQL** retrievers, **knowledge-graph** hops. Used when the answer is a *relationship* (“who reports to X”), not a similar paragraph.

### Long-context reorder

Not a retriever, but asked with them: models ignore the **middle** of a long prompt. After retrieve, put the **best chunks at the start and end**. `LongContextReorder` in LangChain.

---

## Comparison (revision)

| Retriever | Extra LLM? | Extra index? | Main win | Main cost |
| --- | --- | --- | --- | --- |
| Wikipedia | no | no (API) | no local data | network, not your docs |
| Vector store | no | embeddings | semantic RAG | chunk quality |
| MMR | no | same store | diverse top-k | may drop a good dupe |
| Multi-query | yes | same store | better recall on messy questions | N searches + rewrite |
| Compression | often yes | same store | shorter, cleaner context | latency / extract errors |
| BM25 | no | keyword | exact terms | no paraphrases |
| Ensemble / hybrid | no | both | production default | two indexes |
| Self-query | yes | store + schema | metadata filters from English | schema + LLM errors |
| Parent / multi-vector | no | two stores | small search, big context | more plumbing |
| HyDE | yes | same store | short query vs long docs | one generation |

---

## Tiny LCEL RAG (what you will write next)

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

retriever = store.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate.from_template(
    "Answer from the context only.\n\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
print(chain.invoke("who is a good bowler"))
```

`retriever | format_docs` is a **runnable**: question in, stuffed context string out. That is why retrievers exist as a type, not just `store.similarity_search`.

---

## Mental model

```text
                     ┌─ Wikipedia / Arxiv / Tavily     (external)
query ── retriever ──┼─ Vector store (± MMR, filters)  (your corpus)
                     └─ BM25                           (keywords)

wrappers:  Multi-query, HyDE, Self-query     → change the *query*
           Compression, Parent, Reorder      → change the *docs you keep*
           Ensemble                          → mix several retrievers
```

- **Retriever** = librarian API (`invoke` → `list[Document]`).
- **Vector store retriever** = librarian who uses *your* shelves (`chroma_db.py`).
- **Wikipedia** = librarian who phones a public library.
- **MMR** = “don’t hand me five copies of the same book.”
- **Multi-query** = “ask the question three ways.”
- **Compression** = “highlight only the relevant sentences.”
- **Hybrid** = what companies actually ship.

When you add scripts here, start with `store.as_retriever()`, then wrap MMR / multi-query / compression one at a time so you can see each effect on the same cricket index.
