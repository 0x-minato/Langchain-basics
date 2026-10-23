# Vector stores

This folder sits **after embeddings**. You already turn text into vectors in `embeddedModels/`. A vector store is where those vectors (plus the original chunks) **live**, so you can search them later without recomputing everything.

```text
loader → splitter → embed → vector store → retrieve top-k → LLM
```

`embeddedModels/embedding_model.py` did retrieval **by hand**: a Python list of vectors + `sklearn` cosine similarity + sort. That is a vector store with training wheels. This topic is the real thing.

---

## Why we need them

An embedding is just a list of floats. Useful search needs more than “I have a list in RAM right now.”

### 1. You cannot stuff the whole corpus into the LLM

Models have a context limit. RAG’s job is: **find the few chunks that match the question**, then send only those. The store is the index you query to get those chunks.

### 2. Brute-force lists do not scale

In `embedding_model.py` you compared the query to **every** document (`O(n)`). Fine for 20 cricket bios. Painful for 2 million PDF chunks.

Vector stores use **nearest-neighbour indexes** (HNSW, IVF, …) so search is closer to `O(log n)` — approximate, but fast.

### 3. You need the text back, not only the numbers

Similarity search returns **the original chunk** (and metadata: source file, page, heading). The LLM needs `page_content`, not a 384-dim array.

### 4. Persistence

If vectors only live in a Python list, they vanish when the script ends. You would re-embed the whole library on every run (slow, costs API money). A store can save to disk or a server and reload.

### 5. Updates and filters

Real apps add a new PDF, delete a page, or ask “only from `books/`, year > 2020.” A store (especially a vector **database**) supports add / delete / metadata filters. A numpy array does not.

**One line:** embeddings *create* meaning-vectors; a vector store *keeps and searches* them.

---

## What a vector store is

A **vector store** is a system that:

1. Accepts documents (text + optional metadata).
2. Calls an embedding model to get a vector per chunk.
3. Saves `(vector, text, metadata)` in an index.
4. On query: embed the question → find the **nearest** stored vectors → return those documents.

Distance metrics you will see:

| Metric | Idea | Typical use |
| --- | --- | --- |
| Cosine similarity | angle between vectors (direction) | sentence embeddings, like MiniLM |
| L2 / Euclidean | straight-line distance | some FAISS indexes |
| Inner product (dot) | similar to cosine if vectors are normalised | some ANN indexes |

“Nearest” = most similar under that metric. `k` (top-k) is how many chunks to return.

Two search styles:

| | Exact kNN | Approximate NN (ANN) |
| --- | --- | --- |
| What | compare to all vectors | jump through a graph / clusters |
| Speed | slow at scale | fast |
| Accuracy | perfect | slightly lossy, usually good enough |
| Example | your sklearn script | FAISS, Chroma HNSW, Pinecone |

Most production stores use ANN. You trade a tiny recall loss for huge speed.

What gets stored **besides** the vector:

- `page_content` — the chunk string
- `metadata` — source, page, headers from document splitters
- an **id** — so you can update or delete later

Without metadata, search still works, but you cannot cite “page 12 of cricket.pdf.”

---

## Vector store vs vector database

People (and LangChain) mix these words. The useful distinction:

| | Vector **store** | Vector **database** |
| --- | --- | --- |
| What it is | an **index** of embeddings + text | a **database product** built around that index |
| Runs as | library in your process (or a thin local folder) | a server / cloud service |
| Persistence | optional file on disk (FAISS `.faiss` + `.pkl`) | first-class (replication, backups) |
| Features | add + similarity search | plus auth, filters, hybrid search, multi-tenancy, scaling |
| Examples | FAISS, Annoy, in-memory Chroma | Pinecone, Qdrant, Weaviate, Milvus, pgvector, Chroma server |

**All vector databases can act as vector stores.**  
**Not every vector store is a database.** FAISS is “just” a similarity library: no users, no SQL, no HTTP API unless you wrap it.

LangChain’s class is named **`VectorStore`** for **both**. Chroma, FAISS, and Pinecone all implement the same interface. So in this repo, “vector store” = that interface. “Vector database” = the heavy, standalone product when you outgrow a local index.

Rule of thumb for your machine:

- Learning / one laptop / few PDFs → **FAISS** or **Chroma** (embedded).
- App with many users, lots of data, need filters and uptime → **Qdrant / Pinecone / pgvector**.

---

## Vector stores in LangChain

LangChain does **not** invent a new database. It wraps existing ones behind one API so you can swap FAISS for Pinecone later without rewriting the chain.

You always pass an **Embeddings** object (e.g. `HuggingFaceEmbeddings` with MiniLM). The store calls `embed_documents` on add and `embed_query` on search — same split you already practised.

### The interface (what every store implements)

```python
from langchain_core.vectorstores import VectorStore  # ABC; you use a subclass
```

| Method | Job |
| --- | --- |
| `from_documents(docs, embedding)` | embed + index a list of `Document`s (most common start) |
| `from_texts(texts, embedding)` | same, from raw strings |
| `add_documents(docs)` | append more chunks later |
| `similarity_search(query, k=4)` | embed query, return top-k `Document`s |
| `similarity_search_with_score(query, k=4)` | also return distance/similarity |
| `max_marginal_relevance_search(...)` | top-k that are relevant **and** not duplicates (MMR) |
| `as_retriever()` | wrap as a LangChain retriever for RAG chains |
| `delete(ids=...)` | remove points (if the backend supports it) |

`similarity_search` returns `Document`s, not vectors. That is what you stuff into the prompt.

### Typical LangChain RAG snippet

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = [
    Document(page_content="Bumrah is known for yorkers.", metadata={"source": "bio"}),
    Document(page_content="Kohli is known for chasing totals.", metadata={"source": "bio"}),
]

store = FAISS.from_documents(docs, embeddings)

hits = store.similarity_search("who is a good bowler", k=2)
print(hits[0].page_content)

retriever = store.as_retriever(search_kwargs={"k": 2})
# later: retriever | prompt | model | parser
```

Save / load FAISS (so you do not re-embed):

```python
store.save_local("vector_store/faiss_index")
store = FAISS.load_local("vector_store/faiss_index", embeddings, allow_dangerous_deserialization=True)
```

Chroma instead of FAISS (local folder that feels more like a tiny DB):

```python
store = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="vector_store/chroma_db",
)
```

### Stores you will meet in LangChain

| Integration | Package (typical) | Store or DB? | Good for |
| --- | --- | --- | --- |
| **FAISS** | `faiss-cpu` + `langchain-community` | store (library) | local, fast, homework |
| **Chroma** | `chromadb` + `langchain-chroma` | embedded DB | local persist, metadata filters |
| **Pinecone** | `langchain-pinecone` | hosted DB | production, managed |
| **Qdrant** | `langchain-qdrant` | DB (self-host or cloud) | filters, production |
| **pgvector** | Postgres extension | DB | if you already have Postgres |

Install when you start coding here, e.g. `pip install faiss-cpu chromadb`.

### How this compares to `embedding_model.py`

| Hand-rolled (that file) | LangChain vector store |
| --- | --- |
| `embed_documents` + numpy list | store holds vectors for you |
| `cosine_similarity` + `sort` | `similarity_search` |
| no files | `save_local` / `persist_directory` |
| top-1 only | `k=4` (or any k) |
| no metadata | `Document.metadata` |
| rewrite search if you switch tools | swap `FAISS` ↔ `Chroma` ↔ Pinecone |

Same maths. Different packaging.

---

## Mental model

```text
                    ┌─ vector (floats)
each chunk ──embed──┤
                    └─ stored with text + metadata

query ──embed──► nearest vectors in the index ──► top-k texts ──► LLM
```

- **Embeddings** = how you *measure* meaning.  
- **Vector store** = where you *put* those measurements and *look them up*.  
- **Vector database** = a store that is also a real database (server, scale, extra features).  
- **LangChain `VectorStore`** = one Python API over FAISS, Chroma, Pinecone, …

When you add scripts in this folder, start with **FAISS + MiniLM**, the same embedding as `embeddedModels/`. That is the smallest step from the sklearn list to a real index.
