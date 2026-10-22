# Embedded models (embeddings)

This folder turns **text into vectors** (lists of floats) so you can compare meaning, not keywords.

A sentence like `"who is a good bowler"` becomes a point in high-dimensional space. Texts about the same idea sit close together. Distance (here: **cosine similarity**) is the score.

This is the core of RAG later: embed chunks → store → embed the question → nearest chunks → send those to the LLM.

---

## `embedding_model.py` — similarity search by hand

### Setup

- `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`  
  Small, fast, local-enough model meant for **sentences**, not chat.
- `HF_TOKEN` is copied from `HUGGINGFACEHUB_API_TOKEN` in `.env` so Hugging Face will let you download the model.

### Two embed methods (do not mix them up)

| Method | Use on |
| --- | --- |
| `embed_documents(list[str])` | many chunks (your cricket bios) |
| `embed_query(str)` | the user question |

Some embedding models use slightly different prefixes for docs vs queries. Always embed the corpus with `embed_documents` and the question with `embed_query`.

### Scoring

```python
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]
```

- `cosine_similarity` wants 2D arrays, so the query is wrapped in `[query_embedding]`.
- `scores[i]` is how close document `i` is to the query (`1` = identical direction, `0` = unrelated).
- `enumerate` keeps `(index, score)`. Sort by score, take `[-1]` = **best match**.

Then you print the query, the winning document, and the score.

Query in the file: `"who is a good bowler"`. The list is cricket bios, so the nearest row should be someone like Bumrah, Starc, Cummins — not Kohli.

---

## What you are practising

A tiny retrieval pipeline **without** a vector database:

1. Embed all documents once.
2. Embed the query.
3. Rank by cosine similarity.
4. Return the top document.

Later you will replace the Python list + sklearn with Chroma / FAISS, and you will **split** long files first (`text_splitting/`). Embeddings work best on short, focused chunks — that is why splitters exist.

---

## Run

```powershell
python .\embeddedModels\embedding_model.py
```

First run may download `all-MiniLM-L6-v2`. After that it should be cached.
