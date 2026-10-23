"""Chroma vector store examples (LangChain).

Run from the repo root:
    python .\\vector_store\\chroma_db.py

Each rerun wipes the local collection so add / update / delete stay easy to follow.
Chroma distance: LOWER score = MORE similar.
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

PERSIST_DIR = Path("vector_store/chroma_db")
COLLECTION = "cricket_bios"


def heading(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def show_docs(docs: list[Document]) -> None:
    for doc in docs:
        print(f"- [{doc.id}] {doc.page_content}")
        print(f"    metadata: {doc.metadata}")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# fresh folder each run so this file is a clean demo, not a growing duplicate pile
if PERSIST_DIR.exists():
    shutil.rmtree(PERSIST_DIR)

store = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIR),
)

# ---------------------------------------------------------------------------
# 1. ADD DOCUMENTS
# ---------------------------------------------------------------------------
heading("1. ADD DOCUMENTS")

docs = [
    Document(
        page_content="Virat Kohli is an Indian batter known for chasing totals and consistency in ODIs.",
        metadata={"country": "India", "role": "batter", "format": "ODI"},
    ),
    Document(
        page_content="Sachin Tendulkar is a former Indian opener and the leading run-scorer in international cricket.",
        metadata={"country": "India", "role": "batter", "format": "Test"},
    ),
    Document(
        page_content="Jasprit Bumrah is an Indian fast bowler famous for yorkers and death-over bowling.",
        metadata={"country": "India", "role": "bowler", "format": "ODI"},
    ),
    Document(
        page_content="Pat Cummins is an Australian fast bowler and Test captain.",
        metadata={"country": "Australia", "role": "bowler", "format": "Test"},
    ),
    Document(
        page_content="Kane Williamson is a New Zealand batter known for calm technique and captaincy.",
        metadata={"country": "New Zealand", "role": "batter", "format": "Test"},
    ),
    Document(
        page_content="Rashid Khan is an Afghan leg-spinner famous in T20 leagues worldwide.",
        metadata={"country": "Afghanistan", "role": "bowler", "format": "T20"},
    ),
]

# stable ids make get / update / delete obvious (better than random uuids for learning)
ids = ["kohli", "sachin", "bumrah", "cummins", "williamson", "rashid"]

added_ids = store.add_documents(documents=docs, ids=ids)
print("stored ids:", added_ids)

# ---------------------------------------------------------------------------
# 2. GET DOCUMENTS
# ---------------------------------------------------------------------------
heading("2. GET DOCUMENTS")

print("\n-- all documents (store.get) --")
raw = store.get(include=["documents", "metadatas"])
for doc_id, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
    print(f"- [{doc_id}] {text} | {meta}")

print("\n-- by id (store.get_by_ids) --")
show_docs(store.get_by_ids(["kohli", "bumrah"]))

# ---------------------------------------------------------------------------
# 3. SIMILARITY SEARCH
# ---------------------------------------------------------------------------
heading("3. SIMILARITY SEARCH")

query = "who is a good bowler"
hits = store.similarity_search(query, k=3)
print(f"query: {query!r}")
show_docs(hits)

# ---------------------------------------------------------------------------
# 4. SIMILARITY SEARCH WITH SCORE
# ---------------------------------------------------------------------------
heading("4. SIMILARITY SEARCH WITH SCORE")
print("Chroma score is DISTANCE. Lower = closer / more similar.\n")

scored = store.similarity_search_with_score(query, k=3)
for doc, score in scored:
    print(f"distance={score:.4f}  [{doc.id}] {doc.page_content}")

print("\n-- same search as 0-1 relevance (higher = better) --")
relevant = store.similarity_search_with_relevance_scores(query, k=3)
for doc, rel in relevant:
    print(f"relevance={rel:.4f}  [{doc.id}] {doc.page_content}")

# ---------------------------------------------------------------------------
# 5. METADATA FILTERING
# ---------------------------------------------------------------------------
heading("5. METADATA FILTERING")

print("\n-- only Indian bowlers --")
india_bowlers = store.similarity_search(
    query,
    k=5,
    filter={"$and": [{"country": "India"}, {"role": "bowler"}]},
)
show_docs(india_bowlers)

print("\n-- only role=batter (no query ranking needed, still uses similarity) --")
batters = store.similarity_search(
    "who opens the batting",
    k=5,
    filter={"role": "batter"},
)
show_docs(batters)

print("\n-- get() with a metadata where-filter (not similarity, just lookup) --")
filtered_get = store.get(where={"country": "Australia"})
print(list(zip(filtered_get["ids"], filtered_get["documents"])))

# ---------------------------------------------------------------------------
# 6. UPDATE DOCUMENT
# ---------------------------------------------------------------------------
heading("6. UPDATE DOCUMENT")

print("before:", store.get_by_ids(["bumrah"])[0].page_content)

store.update_document(
    document_id="bumrah",
    document=Document(
        page_content="Jasprit Bumrah is India's spearhead, famous for toe-crushing yorkers and death bowling.",
        metadata={"country": "India", "role": "bowler", "format": "all-format"},
    ),
)

print("after: ", store.get_by_ids(["bumrah"])[0].page_content)
print("meta:  ", store.get_by_ids(["bumrah"])[0].metadata)

# ---------------------------------------------------------------------------
# 7. DELETE DOCUMENT
# ---------------------------------------------------------------------------
heading("7. DELETE DOCUMENT")

print("ids before delete:", store.get()["ids"])
store.delete(ids=["rashid"])
print("ids after delete: ", store.get()["ids"])

print("\nget_by_ids(['rashid']) after delete (should be empty):")
print(store.get_by_ids(["rashid"]))
