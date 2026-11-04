# pip install -U langchain langchain-cohere langchain-community faiss-cpu pypdf python-dotenv langsmith

import os
import warnings
from dotenv import load_dotenv

from langsmith import traceable  # <-- key import

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# pypdf dumps font /Widths tables as warnings — those are not embeddings
warnings.filterwarnings("ignore", message="fontTools is required")

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "pdf_rag_demo"

HERE = os.path.dirname(__file__)
PDF_PATH = os.path.join(HERE, "islr.pdf")
INDEX_DIR = os.path.join(HERE, "faiss_islr")

# local MiniLM — API embedding a 600-page textbook hits trial rate limits
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _n_docs(docs):
    return {"n_docs": len(docs)}


def _vs_out(vs):
    return {"n_vectors": vs.index.ntotal}


# ---------- traced setup steps ----------
@traceable(
    name="load_pdf",
    tags=["setup", "loader", "pdf"],
    metadata={"loader": "PyPDFLoader", "file": "islr.pdf"},
    process_outputs=_n_docs,
)
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()  # list[Document]

@traceable(
    name="split_documents",
    tags=["setup", "splitter"],
    metadata={"splitter": "RecursiveCharacterTextSplitter", "chunk_size": 1000, "chunk_overlap": 150},
    process_inputs=lambda inputs: {
        "n_docs": len(inputs.get("docs", [])),
        "chunk_size": inputs.get("chunk_size"),
        "chunk_overlap": inputs.get("chunk_overlap"),
    },
    process_outputs=_n_docs,
)
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(
    name="build_vectorstore",
    tags=["setup", "index", "embed"],
    metadata={
        "vectorstore": "FAISS",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2",
        "persist_dir": "faiss_islr",
    },
    process_outputs=_vs_out,
)
def build_vectorstore(splits):
    vs = FAISS.from_documents(splits, emb)
    vs.save_local(INDEX_DIR)
    return vs

@traceable(
    name="load_vectorstore",
    tags=["setup", "index", "cache"],
    metadata={
        "vectorstore": "FAISS",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2",
        "persist_dir": "faiss_islr",
    },
    process_outputs=_vs_out,
)
def load_vectorstore():
    return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

@traceable(
    name="setup_pipeline",
    tags=["setup", "rag"],
    metadata={"pdf": "islr.pdf", "index_dir": "faiss_islr"},
    process_outputs=_vs_out,
)
def setup_pipeline(pdf_path: str):
    # always run these so LangSmith shows nested children, not one skip-span
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    if os.path.isdir(INDEX_DIR):
        return load_vectorstore()
    print(f"Indexing {len(docs)} pages -> {len(splits)} chunks (one-time, local embeddings)...")
    return build_vectorstore(splits)

# ---------- pipeline ----------
llm = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ---------- run a query (setup + answer in ONE parent trace) ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ").strip()

@traceable(
    name="pdf_rag_query",
    tags=["rag", "query"],
    metadata={
        "k": 4,
        "search_type": "similarity",
        "llm": "command-a-03-2025",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2",
    },
)
def run_query(question: str):
    vector_store = setup_pipeline(PDF_PATH)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    chain = parallel | prompt | llm | StrOutputParser()
    return chain.invoke(
        question,
        config={"tags": ["rag", "query"], "metadata": {"k": 4}},
    )

ans = run_query(q)
print("\nA:", ans)
