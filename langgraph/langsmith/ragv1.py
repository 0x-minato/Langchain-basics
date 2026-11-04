
import os
import warnings
from dotenv import load_dotenv
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

os.environ['LANGCHAIN_PROJECT'] = 'RAG Chatbot'

load_dotenv()

HERE = os.path.dirname(__file__)
PDF_PATH = os.path.join(HERE, "islr.pdf")
INDEX_DIR = os.path.join(HERE, "faiss_islr")

# local MiniLM — Cohere trial keys die on a 600-page textbook (100k tokens/min, 100 calls/min)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.isdir(INDEX_DIR):
    vs = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
else:
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    print(f"Indexing {len(docs)} pages -> {len(splits)} chunks (one-time, local embeddings)...")
    vs = FAISS.from_documents(splits, emb)
    vs.save_local(INDEX_DIR)

retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | model | StrOutputParser()

print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
