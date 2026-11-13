from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import asyncio
import os
import queue
import tempfile
import sys
import threading

os.environ['LANGCHAIN_PROJECT'] = 'Langgraph Chatbot'

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if not thread_id:
        return None
    return _THREAD_RETRIEVERS.get(str(thread_id))

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embedding)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
llm = ChatCohere(model="command-a-03-2025", disable_streaming="tool_calling")

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    """
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

MATH_SERVER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mcp_servers", "main.py")
)

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [MATH_SERVER],
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [search_tool, *mcp_tools, rag_tool]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def _align_tool_results(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Cohere requires every assistant tool_call_id to have a matching ToolMessage."""
    aligned: list[BaseMessage] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            aligned.append(msg)
            call_ids = [tc.get("id") for tc in msg.tool_calls if tc.get("id")]
            tool_msgs: list[ToolMessage] = []
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                tool_msgs.append(messages[j])
                j += 1
            if len(tool_msgs) == len(call_ids) and any(
                tm.tool_call_id not in call_ids for tm in tool_msgs
            ):
                tool_msgs = [
                    ToolMessage(content=tm.content, tool_call_id=call_id, name=tm.name)
                    for tm, call_id in zip(tool_msgs, call_ids)
                ]
            seen = {tm.tool_call_id for tm in tool_msgs}
            aligned.extend(tool_msgs)
            for tc in msg.tool_calls:
                call_id = tc.get("id")
                if call_id and call_id not in seen:
                    aligned.append(
                        ToolMessage(
                            content="Tool did not return a result.",
                            tool_call_id=call_id,
                            name=tc.get("name") or "tool",
                        )
                    )
            i = j
            continue
        aligned.append(msg)
        i += 1
    return aligned


async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = _align_tool_results(state["messages"])
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
chat_bot = chatbot

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())


def stream_messages(input, config):
    """Sync iterator over astream(messages) on the backend loop."""
    items = queue.Queue()

    async def _run():
        try:
            async for item in chatbot.astream(
                input, config=config, stream_mode="messages"
            ):
                items.put(item)
        except Exception as exc:
            items.put(exc)
        finally:
            items.put(None)

    future = _submit_async(_run())
    while True:
        item = items.get()
        if item is None:
            break
        if isinstance(item, Exception):
            future.result()
            raise item
        yield item
    future.result()

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})