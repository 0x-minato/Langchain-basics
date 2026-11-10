from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_cohere import ChatCohere
from langchain_community.tools import DuckDuckGoSearchRun, YahooFinanceNewsTool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

os.environ['LANGCHAIN_PROJECT'] = 'Langgraph Chatbot'

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatCohere(
    model="command-a-03-2025"
)

# tools 
search_tool = DuckDuckGoSearchRun(region='us-en')
news_tool = YahooFinanceNewsTool()

tools = [search_tool, news_tool]
model_with_tools = model.bind_tools(tools)

def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    res = model_with_tools.invoke(messages)
    return {'messages': [res]}

tool_node = ToolNode(tools)

connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chat_bot = graph.compile(checkpointer = checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
