from langchain_core.tools import tool
from langchain_cohere import ChatCohere
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Multiply 2 numbers"""
    return a * b

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

model_tool = model.bind_tools([multiply])

query = HumanMessage('multiply 10 with 6')
messages = [query]
res = model_tool.invoke(messages)
messages.append(res)
print(res.tool_calls[0])

def extract_runnable(res):
    return res.tool_calls[0]

def append_message(message):
    messages.append(message)
    return messages

res1 = multiply.invoke(res.tool_calls[0])
print(res1)
messages.append(res1)

result = model_tool.invoke(messages)
print(result.content)

# chain = model_tool | append_message |  multiply
# result = chain.invoke('multiply 10 with 6')
# print(result)