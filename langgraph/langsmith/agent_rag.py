from langchain_cohere import ChatCohere
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "agent_rag"

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'

  response = requests.get(url)

  return response.json()

llm = ChatCohere(
    model="command-a-03-2025",
    temperature=0,
)

# local copy of hwchase17/react — hub.pull of public prompts is blocked
prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

response = agent_executor.invoke(
    {"input": "What is the current temp of gurgaon"},
    config={
        "run_name": "agent_rag_query",
        "tags": ["agent", "react", "weather"],
        "metadata": {"city": "gurgaon", "tools": ["duckduckgo", "get_weather_data"]},
    },
)
print(response)

print(response['output'])
