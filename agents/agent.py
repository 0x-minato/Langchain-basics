from langchain_core.tools import tool
from langchain_cohere import ChatCohere
import requests 
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool 
def get_whether_data(city: str,) -> str:
    """
    This function fetches whether data for a particular city
    """
    url = f'http://api.weatherapi.com/v1/current.json?key=1d961613160f4a01a8a145810263010&q={city}'
    response = requests.get(url)
    return response.json()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0,
)

# local copy of hwchase17/react — hub.pull of public prompts is blocked
# unless dangerously_pull_public_prompt=True
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
    llm=model,
    tools=[search_tool, get_whether_data],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_whether_data],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
)

response = agent_executor.invoke({"input": "find the capital of Antarctica and find its whether condition"})
print(response)