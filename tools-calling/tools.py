from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, StructuredTool, BaseTool
from pydantic import BaseModel, Field
from typing import Type

# BUILD IN TOOL

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('todays geopolitics news')

# print(results)

# CUSTOM TOOL (USING @tool DECORATOR)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply 2 numbers"""
    return a*b

result = multiply.invoke({'a': 2, 'b': 3})
# print(result)

# print(multiply.name)
# print(multiply.args)
# print(multiply.description)

# # what the llm sees 
# print(multiply.args_schema.model_json_schema())

# CUSTOM TOOL (USING STRUCTURED TOOL)

class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to add")
    b: int = Field(description="The second number to add")

def multiply(a: int,b: int) -> int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name='multiply',
    description='multiply 2 numbers',
    args_schema=MultiplyInput
)

res = multiply_tool.invoke({'a': 2, 'b': 3})
# print(res)
# print(multiply_tool.name)
# print(multiply_tool.args)
# print(multiply_tool.description)

# # what the llm sees 
# print(multiply_tool.args_schema.model_json_schema())

# CUSTOM TOOL (BaseTool)

class MultiplyInput1(BaseModel):
    a: int = Field(description="The first number to add")
    b: int = Field(description="The second number to add")

class MultiplyTool(BaseTool): 
    name: str = 'multiply'
    description: str = 'multiply 2 nos'

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int: 
        return a*b

multiply_tool1 = MultiplyTool()

res = multiply_tool1.invoke({'a': 2, 'b': 3})
# print(res)
# print(multiply_tool1.name)
# print(multiply_tool1.args)
# print(multiply_tool1.description)

# what the llm sees 
# print(multiply_tool1.args_schema.model_json_schema())

# TOOLKIT

@tool
def multiply(a: int, b: int) -> int:
    """Multiply 2 numbers"""
    return a*b

@tool
def add(a: int, b: int) -> int:
    """add 2 numbers"""
    return a+b

class MathToolKit:
    def get_tools(self):
        return[add, multiply]

toolkit = MathToolKit()
tools = toolkit.get_tools()
res = tools[1].invoke({'a': 2, 'b': 5})
print(res)