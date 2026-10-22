import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model= 'gemini-3.8-flash'
) 

class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(gt=18, description="age of the person")
    city: str = Field(description="city the person belongs to")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=(
        "Generate a fictional {place} person.\n {instruction}"
    ),
    input_variables=["place"],
    partial_variables={'instruction': parser.get_format_instructions()}
)
print(template)

# prompt = template.format(place="spanish")
# res = model.invoke(prompt)
# print("RAW:\n", res.content)
# person = parser.parse(res.content)
# print("PARSED:\n", person)

chain = template | model | parser
res = chain.invoke({'place': 'spanish'})
print(res)