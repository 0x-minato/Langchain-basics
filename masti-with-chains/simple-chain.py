from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model= 'gemini-3.8-flash'
)

parser = StrOutputParser()

template = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic'],
)

chain = template | model | parser
res = chain.invoke({'topic': 'India'})

print(res)

chain.get_graph().print_ascii()