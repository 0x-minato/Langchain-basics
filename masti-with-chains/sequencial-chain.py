from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model= 'gemini-3.8-flash'
)

template1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Generate a short summary on {topic2}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

user_input = input()
res = chain.invoke({'topic': user_input})
print(res)