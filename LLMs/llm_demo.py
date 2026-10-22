from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model= 'gemini-flash-latest'
)

result = llm.invoke("What is the capital of India")
print(result)