from langchain_google_gemini import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-flash-latest')

result = model.invoke('what is the capital of india')
print(result.content)