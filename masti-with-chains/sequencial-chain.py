from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model1 = ChatCohere(
    model="command-a-03-2025", 
    temperature=0.8
)
model2 = ChatCohere(
    model="command-a-03-2025", 
    temperature=0.3
)

template1 = PromptTemplate(
    template='give an interesting fact about this topic {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='You are a expert Rate this interesting fact {topic2} and add this topic in your answer',
    input_variables=['topic2']
)

parser = StrOutputParser()

chain = template1 | model1 | parser | template2 | model2 | parser

config = {
    'run_name': 'sequential_chain',
    'tags': ['llm_app', 'report_generation', 'summarization'],
    'metadata': {'model1': 'command-a-03-2025', 'model1temp': 0.8}
}

user_input = input('topic:')
res = chain.invoke({'topic': user_input}, config=config)
print(res)