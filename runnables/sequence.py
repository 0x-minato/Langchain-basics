from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

prompt = PromptTemplate(
    template='write a joke on topic {topic}',
    input_variables=['topic']
)

prompt1 = PromptTemplate(
    template='Explain the joke {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt1, model, parser)

res = chain.invoke({'topic', 'Deep Learning'})
print(res)