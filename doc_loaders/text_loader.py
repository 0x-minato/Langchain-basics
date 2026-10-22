# text loader is good for loading .txt files , scraping texts, code snippets, chat logs

from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_community.document_loaders import TextLoader

load_dotenv()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a summary on topic {topic}',
    input_variables=['topic']
)

loader = TextLoader('doc_loaders/cricket.txt', encoding='utf-8')
def loader_runnable(_): # _ since it takes param from invoke
    return loader.load()[0].page_content
docs = RunnableLambda(loader_runnable)

chain = docs | prompt1 | model | parser

res = chain.invoke({})
print(res)