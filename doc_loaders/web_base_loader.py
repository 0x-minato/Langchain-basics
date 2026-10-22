from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

url = 'https://www.troves.fi'
loader = WebBaseLoader(url)

def loader_fn():
    return loader.load()[0].page_content


chain = prompt1 | model | parser

res = chain.invoke({'question': 'what is the Average APY', 'text': loader_fn()})
print(res)