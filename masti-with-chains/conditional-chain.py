import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

parser = StrOutputParser()

class FeedBack(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='give the sentiment of the feedback')

pydantic_parser = PydanticOutputParser(pydantic_object=FeedBack)

template = PromptTemplate(
    template='Generate the sentiment of the following feedback into positive or negative: {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': pydantic_parser.get_format_instructions()}
)

classifier_chain = template | model | pydantic_parser

positive_template = PromptTemplate(
    template='Write an appropriate response for a positive feedback: {feedback}',
    input_variables=['feedback'],
)
negative_template = PromptTemplate(
    template='Write an appropriate response for a negative feedback: {feedback}',
    input_variables=['feedback'],
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', positive_template | model | parser),
    (lambda x:x.sentiment == 'negative', negative_template | model | parser),
    RunnableLambda(lambda x: 'could not find anything')
)

final_chain = classifier_chain | branch_chain

res = final_chain.invoke({'feedback': 'this is a great phone, Im so happy'})
print(res)