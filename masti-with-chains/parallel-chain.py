import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import BitsAndBytesConfig

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation",
    model_kwargs={
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        "device_map": "auto",
    },
    pipeline_kwargs=dict(
        temperature=0.8,
        do_sample=True,
        max_new_tokens=500,
        return_full_text=False
    ),
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template='Generate study notes for {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Generate Quiz with 5 questions for {topic}',
    input_variables=['topic']
)

template3 = PromptTemplate(
    template='Merge {notes} and {quiz} and generate a proper report',
    input_variables=['notes', 'quiz']
)

parallel_chain = RunnableParallel({
    'notes': template1 | model1 | parser,
    'quiz': template2 | model2 | parser
})

merge_chain = template3 | model1 | parser

chain = parallel_chain | merge_chain

res = chain.invoke({'topic': 'India'})
print(res)

# chain.get_graph().print_ascii()