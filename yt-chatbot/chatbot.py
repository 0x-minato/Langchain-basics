from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_api._errors import NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_cohere import CohereEmbeddings, ChatCohere
from dotenv import load_dotenv

load_dotenv()

video_id = "kHCamT4zfH4"
transcript_global = ''
try:
    ytt_api = YouTubeTranscriptApi()
    # this video has no English captions — only Hindi auto-generated
    transcript_list = ytt_api.fetch(video_id, languages=["en"])
    transcript_global = " ".join(chunk.text for chunk in transcript_list)
    # print(transcript_global)
except (TranscriptsDisabled, NoTranscriptFound):
    print("No captions available for this video")

# TEXT SPLITTING

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.create_documents([transcript_global])
# for i in range(len(chunks)):
#     print('CHUNK ---------------------------------------', i)
#     print(chunks[i])
#     print('CHUNK END')
# print(len(chunks))

# VECTOR STORE FAISS

embedding = CohereEmbeddings(
    model="embed-english-v3.0",
)
vector_store = FAISS.from_documents(chunks, embedding)
# print(vector_store.index_to_docstore_id)

# PROMPT TEMPLATE

prompt = PromptTemplate(
    template=""""
    You are a helpful assistant.
    Answer only from the provided transcript context
    If context is insufficient, just say you dont know
    
    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

# RETRIEVAL VECTOR STORE

question = 'how beautiful is rome'
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
# for doc in retriever_docs: 
#     print(doc)
#     print('new doc------------------------------------------------------------------------')

def get_context_text(input):
    return " ".join(doc.page_content for doc in input)

parser = StrOutputParser()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

parallel_chain = RunnableParallel({
    'question': RunnablePassthrough(),
    'context': retriever | RunnableLambda(get_context_text)
})

last_chain = prompt | model | parser

final_chain = parallel_chain | last_chain

answer = final_chain.invoke(question)
print(answer)