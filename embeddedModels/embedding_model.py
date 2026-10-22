from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os 

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian batter known for chasing totals and consistency in ODIs.",
    "Sachin Tendulkar is a former Indian opener and the leading run-scorer in international cricket.",
    "Jasprit Bumrah is an Indian fast bowler famous for yorkers and death-over bowling.",
    "Rohit Sharma is an Indian opener and former captain, known for double centuries in ODIs.",
    "MS Dhoni is a former Indian wicketkeeper-batter and World Cup winning captain.",
    "Rahul Dravid is a former Indian batter nicknamed The Wall for his defence.",
    "Ravindra Jadeja is an Indian all-rounder who bowls left-arm spin and fields brilliantly.",
    "KL Rahul is an Indian wicketkeeper-batter who opens in limited-overs cricket.",
    "Hardik Pandya is an Indian all-rounder who bowls medium pace and finishes innings.",
    "Shubman Gill is a young Indian batter and current ODI opener.",
    "AB de Villiers is a former South African batter famous for 360-degree striking.",
    "Kane Williamson is a New Zealand batter known for calm technique and captaincy.",
    "Steve Smith is an Australian batter with an unorthodox stance and high Test average.",
    "Joe Root is an England batter and former Test captain.",
    "Babar Azam is a Pakistani batter regarded as one of the best in ODIs.",
    "Pat Cummins is an Australian fast bowler and Test captain.",
    "Mitchell Starc is an Australian left-arm fast bowler known for inswinging yorkers.",
    "Shaheen Afridi is a Pakistani left-arm fast bowler who takes new-ball wickets.",
    "Rashid Khan is an Afghan leg-spinner famous in T20 leagues worldwide.",
    "Jasprit Bumrah and Mohammed Shami have formed India's pace attack with Bhuvneshwar Kumar.",
]

query = "who is a good bowler"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is", score)