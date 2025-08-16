# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
import numpy as np
import pandas as pd
import random
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DataFrameLoader
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

# =========================
# Initialization
# =========================

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_methods=["*"],
    allow_headers=["*"]
)

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer('models/all-MiniLM-L6-v2"').to(device)
groq_api_key = os.getenv("GROQ_API_KEY")
chat_model = ChatGroq(model="mistral-saba-24b", api_key=groq_api_key)

# Load CSV and create questions
df = pd.read_csv('Software-Questions.csv', encoding='ISO-8859-1')
loader = DataFrameLoader(df, page_content_column="Question")
questions = loader.load()
categories = list(set(doc.metadata["Category"] for doc in questions))
questions_by_category = {category: [] for category in categories}
for doc in questions:
    category = doc.metadata["Category"]
    questions_by_category[category].append(doc)

# Load embeddings
question_embeddings = np.load('question_embeddings.npy')
answer_embeddings = np.load('answer_embeddings.npy')
category_embeddings = np.load('category_embeddings.npy')

# =========================
# Utility Functions
# =========================

def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def generate_response(user_input, correct_answer):
    messages = [
        SystemMessage(content="You are a helpful interview bot that evaluates user responses and gives them a score out of 5."),
        HumanMessage(content=f"User's answer: {user_input}\nExpected answer: {correct_answer}\nEvaluate correctness and give a response.")
    ]
    response = chat_model.invoke(messages)
    return response.content

def get_new_question(previous_question_idx, category_questions, asked_questions, question_embeddings):
    while True:
        new_idx = random.randint(0, len(category_questions) - 1)
        if new_idx == previous_question_idx or new_idx in asked_questions:
            continue
        if previous_question_idx is not None:
            similarity = cosine_similarity(
                question_embeddings[previous_question_idx], #aa
                question_embeddings[new_idx]
            )
            if similarity > 0.7:
                continue
        asked_questions.add(new_idx)
        return new_idx

# =========================
# Request Models
# =========================

class QuestionRequest(BaseModel):
    category: str
    previous_question_idx: int | None = None
    asked_questions: list[int] = []

class EvaluateRequest(BaseModel):
    user_input: str
    correct_answer: str
    question_idx: int

# =========================
# API Endpoints
# =========================

@app.get("/categories")
def get_categories():
    return {"categories": categories}

@app.post("/question")
def get_question(data: QuestionRequest):
    if data.category not in categories:
        raise HTTPException(status_code=400, detail="Invalid category")
    category_questions = questions_by_category[data.category]
    idx = get_new_question(data.previous_question_idx, category_questions, set(data.asked_questions), question_embeddings)
    question = category_questions[idx].page_content
    answer = category_questions[idx].metadata['Answer']
    return {
        "question_idx": idx,
        "question": question,
        "answer": answer,  # remove in production if needed
        "category": data.category
    }

@app.post("/evaluate")
def evaluate_answer(data: EvaluateRequest):
    user_response_embedding = embedding_model.encode(data.user_input)
    expected_answer_embedding = answer_embeddings[data.question_idx]
    similarity = cosine_similarity(user_response_embedding, expected_answer_embedding)
    feedback = generate_response(data.user_input, data.correct_answer)
    return {
        "similarity": similarity,
        "feedback": feedback
    }

# =========================
# Run locally
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
