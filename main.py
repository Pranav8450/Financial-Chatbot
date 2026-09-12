"""
main.py
--------
This is your chatbot as a real web server, using Groq's free API to run
a large open-source model (Llama 3.3 70B) — much bigger and more
reliable at following instructions than the small local models tested
on Ollama.

HOW TO TEST LOCALLY:
1. Make sure .env has a line: GROQ_API_KEY=your_key_here
2. pip3 install fastapi uvicorn groq chromadb python-dotenv
3. Run: uvicorn main:app --reload
4. Open http://127.0.0.1:8000 in your browser
"""

import os
from typing import List, Optional
import chromadb
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

DB_FOLDER = "chroma_db"
COLLECTION_NAME = "financial_literacy"
MODEL_NAME = "openai/gpt-oss-120b"   # large model, hosted free by Groq (replaces the retired llama-3.3-70b-versatile)
NUM_CHUNKS_TO_RETRIEVE = 10
MAX_HISTORY_MESSAGES = 10   # how many past turns to include, keeps requests small

SYSTEM_PROMPT = """You are a friendly financial literacy assistant for beginners in India.

Rules you must always follow:
- Answer ONLY using the context provided below each question. Do not use outside knowledge for specific facts, numbers, or tax rules.
- Use ONLY the exact terminology, numbers, and wording found in the provided context. Do not substitute your own general knowledge terms (for example, never say "dollar-cost averaging" — if the context says "rupee cost averaging," use that exact term, since this is for an Indian audience).
- If the context covers multiple aspects of a topic (e.g. how something works, its types, AND its taxation), cover all of them in your answer when the user asks for detail — do not skip sections that are present in the context.
- If the context doesn't contain the answer, say so honestly instead of guessing.
- Keep answers clear and simple, like explaining to someone with zero financial background.
- Stay strictly educational. Never recommend a specific stock, fund, insurer, or bank.
- Never tell the user what they personally should invest in or buy — explain concepts and let them decide.
- The user may refer back to earlier parts of this conversation — use that context to understand follow-up questions naturally.
- For casual greetings or small talk, respond naturally and warmly without needing context.

Formatting rules for every answer (applies consistently across all topics):
- Use a short ## heading for the main topic if the answer covers more than one section.
- Use ### sub-headings to break up distinct sections (e.g. "How it works", "Types", "Taxation") instead of one long paragraph block.
- Bold key terms and numbers the first time they appear (e.g. **NAV**, **₹1.5 lakh**).
- Use bullet points or numbered lists for steps, types, or multiple related facts — not dense paragraphs.
- Use a simple markdown table when comparing 2 or more items across the same set of attributes (e.g. comparing SIP types, comparing index fund vs flexi-cap fund, or comparing tax rates by holding period) — a table works just as well for comparing 2 things as it does for 3+. Do not force a table for content that isn't naturally comparative (a single concept explanation, a list of steps).
- Avoid a "checklist for whether this suits you" or "is this a good fit for you" framing — that edges toward personalized advice. Present facts and types neutrally; let the user draw their own conclusion without a structured self-assessment tool.
- End longer answers with a short 1-2 line plain-text summary, not a heading called "Bottom line" — keep it understated."""

# If the QUESTION itself contains any of these, skip the AI model entirely
# and return a fixed safe message — more reliable than trusting the model
# to decline on its own.
TIMING_TRIGGERS = [
    "should i invest", "should i buy", "should i sell", "right time to",
    "good time to", "market crash", "market is crashing", "market falling",
    "war is going", "because of the war", "because of war", "recession",
    "is it safe to invest", "invest now", "sell now", "buy now",
]

TIMING_FALLBACK = (
    "I can't suggest whether to invest, buy, or sell right now — that's a "
    "timing decision, and this bot is for financial education only, not "
    "investment advice. If you'd like, I can explain the general concept "
    "instead (e.g. what this type of investment is and how it works) — just "
    "ask without the 'right now' part."
)


def is_timing_question(question):
    lowered = question.lower()
    return any(trigger in lowered for trigger in TIMING_TRIGGERS)


# ---------- SETUP (runs once when the server starts) ----------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
db_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = db_client.get_collection(name=COLLECTION_NAME)

app = FastAPI()


class HistoryTurn(BaseModel):
    role: str        # "user" or "assistant"
    content: str


class Question(BaseModel):
    question: str
    history: Optional[List[HistoryTurn]] = []


def retrieve_chunks(question, n_results=NUM_CHUNKS_TO_RETRIEVE):
    results = collection.query(query_texts=[question], n_results=n_results)
    documents = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return documents, sources


def build_user_message(question, chunks):
    context_text = "\n\n---\n\n".join(chunks)
    return f"""Context from the knowledge base:

{context_text}

---

User's question: {question}"""


@app.post("/ask")
def ask(q: Question):
    if is_timing_question(q.question):
        return {"answer": TIMING_FALLBACK, "sources": []}

    chunks, sources = retrieve_chunks(q.question)
    user_message = build_user_message(q.question, chunks)

    # Build the full message list: system prompt + past conversation + new question
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent_history = (q.history or [])[-MAX_HISTORY_MESSAGES:]
    for turn in recent_history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": user_message})

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1500,
        messages=messages,
    )

    answer = response.choices[0].message.content
    return {"answer": answer, "sources": list(set(sources))}


# Serves the chat webpage (static/index.html) at the root URL
app.mount("/", StaticFiles(directory="static", html=True), name="static")