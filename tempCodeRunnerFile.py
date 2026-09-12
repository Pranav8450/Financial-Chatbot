"""
main.py
--------
This is your chatbot as a real web server, using Ollama (running free,
locally on your own Mac) instead of a paid API — no API key needed.

IMPORTANT: this only works while Ollama is running on YOUR computer.
It's fine for local testing, but a live website needs an always-on
service (like Groq) since your laptop won't always be on. See the
note at the bottom of this file for switching back to Groq later.

HOW TO TEST LOCALLY:
1. Make sure Ollama is running: ollama run llama3.2 (in another terminal, leave it open)
2. pip3 install fastapi uvicorn ollama chromadb
3. Run: uvicorn main:app --reload
4. Open http://127.0.0.1:8000 in your browser
"""

from typing import List, Optional
import chromadb
import ollama
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DB_FOLDER = "chroma_db"
COLLECTION_NAME = "financial_literacy"
MODEL_NAME = "llama3.2"   # must match the model you already downloaded via Ollama
NUM_CHUNKS_TO_RETRIEVE = 4
MAX_HISTORY_MESSAGES = 10   # how many past turns to include, keeps requests small

SYSTEM_PROMPT = """You are a friendly financial literacy assistant for beginners in India.

Rules you must always follow:
- Answer ONLY using the context provided below each question. Do not use outside knowledge for specific facts, numbers, or tax rules.
- If the context doesn't contain the answer, say so honestly instead of guessing.
- Keep answers clear and simple, like explaining to someone with zero financial background.
- Stay strictly educational. Never recommend a specific stock, fund, insurer, or bank.
- Never tell the user what they personally should invest in or buy — explain concepts and let them decide.
- The user may refer back to earlier parts of this conversation — use that context to understand follow-up questions naturally.
- For casual greetings or small talk, respond naturally and warmly without needing context."""

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

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
    )

    answer = response["message"]["content"]
    return {"answer": answer, "sources": list(set(sources))}


# Serves the chat webpage (static/index.html) at the root URL
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ---------------------------------------------------------------------
# NOTE FOR LATER: when you're ready to deploy this as a real, always-on
# website, Ollama won't work anymore since it only runs on your own Mac.
# At that point, swap the "import ollama" + ollama.chat(...) call back to
# the Groq version (client = Groq(api_key=...), client.chat.completions.
# create(...)) — everything else in this file (retrieval, history,
# safety check, frontend) stays exactly the same.
# ---------------------------------------------------------------------