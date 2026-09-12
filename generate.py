"""
generate.py
------------
This is the full RAG pipeline: it retrieves relevant chunks from your
Chroma database (same as query.py), then sends those chunks + your
question to Claude, and prints back a real written answer grounded in
your knowledge base — instead of raw chunks.

HOW TO USE:
1. Make sure you've already run ingest.py at least once.
2. Make sure you have a .env file in this folder with a line like:
       ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
3. Make sure (venv) is active in your terminal.
4. Run:
       python3 generate.py
5. Type a question. Type "quit" to exit.
"""

import os
import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv

# ---------- SETTINGS ----------
DB_FOLDER = "chroma_db"
COLLECTION_NAME = "financial_literacy"
MODEL_NAME = "claude-haiku-4-5-20251001"   # cheap, fast model — good for testing
NUM_CHUNKS_TO_RETRIEVE = 4                  # how many chunks to pull per question

# This is the bot's personality and rules — it gets sent with EVERY request.
SYSTEM_PROMPT = """You are a friendly financial literacy assistant for beginners in India.

Rules you must always follow:
- Answer ONLY using the context provided below each question. Do not use outside knowledge for specific facts, numbers, or tax rules.
- If the context doesn't contain the answer, say so honestly instead of guessing.
- Keep answers clear and simple, like explaining to someone with zero financial background.
- Stay strictly educational. Never recommend a specific stock, fund, insurer, or bank.
- Never tell the user what they personally should invest in or buy — explain concepts and let them decide.
- For casual greetings or small talk, respond naturally and warmly without needing context."""


def load_env_and_client():
    """Loads the API key from .env and sets up the Claude client."""
    load_dotenv()  # reads the .env file and loads its variables
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: No API key found. Make sure your .env file exists")
        print("and contains a line like: ANTHROPIC_API_KEY=sk-ant-...")
        exit(1)
    return Anthropic(api_key=api_key)


def retrieve_chunks(collection, question, n_results=NUM_CHUNKS_TO_RETRIEVE):
    """Same retrieval step as query.py — finds the most relevant chunks."""
    results = collection.query(query_texts=[question], n_results=n_results)
    documents = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return documents, sources


def build_user_message(question, chunks):
    """Combines the retrieved chunks and the question into one message for Claude."""
    context_text = "\n\n---\n\n".join(chunks)
    return f"""Context from the knowledge base:

{context_text}

---

User's question: {question}"""


def main():
    client = load_env_and_client()

    db_client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = db_client.get_collection(name=COLLECTION_NAME)

    print(f"Loaded knowledge base with {collection.count()} chunks.")
    print("Ask a question (or 'quit' to exit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        # Step 1: Retrieve relevant chunks (the "R" in RAG)
        chunks, sources = retrieve_chunks(collection, question)
        user_message = build_user_message(question, chunks)

        # Step 2: Send to Claude (the "G" in RAG - Generation)
        print("\nThinking...\n")
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        print(f"Bot: {answer}\n")
        print(f"(sources used: {', '.join(set(sources))})\n")


if __name__ == "__main__":
    main()
