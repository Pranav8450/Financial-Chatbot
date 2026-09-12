"""
generate_ollama.py
--------------------
Same as generate.py, but uses Ollama (running free, locally on your own
Mac) instead of Claude or Groq. No API key needed, no internet needed
once the model is downloaded — but only works while Ollama is running
on YOUR computer.

HOW TO USE:
1. Make sure Ollama is running (open the Ollama app, or run
   "ollama run llama3.2" in another terminal and leave it open).
2. pip3 install ollama
3. Run: python3 generate_ollama.py
4. Type a question. Type "quit" to exit.
"""

import chromadb
import ollama

DB_FOLDER = "chroma_db"
COLLECTION_NAME = "financial_literacy"
MODEL_NAME = "qwen2.5:7b"   # must match the model you already downloaded
NUM_CHUNKS_TO_RETRIEVE = 6

SYSTEM_PROMPT = """You are a friendly financial literacy assistant for beginners in India.

Rules you must always follow:
- Answer ONLY using the context provided below each question. Do not use outside knowledge for specific facts, numbers, or tax rules.
- Use ONLY the exact terminology, numbers, and wording found in the provided context. Do not substitute your own general knowledge terms (for example, never say "dollar-cost averaging" — if the context says "rupee cost averaging," use that exact term, since this is for an Indian audience).
- If the context covers multiple aspects of a topic (e.g. how something works, its types, AND its taxation), cover all of them in your answer when the user asks for detail — do not skip sections that are present in the context.
- If the context doesn't contain the answer, say so honestly instead of guessing.
- Keep answers clear and simple, like explaining to someone with zero financial background.
- Stay strictly educational. Never recommend a specific stock, fund, insurer, or bank.
- Never tell the user what they personally should invest in or buy — explain concepts and let them decide.
- If a question is framed around timing or current events ("is X good to invest in right now because of [war/crisis/news event]"), do NOT answer the timing question directly. Instead, explicitly say you can't advise on investment timing or current events, then pivot to explaining the general, timeless concept only (e.g., what role that asset class typically plays in a portfolio). Do not use phrases like "can be a good option right now" or "during times like this" — these imply timing advice even with caveats attached.
- For casual greetings or small talk, respond naturally and warmly without needing context."""


# If the user's QUESTION itself contains any of these patterns, we skip
# the AI model entirely for that question and show a fixed safe message.
# This is more reliable than scanning the answer afterward, since it
# doesn't depend on the model's wording at all.
TIMING_TRIGGERS = [
    "should i invest",
    "should i buy",
    "should i sell",
    "right time to",
    "good time to",
    "right now",
    "market crash",
    "market is crashing",
    "market falling",
    "war is going",
    "because of the war",
    "because of war",
    "recession",
    "is it safe to invest",
    "invest now",
    "sell now",
    "buy now",
]

TIMING_CLARIFY = (
    "I can't tell you whether to buy, sell, or hold right now — that's a "
    "timing decision this bot doesn't make. But I can explain the general, "
    "timeless characteristics of whatever you're asking about. Which is it: "
    "large-cap stocks, mid-cap stocks, small-cap stocks, or mutual funds?"
)

CATEGORY_KEYWORDS = {
    "large cap": "large-cap stocks",
    "large-cap": "large-cap stocks",
    "mid cap": "mid-cap stocks",
    "mid-cap": "mid-cap stocks",
    "small cap": "small-cap stocks",
    "small-cap": "small-cap stocks",
    "mutual fund": "mutual funds",
}

CATEGORY_SYSTEM_PROMPT = """You are a financial literacy assistant. The user
previously asked a timing-based question (e.g. "should I sell now") and has
now specified a category. Explain ONLY the general, timeless characteristics
of that category (using the context provided) — typical risk level,
volatility, and what kind of investor it tends to suit. Do NOT say whether
now is a good or bad time to buy/sell/hold, and do NOT reference any current
event, crash, or war. Stay factual and general."""

DISCLAIMER = (
    "\n\n---\nThis is general educational information only, not advice on "
    "whether to buy, sell, or hold right now. For a timing decision, please "
    "consult a financial advisor."
)


def is_timing_question(question):
    """Checks if the question itself is asking for timing/event-based advice."""
    lowered = question.lower()
    return any(trigger in lowered for trigger in TIMING_TRIGGERS)


def match_category(question):
    """Checks if the question names one of the known categories."""
    lowered = question.lower()
    for keyword, category in CATEGORY_KEYWORDS.items():
        if keyword in lowered:
            return category
    return None


def retrieve_chunks(collection, question, n_results=NUM_CHUNKS_TO_RETRIEVE):
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


def main():
    db_client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = db_client.get_collection(name=COLLECTION_NAME)

    print(f"Loaded knowledge base with {collection.count()} chunks.")
    print("Ask a question (or 'quit' to exit).\n")

    awaiting_category = False   # tracks if we just asked "which category?"

    while True:
        question = input("You: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        # If we're waiting for a category answer from a previous timing question:
        if awaiting_category:
            category = match_category(question)
            if not category:
                print("\nBot: Please pick one: large-cap stocks, mid-cap stocks, small-cap stocks, or mutual funds.\n")
                continue

            chunks, sources = retrieve_chunks(collection, category)
            user_message = build_user_message(f"Explain the general characteristics of {category}.", chunks)

            print("\nThinking...\n")
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": CATEGORY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            answer = response["message"]["content"] + DISCLAIMER
            print(f"Bot: {answer}\n")
            print(f"(sources used: {', '.join(set(sources))})\n")
            awaiting_category = False
            continue

        # Check the QUESTION itself first — if it's asking for timing advice,
        # ask which category instead of answering the timing question.
        if is_timing_question(question):
            print(f"\nBot: {TIMING_CLARIFY}\n")
            awaiting_category = True
            continue

        chunks, sources = retrieve_chunks(collection, question)
        user_message = build_user_message(question, chunks)

        print("\nThinking... (this may take longer than a cloud API, since it's running on your Mac)\n")

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        answer = response["message"]["content"]
        print(f"Bot: {answer}\n")
        print(f"(sources used: {', '.join(set(sources))})\n")


if __name__ == "__main__":
    main()