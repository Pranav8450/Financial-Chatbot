"""
query.py
----------
This script lets you type a question and see which chunks from your
knowledge base Chroma thinks are most relevant — this is the "retrieval"
part of RAG (the "R"). It does NOT call an LLM yet; it just proves that
search is working correctly before we connect it to an LLM.

HOW TO USE:
1. Run ingest.py first (only needs to be done once, or after you change
   your markdown files).
2. Make sure your virtual environment is active ("(venv)" showing).
3. Run this script:
       python3 query.py
4. Type a question when prompted, e.g. "what is a SIP" or "how is FD taxed"
5. Type "quit" to exit.
"""

import chromadb

DB_FOLDER = "chroma_db"
COLLECTION_NAME = "financial_literacy"


def main():
    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Loaded database with {collection.count()} chunks.")
    print("Type a question to search (or 'quit' to exit).\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        # Ask Chroma for the 3 most relevant chunks to this question.
        results = collection.query(
            query_texts=[question],
            n_results=3,
        )

        print("\n--- Top matching chunks ---")
        documents = results["documents"][0]
        sources = results["metadatas"][0]
        for i, (doc, meta) in enumerate(zip(documents, sources), start=1):
            print(f"\n[{i}] from {meta['source']}:")
            print(doc[:300] + ("..." if len(doc) > 300 else ""))

        print("\n(In the full chatbot, these chunks would be sent to the LLM")
        print("along with your question, so it can write a full answer")
        print("grounded in this exact content.)\n")


if __name__ == "__main__":
    main()
