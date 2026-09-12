"""
ingest.py (v2 — section-based chunking)
------------------------------------------
Improvement over the first version: instead of cutting text every 800
characters (which can slice a topic in half), this splits each file by
its markdown section headers (##), so each chunk is one complete,
coherent topic — e.g. the whole "What is a Fixed Deposit?" section
stays together as one chunk, instead of getting split awkwardly.

Long sections are still sub-split (with overlap) so no single chunk
gets too large for the model to handle well.

HOW TO USE:
1. Same folder structure as before (knowledge_base/ with your .md files).
2. Run: python3 ingest.py
   (This REPLACES your old database with the new, better-chunked one —
   you don't need to delete anything manually, the script handles it.)
"""

import os
import re
import chromadb

KNOWLEDGE_BASE_FOLDER = "knowledge_base"
DB_FOLDER = "chroma_db"
COLLECTION_NAME = "financial_literacy"

MAX_CHUNK_SIZE = 1200   # sections longer than this get sub-split
OVERLAP = 100


def split_into_sections(text):
    """
    Splits a markdown file into sections based on '## ' headers.
    The document title (the '# ' line at the top) gets prepended to
    every section, so each chunk still has context about which
    article it came from even in isolation.
    """
    lines = text.split("\n")
    doc_title = ""
    if lines and lines[0].startswith("# "):
        doc_title = lines[0].replace("# ", "").strip()

    # Split on lines that start with "## " (level-2 headers)
    section_pattern = re.compile(r"\n(?=## )")
    raw_sections = section_pattern.split(text)

    sections = []
    for section in raw_sections:
        section = section.strip()
        if not section or section.startswith("# ") and len(section) < 5:
            continue
        # Prepend the document title for context, unless this chunk IS the title block
        if doc_title and not section.startswith("# "):
            section = f"[From article: {doc_title}]\n\n{section}"
        sections.append(section)

    return sections


def sub_split_if_needed(section, max_size=MAX_CHUNK_SIZE, overlap=OVERLAP):
    """If a section is too long, break it into smaller overlapping pieces."""
    if len(section) <= max_size:
        return [section]

    pieces = []
    start = 0
    while start < len(section):
        end = start + max_size
        pieces.append(section[start:end].strip())
        start = end - overlap
    return [p for p in pieces if p]


def load_and_chunk_files(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            sections = split_into_sections(text)
            file_chunks = []
            for section in sections:
                file_chunks.extend(sub_split_if_needed(section))

            for chunk in file_chunks:
                all_chunks.append((chunk, filename))

            print(f"  {filename}: split into {len(file_chunks)} section-based chunks")
    return all_chunks


def main():
    print("Step 1: Reading and chunking your markdown files by section...")
    chunks_with_sources = load_and_chunk_files(KNOWLEDGE_BASE_FOLDER)
    print(f"Total chunks created: {len(chunks_with_sources)}\n")

    print("Step 2: Setting up the local Chroma database...")
    client = chromadb.PersistentClient(path=DB_FOLDER)

    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        print("  (found an old version of the database, replacing it with the improved chunking)")

    collection = client.create_collection(name=COLLECTION_NAME)

    print("Step 3: Embedding and storing chunks (this may take a moment)...")
    documents = [chunk for chunk, source in chunks_with_sources]
    metadatas = [{"source": source} for chunk, source in chunks_with_sources]
    ids = [f"chunk_{i}" for i in range(len(chunks_with_sources))]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print(f"\nDone! Stored {collection.count()} chunks in the '{COLLECTION_NAME}' collection.")
    print("Chunks are now section-based instead of character-based — should retrieve more complete answers.")


if __name__ == "__main__":
    main()
