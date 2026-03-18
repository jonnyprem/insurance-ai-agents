from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so imports like `from app...` work
# when running this script directly (e.g. `python scripts/ingest.py`).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.retriever import get_embeddings

DATA_DIR = Path("data")
DB_DIR = os.getenv("CHROMA_DIR", "./db")


def load_documents():
    documents = []
    faq_path = DATA_DIR / "faq.txt"
    if faq_path.exists():
        documents.extend(TextLoader(str(faq_path), encoding="utf-8").load())

    for pdf_path in DATA_DIR.glob("*.pdf"):
        documents.extend(PyPDFLoader(str(pdf_path)).load())

    if not documents:
        raise FileNotFoundError("No source documents found in data/.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def main() -> None:
    docs = load_documents()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=DB_DIR,
    )
    print(f"Indexed {len(docs)} chunks into {DB_DIR}")
    print(vectorstore._collection.count())


if __name__ == "__main__":
    main()
