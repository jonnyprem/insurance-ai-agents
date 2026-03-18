from __future__ import annotations

import os

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = os.getenv("CHROMA_DIR", "./db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_retriever(search_k: int = 3):
    db = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
    return db.as_retriever(search_kwargs={"k": search_k})
