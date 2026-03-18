from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.graph import graph
from app.llm import get_llm
from app.rag_pipeline import PromptTemplate
from app.retriever import get_retriever
from app.schemas import Answer

LOG_PATH = Path(os.getenv("QUERY_LOG_PATH", "logs/query_log.jsonl"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("insurance_ai_assistant")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

app = FastAPI(title="Insurance AI Assistant", version="1.0.0")

retriever = get_retriever()
llm = get_llm()
prompt_template = PromptTemplate()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ask", response_model=Answer)
def ask(query: str) -> Answer:
    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters long.")

    result = graph.invoke(
        {
            "query": query,
            "retriever": retriever,
            "llm": llm,
            "prompt_template": prompt_template,
        }
    )
    answer: Answer = result["final_answer"]
    logger.info(
        json.dumps(
            {
                "query": query,
                "answer": answer.answer,
                "confidence": answer.confidence,
                "validated": answer.validated,
            }
        )
    )
    return answer
