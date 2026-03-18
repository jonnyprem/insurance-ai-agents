from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document

from app.schemas import Answer, SourceDocument


@dataclass(frozen=True)
class PromptTemplate:
    system_message: str = (
        "You are an internal insurance assistant. Answer only from the provided context. "
        "If the context is insufficient, say exactly: I don't know based on the provided documents."
    )

    def render(self, query: str, docs: Iterable[Document]) -> str:
        context = "\n\n".join(doc.page_content for doc in docs)
        return (
            f"{self.system_message}\n\n"
            f"Context:\n{context or 'No relevant context retrieved.'}\n\n"
            f"Question:\n{query}\n\n"
            "Return a concise answer for an internal insurance support user."
        )


def build_sources(docs: Iterable[Document]) -> list[SourceDocument]:
    return [
        SourceDocument(
            source=str(doc.metadata.get("source", "unknown")),
            excerpt=doc.page_content[:240],
        )
        for doc in docs
    ]


def generate_answer(query: str, retriever, llm, prompt_template: PromptTemplate | None = None) -> Answer:
    prompt_template = prompt_template or PromptTemplate()
    docs = retriever.invoke(query)
    response = llm.invoke(prompt_template.render(query, docs)).strip()

    if not docs or "I don't know" in response:
        return Answer(
            answer="I don't know based on the provided documents.",
            confidence="low",
            sources=build_sources(docs),
            validated=False,
        )

    return Answer(answer=response, confidence="high", sources=build_sources(docs), validated=True)
