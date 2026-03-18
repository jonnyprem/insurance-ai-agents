from __future__ import annotations

from app.llm import get_llm
from app.rag_pipeline import generate_answer
from app.retriever import get_retriever

TEST_QUERIES = [
    "What does the standard health insurance plan cover?",
    "How long does claims processing take?",
    "Does the policy cover cosmetic surgery?",
    "What is the dental reimbursement cap?",
]


def main() -> None:
    retriever = get_retriever()
    llm = get_llm()
    for query in TEST_QUERIES:
        result = generate_answer(query=query, retriever=retriever, llm=llm)
        print(f"Q: {query}\nA: {result.answer}\nConfidence: {result.confidence}\nValidated: {result.validated}\n")


if __name__ == "__main__":
    main()
