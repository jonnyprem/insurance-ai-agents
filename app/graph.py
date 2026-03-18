from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.rag_pipeline import PromptTemplate, build_sources
from app.schemas import Answer


class GraphState(TypedDict, total=False):
    query: str
    retriever: object
    llm: object
    prompt_template: PromptTemplate
    context: list[Document]
    response: str
    final_answer: Answer
    validation_reason: str


def retrieve_agent(state: GraphState) -> GraphState:
    docs = state["retriever"].invoke(state["query"])
    return {"context": docs}


def generate_agent(state: GraphState) -> GraphState:
    prompt_template = state.get("prompt_template") or PromptTemplate()
    prompt = prompt_template.render(state["query"], state.get("context", []))
    response = state["llm"].invoke(prompt).strip()
    return {"response": response}


def validate_agent(state: GraphState) -> GraphState:
    docs = state.get("context", [])
    response = state.get("response", "").strip()

    if not docs:
        return {
            "final_answer": Answer(
                answer="I don't know based on the provided documents.",
                confidence="low",
                sources=[],
                validated=False,
            ),
            "validation_reason": "No supporting documents were retrieved.",
        }

    context_text = "\n\n".join(doc.page_content for doc in docs)
    validation_prompt = (
        "You are a validation agent. Determine whether the answer is grounded in the supplied context. "
        "Respond with only VALID or INVALID.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Answer:\n{response}"
    )
    verdict = state["llm"].invoke(validation_prompt).strip().upper()

    if "INVALID" in verdict or len(response) < 10 or "I DON'T KNOW" in response.upper():
        return {
            "final_answer": Answer(
                answer="Sorry, I cannot confidently answer that from the current documents.",
                confidence="low",
                sources=build_sources(docs),
                validated=False,
            ),
            "validation_reason": f"Validator verdict: {verdict or 'INVALID'}",
        }

    return {
        "final_answer": Answer(
            answer=response,
            confidence="high",
            sources=build_sources(docs),
            validated=True,
        ),
        "validation_reason": f"Validator verdict: {verdict or 'VALID'}",
    }


builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve_agent)
builder.add_node("generate", generate_agent)
builder.add_node("validate", validate_agent)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "validate")
builder.add_edge("validate", END)

graph = builder.compile()
