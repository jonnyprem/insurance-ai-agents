# Insurance AI Assistant

A local-first insurance support assistant built with FastAPI, LangGraph, LangChain, Chroma, and Ollama. The app retrieves policy content from local documents, generates answers with an Ollama-hosted model, validates the answer in a second agent step, and exposes both an API and a Streamlit UI.

## Features

- Retrieval-augmented generation (RAG) over local insurance documents.
- Multi-agent LangGraph workflow with retriever, generator, and validator steps.
- Ollama-backed local inference using `mistral` or `llama3`.
- Pydantic response validation with grounded-answer fallback.
- Query/response logging for simple observability.
- Evaluation script for demo-ready testing.
- Streamlit front end for quick portfolio demos.

## Project Structure

```
insurance-ai-assistant/
├── app/
│   ├── graph.py
│   ├── llm.py
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   └── schemas.py
├── data/
│   └── faq.txt
├── scripts/
│   ├── evaluate.py
│   └── ingest.py
├── ui/
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Start Ollama and pull a model if needed:

   ```bash
   ollama run mistral
   ```

3. Ingest the sample knowledge base:

   ```bash
   python scripts/ingest.py
   ```

4. Run the FastAPI backend:

   ```bash
   uvicorn app.main:app --reload
   ```

5. In another terminal, run the Streamlit UI:

   ```bash
   streamlit run ui/app.py
   ```

## API Usage

```bash
curl "http://localhost:8000/ask?query=What%20is%20excluded%20from%20coverage%3F"
```

## Evaluation

```bash
python scripts/evaluate.py
```

## Interview Talking Points

- The assistant uses a multi-agent LangGraph state machine instead of a single prompt call.
- The validator agent introduces a lightweight groundedness check before returning answers.
- The system is local-first, so it can run in a privacy-sensitive insurance environment.
- The FastAPI + Streamlit split makes the project easy to demo and extend.
