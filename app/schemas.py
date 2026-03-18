from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User question about a policy or claim")


class SourceDocument(BaseModel):
    source: str
    excerpt: str


class Answer(BaseModel):
    answer: str
    confidence: Literal["high", "low"] = "high"
    sources: List[SourceDocument] = Field(default_factory=list)
    validated: bool = True
