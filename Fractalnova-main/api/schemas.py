"""Schemi Pydantic (v2) per request/response dell'API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class UserCreate(BaseModel):
    email: str
    password: str = Field(min_length=8)


class UserOut(BaseModel):
    id: int
    email: str
    plan: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class GenerateRequest(BaseModel):
    title: Optional[str] = None
    genre: str = "fiction"
    plot: str = ""
    structure: str = Field(default="misto", pattern="^(breve|lungo|misto)$")
    target_pages: int = Field(default=100, ge=1, le=100000)
    language: str = "it"

    @field_validator("title", "plot")
    @classmethod
    def _sanitize(cls, v):
        # Difesa in profondita': rimuove caratteri di controllo e limita la lunghezza.
        if v is None:
            return v
        cleaned = "".join(ch for ch in v if ch in "\n\t" or ord(ch) >= 32)
        return cleaned[:100000]


class JobOut(BaseModel):
    id: str
    status: str
    book_id: Optional[int] = None
    error: Optional[str] = None


class BookOut(BaseModel):
    id: int
    title: str
    genre: str
    status: str
    seo: dict[str, Any] = {}
    formats: dict[str, Any] = {}


class ChatRequest(BaseModel):
    message: str = ""
    image: Optional[str] = None
    model: str = "auto"
    temperature: float = 0.8
    max_tokens: int = 2048
    language: str = "auto"
    conversation: list[dict] = []


class ChatStreamEvent(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None
    error: Optional[str] = None
    done: bool = False
