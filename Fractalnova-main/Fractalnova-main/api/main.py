"""FractalNova API · applicazione FastAPI.

Endpoint principali:
  GET  /health                 stato del servizio
  POST /auth/register          crea utente
  POST /auth/token             login OAuth2 -> JWT
  GET  /v1/me                  profilo utente
  POST /v1/books               avvia un job di generazione (async) -> JobOut
  GET  /v1/jobs/{job_id}       stato del job
  GET  /v1/books/{book_id}     libro generato + formati di export
  POST /v1/chat/stream         chat streaming SSE (FractalNovaInference)
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import services
from .config import settings
from .db import Book, Job, User, get_db, init_db
from .schemas import BookOut, GenerateRequest, JobOut, Token, UserCreate, UserOut
from .security import create_access_token, get_current_user, hash_password, verify_password


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="API enterprise per la generazione e pubblicazione di libri (FractalNova).",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting in-memory (per utente). In produzione: Redis (vedi api/README.md).
_buckets: dict[int, deque] = defaultdict(lambda: deque(maxlen=10_000))


def rate_limit(user: User = Depends(get_current_user)) -> User:
    now = time.time()
    q = _buckets[user.id]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= settings.rate_limit_per_min:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit superato")
    q.append(now)
    return user


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "service": settings.app_name, "version": settings.version}


@app.post("/auth/register", response_model=UserOut, status_code=201, tags=["auth"])
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email gia' registrata")
    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserOut(id=user.id, email=user.email, plan=user.plan)


@app.post("/auth/token", response_model=Token, tags=["auth"])
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Email o password errati")
    return Token(access_token=create_access_token(str(user.id)))


@app.get("/v1/me", response_model=UserOut, tags=["users"])
def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, email=user.email, plan=user.plan)


@app.post("/v1/books", response_model=JobOut, status_code=202, tags=["books"])
def create_book(req: GenerateRequest, background: BackgroundTasks,
                user: User = Depends(rate_limit), db: Session = Depends(get_db)):
    job_id = str(uuid.uuid4())
    db.add(Job(id=job_id, user_id=user.id, status="pending"))
    db.commit()
    details = {
        "title": req.title or "Senza Titolo",
        "genre": req.genre,
        "plot": req.plot,
        "chapter_structure": req.structure,
        "target_pages": req.target_pages,
        "language": req.language,
    }
    background.add_task(services.process_job, job_id, user.id, details)
    return JobOut(id=job_id, status="pending")


@app.get("/v1/jobs/{job_id}", response_model=JobOut, tags=["books"])
def get_job(job_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Job non trovato")
    return JobOut(id=job.id, status=job.status, book_id=job.book_id, error=job.error or None)


@app.get("/v1/books/{book_id}", response_model=BookOut, tags=["books"])
def get_book(book_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    book = db.get(Book, book_id)
    if book is None or book.user_id != user.id:
        raise HTTPException(status_code=404, detail="Libro non trovato")
    structure = json.loads(book.content_json or "{}")
    return BookOut(
        id=book.id, title=book.title, genre=book.genre, status=book.status,
        seo=structure.get("seo", {}), formats=json.loads(book.formats_json or "{}"),
    )


@app.get("/v1/books/{book_id}/download/{fmt}", tags=["books"])
def download_book(book_id: int, fmt: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    book = db.get(Book, book_id)
    if book is None or book.user_id != user.id:
        raise HTTPException(status_code=404, detail="Libro non trovato")
    formats = json.loads(book.formats_json or "{}")
    path = formats.get(fmt)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Formato '{fmt}' non disponibile")
    return FileResponse(path, filename=os.path.basename(path))


# Dashboard web (SPA) servita da FastAPI, se presente: http://localhost:8000/app
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if os.path.isdir(_WEB_DIR):
    app.mount("/app", StaticFiles(directory=_WEB_DIR, html=True), name="web")
