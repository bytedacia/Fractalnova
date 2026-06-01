"""Persistenza (SQLAlchemy 2.0). SQLite di default, Postgres via DATABASE_URL."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import settings

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    plan: Mapped[str] = mapped_column(String(32), default="free")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    books: Mapped[list["Book"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class Book(Base):
    __tablename__ = "books"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    genre: Mapped[str] = mapped_column(String(64), default="fiction")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    content_json: Mapped[str] = mapped_column(Text, default="{}")   # book_structure serializzato
    formats_json: Mapped[str] = mapped_column(Text, default="{}")   # percorsi export
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    owner: Mapped[User] = relationship(back_populates="books")


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    book_id: Mapped[int | None] = mapped_column(ForeignKey("books.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|running|completed|failed
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
