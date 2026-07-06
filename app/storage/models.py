"""SQLAlchemy ORM models for session persistence."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Session(Base):
    """Represents a single career coach session."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    title: Mapped[str] = mapped_column(String(256), default="Untitled")
    status: Mapped[str] = mapped_column(String(32), default="NEW")
    raw_requirement: Mapped[str] = mapped_column(Text, default="")
    completeness_score: Mapped[float] = mapped_column(Float, default=0.0)
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, title={self.title}, status={self.status})>"
