"""CRUD repository for session persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.storage.models import Base, Session


class SessionRepository:
    """Handles persistence of architect sessions using SQLite."""

    def __init__(self, db_path: str = "data/architect.db"):
        """Initialize the repository with a database path.

        Args:
            db_path: Path to the SQLite database file.
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        logger.info(f"Database initialized at {db_path}")

    def create_session(
        self, title: str = "Untitled Project", raw_requirement: str = ""
    ) -> Session:
        """Create a new session record."""
        session = Session(
            title=title,
            raw_requirement=raw_requirement,
            status="NEW",
        )
        with self.SessionLocal() as db:
            db.add(session)
            db.commit()
            db.refresh(session)
        logger.info(f"Created session: {session.id}")
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        with self.SessionLocal() as db:
            return db.query(Session).filter(Session.id == session_id).first()

    def list_sessions(self, limit: int = 20) -> list[Session]:
        """List recent sessions."""
        with self.SessionLocal() as db:
            return (
                db.query(Session)
                .order_by(Session.updated_at.desc())
                .limit(limit)
                .all()
            )

    def save_state(self, session_id: str, state: dict[str, Any]) -> None:
        """Save the current workflow state to a session."""
        with self.SessionLocal() as db:
            session = db.query(Session).filter(Session.id == session_id).first()
            if session:
                session.state_json = json.dumps(state, default=str, ensure_ascii=False)
                session.completeness_score = state.get("completeness_score", 0.0)
                session.status = state.get("status", "IN_PROGRESS")
                session.updated_at = datetime.utcnow()
                db.commit()
                logger.info(f"Saved state for session {session_id}")
            else:
                logger.warning(f"Session {session_id} not found")

    def load_state(self, session_id: str) -> dict[str, Any] | None:
        """Load the workflow state from a session."""
        session = self.get_session(session_id)
        if session and session.state_json:
            logger.info(f"Loaded state for session {session_id}")
            return json.loads(session.state_json)
        return None
