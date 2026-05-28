import os
import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Database setup — reads from DATABASE_URL in .env
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rag_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class ChatSession(Base):
    """Stores chat session metadata."""
    __tablename__ = "chat_sessions"

    id = Column(String(50), primary_key=True)
    title = Column(String(255), nullable=True)
    is_pinned = Column(Boolean, default=False, server_default="false")
    is_archived = Column(Boolean, default=False, server_default="false")
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    """Stores uploaded document metadata and status."""
    __tablename__ = "documents"

    id = Column(String(50), primary_key=True)
    session_id = Column(String(50), nullable=False, index=True)
    original_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    status = Column(String(50), default="uploaded")  # uploaded, processing, ready, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class QAHistory(Base):
    """Stores every question + answer pair with metadata."""
    __tablename__ = "qa_history"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(50), nullable=True, index=True) # added for session tracking
    document_name = Column(String(255), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables if they don't exist and run light migrations."""
    Base.metadata.create_all(bind=engine)
    
    # Handle schema migration for existing databases: Add columns
    db = SessionLocal()
    try:
        # qa_history migration
        if "postgresql" in str(engine.url):
            db.execute(text("ALTER TABLE qa_history ADD COLUMN IF NOT EXISTS session_id VARCHAR(50)"))
            db.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS is_pinned BOOLEAN DEFAULT FALSE"))
            db.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS is_archived BOOLEAN DEFAULT FALSE"))
            db.commit()
        else:
            # SQLite migrations
            try:
                db.execute(text("ALTER TABLE qa_history ADD COLUMN session_id VARCHAR(50)"))
                db.commit()
            except Exception:
                db.rollback()
                
            try:
                db.execute(text("ALTER TABLE chat_sessions ADD COLUMN is_pinned BOOLEAN DEFAULT FALSE"))
                db.commit()
            except Exception:
                db.rollback()
                
            try:
                db.execute(text("ALTER TABLE chat_sessions ADD COLUMN is_archived BOOLEAN DEFAULT FALSE"))
                db.commit()
            except Exception:
                db.rollback()
    except Exception as e:
        print(f"Database migration notice: {e}")
        db.rollback()
    finally:
        db.close()


# --- ChatSession Helpers ---

def get_or_create_session(session_id: str, title: str = None) -> ChatSession:
    """Gets an existing session or creates a new one."""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            session = ChatSession(id=session_id, title=title or "New Chat")
            db.add(session)
            db.commit()
            db.refresh(session)
        return session
    finally:
        db.close()


def list_sessions(include_archived: bool = False, limit: int = 50):
    """List all available chat sessions, with pinned ones first, optionally excluding archived."""
    db = SessionLocal()
    try:
        query = db.query(ChatSession)
        if not include_archived:
            query = query.filter(ChatSession.is_archived == False)
        return query.order_by(ChatSession.is_pinned.desc(), ChatSession.created_at.desc()).limit(limit).all()
    finally:
        db.close()


# --- Document Helpers ---

def save_document(document_id: str, session_id: str, original_name: str, file_path: str, status: str = "uploaded") -> Document:
    """Save a new document's metadata to the database."""
    db = SessionLocal()
    try:
        doc = Document(
            id=document_id,
            session_id=session_id,
            original_name=original_name,
            file_path=file_path,
            status=status
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc
    finally:
        db.close()


def update_document_status(document_id: str, status: str, error_message: str = None) -> Optional[Document]:
    """Update a document's processing status."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = status
            doc.error_message = error_message
            db.commit()
            db.refresh(doc)
        return doc
    finally:
        db.close()


def get_document(document_id: str) -> Optional[Document]:
    """Retrieve details of a specific document."""
    db = SessionLocal()
    try:
        return db.query(Document).filter(Document.id == document_id).first()
    finally:
        db.close()


def get_session_documents(session_id: str):
    """Retrieve all documents associated with a chat session."""
    db = SessionLocal()
    try:
        return db.query(Document).filter(Document.session_id == session_id).order_by(Document.created_at.asc()).all()
    finally:
        db.close()


def check_duplicate_document(session_id: str, original_name: str) -> Optional[Document]:
    """Check if a file with the same name is already in the session (and not failed)."""
    db = SessionLocal()
    try:
        return db.query(Document).filter(
            Document.session_id == session_id,
            Document.original_name == original_name,
            Document.status != "failed"
        ).first()
    finally:
        db.close()


def delete_document_record(document_id: str) -> bool:
    """Delete a document's metadata record."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            db.delete(doc)
            db.commit()
            return True
        return False
    finally:
        db.close()


# --- QAHistory Helpers (Updated to support session_id) ---

def save_qa(document_name: str, question: str, answer: str, session_id: str = None):
    """Save a Q&A pair with session association to the database."""
    db = SessionLocal()
    try:
        record = QAHistory(
            session_id=session_id,
            document_name=document_name,
            question=question,
            answer=answer,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


def get_history(document_name: str = None, session_id: str = None, limit: int = 50):
    """Retrieve Q&A history, optionally filtered by document name or session ID."""
    db = SessionLocal()
    try:
        query = db.query(QAHistory).order_by(QAHistory.created_at.desc())
        if session_id:
            query = query.filter(QAHistory.session_id == session_id)
        elif document_name:
            query = query.filter(QAHistory.document_name == document_name)
        return query.limit(limit).all()
    finally:
        db.close()


def delete_history(document_name: str = None, session_id: str = None):
    """Delete history, optionally filtered by document name or session ID."""
    db = SessionLocal()
    try:
        query = db.query(QAHistory)
        if session_id:
            query = query.filter(QAHistory.session_id == session_id)
        elif document_name:
            query = query.filter(QAHistory.document_name == document_name)
        query.delete()
        db.commit()
    finally:
        db.close()


def update_session_title(session_id: str, title: str) -> Optional[ChatSession]:
    """Update the title of a specific chat session."""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.title = title
            db.commit()
            db.refresh(session)
        return session
    finally:
        db.close()


def pin_session(session_id: str, is_pinned: bool) -> Optional[ChatSession]:
    """Toggle pin status of a specific chat session."""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.is_pinned = is_pinned
            db.commit()
            db.refresh(session)
        return session
    finally:
        db.close()


def archive_session(session_id: str, is_archived: bool) -> Optional[ChatSession]:
    """Toggle archive status of a specific chat session."""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.is_archived = is_archived
            db.commit()
            db.refresh(session)
        return session
    finally:
        db.close()
