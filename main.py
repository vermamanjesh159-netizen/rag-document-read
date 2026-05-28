import os
import sys

# 1. Set environment variables to suppress HuggingFace and Python warnings before imports
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import shutil
import logging
import warnings
import uuid
from typing import Optional, List
from pydantic import BaseModel

# 2. Suppress Python warnings completely in this process
warnings.simplefilter("ignore")
warnings.showwarning = lambda *args, **kwargs: None

# 3. Quiet noisy library logging
for logger_name in ["httpx", "huggingface_hub", "sentence_transformers", "urllib3", "sqlalchemy", "uvicorn"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# 4. Configure root logger
logging.basicConfig(level=logging.ERROR)

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Response, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import modules from our project
from ingest import process_document
from search_engine import DocumentSearcher
from database import (
    init_db, save_qa, get_history, delete_history,
    get_or_create_session, list_sessions, save_document,
    update_document_status, get_document, get_session_documents,
    check_duplicate_document, delete_document_record, update_session_title,
    pin_session, archive_session
)

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Our app logger
logger = logging.getLogger("rag-api")
logger.setLevel(logging.INFO)

# Directories
DOCUMENTS_DIR = "documents"
VECTOR_STORES_DIR = "vector_stores"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(VECTOR_STORES_DIR, exist_ok=True)

# Initialize database tables on startup
init_db()

# Load embeddings model once on startup to avoid reloading per request
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "placeholder_token")
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=hf_token
)


def generate_chat_title(question: str) -> str:
    """Generate a short 3-5 word title from the user's first question using Groq."""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return question[:30] + "..." if len(question) > 30 else question
            
        llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=20
        )
        prompt = f"Generate a short, concise, 3-5 word title for a chat conversation that starts with this question: '{question}'. Return ONLY the title itself, with no quotation marks, no punctuation, and no extra text."
        response = llm.invoke(prompt)
        title = response.content.strip().strip('"').strip("'")
        # Keep title within a reasonable length
        if len(title) > 40:
            title = title[:37] + "..."
        return title
    except Exception as e:
        logger.error(f"Error generating chat title: {e}")
        # Fallback to a truncated version of the question
        return question[:30] + "..." if len(question) > 30 else question

# Cache for active session searchers in memory: session_id -> { "searcher": DocumentSearcher, "document_ids": List[str] }
active_searchers = {}

app = FastAPI(
    title="RAG Document Query API",
    description="FastAPI Backend for Uploading Documents and Querying them using LangChain, FAISS and Groq",
    version="1.1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    session_id: str
    question: str


class SessionCreateRequest(BaseModel):
    session_id: str
    title: Optional[str] = None


async def process_document_task(document_id: str, file_path: str):
    """Processes document in the background to build FAISS index."""
    try:
        update_document_status(document_id, "processing")
        logger.info(f"Background task: Started processing document {document_id}")
        
        # Process and create vectorstore
        vectorstore = process_document(file_path)
        if not vectorstore:
            raise Exception("Document loader or splitter returned empty chunks.")
            
        # Save vectorstore to disk under its unique document_id
        index_path = os.path.join(VECTOR_STORES_DIR, document_id)
        vectorstore.save_local(index_path)
        
        # Mark as ready
        update_document_status(document_id, "ready")
        logger.info(f"Background task: Document {document_id} processed successfully.")
        
    except Exception as e:
        logger.error(f"Background task error processing document {document_id}: {e}")
        update_document_status(document_id, "failed", str(e))


def get_or_load_session_searcher(session_id: str) -> DocumentSearcher:
    """Gets a merged DocumentSearcher for all ready documents in a session."""
    # 1. Get all ready documents for this session
    docs = get_session_documents(session_id)
    ready_docs = [d for d in docs if d.status == "ready"]
    
    if not ready_docs:
        raise HTTPException(
            status_code=400,
            detail="No processed documents found in this chat session. Please upload a document first."
        )
        
    ready_doc_ids = sorted([d.id for d in ready_docs])
    
    # 2. Check cache
    cached = active_searchers.get(session_id)
    if cached and cached.get("document_ids") == ready_doc_ids:
        return cached["searcher"]
        
    # 3. Load and merge vectorstores
    try:
        merged_vectorstore = None
        for doc in ready_docs:
            index_path = os.path.join(VECTOR_STORES_DIR, doc.id)
            if not os.path.exists(index_path):
                logger.warning(f"Index not found on disk for document {doc.id} ({doc.original_name})")
                continue
                
            logger.info(f"Loading vector store for {doc.original_name} from {index_path}...")
            vectorstore = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            if merged_vectorstore is None:
                merged_vectorstore = vectorstore
            else:
                merged_vectorstore.merge_from(vectorstore)
                
        if merged_vectorstore is None:
            raise HTTPException(
                status_code=404,
                detail="None of the document vector stores could be loaded from disk."
            )
            
        # 4. Create searcher and cache it
        searcher = DocumentSearcher(merged_vectorstore)
        active_searchers[session_id] = {
            "searcher": searcher,
            "document_ids": ready_doc_ids
        }
        return searcher
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging vector stores for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load search index for conversation documents: {str(e)}"
        )


@app.get("/", response_class=HTMLResponse)
def get_index():
    """Serves the front-end playground."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found.")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serves the favicon image."""
    return FileResponse("favicon.png")


@app.get("/pin.png", include_in_schema=False)
def pin_icon():
    """Serves the pin icon."""
    return FileResponse("pin.png")


# --- API Endpoints ---

@app.post("/api/sessions")
def create_session(request: SessionCreateRequest):
    """Register or restore a chat session."""
    try:
        session = get_or_create_session(request.session_id, request.title)
        return {
            "session_id": session.id,
            "title": session.title,
            "created_at": session.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TogglePinRequest(BaseModel):
    is_pinned: bool


class ToggleArchiveRequest(BaseModel):
    is_archived: bool


@app.get("/api/sessions")
def get_sessions_endpoint(include_archived: bool = False, limit: int = 50):
    """Retrieve list of all chat sessions."""
    try:
        sessions = list_sessions(include_archived=include_archived, limit=limit)
        return {
            "sessions": [
                {
                    "id": s.id,
                    "title": s.title,
                    "is_pinned": s.is_pinned,
                    "is_archived": s.is_archived,
                    "created_at": s.created_at.isoformat()
                }
                for s in sessions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/sessions/{session_id}/pin")
def toggle_pin_session(session_id: str, request: TogglePinRequest):
    """Pin or unpin a chat session."""
    try:
        session = pin_session(session_id, request.is_pinned)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "success", "session_id": session_id, "is_pinned": session.is_pinned}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/sessions/{session_id}/archive")
def toggle_archive_session(session_id: str, request: ToggleArchiveRequest):
    """Archive or unarchive a chat session."""
    try:
        session = archive_session(session_id, request.is_archived)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "success", "session_id": session_id, "is_archived": session.is_archived}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}")
def delete_session_endpoint(session_id: str):
    """Delete a chat session and all its associated documents and history."""
    try:
        # 1. Delete all documents for this session
        docs = get_session_documents(session_id)
        for doc in docs:
            # Delete physical source file
            if os.path.exists(doc.file_path):
                os.remove(doc.file_path)

            # Delete FAISS vector index folder
            index_path = os.path.join(VECTOR_STORES_DIR, doc.id)
            if os.path.exists(index_path):
                shutil.rmtree(index_path)

            # Remove from DB
            delete_document_record(doc.id)

        # 2. Delete history
        delete_history(session_id=session_id)

        # 3. Delete session record from DB
        from database import SessionLocal, ChatSession
        db = SessionLocal()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                db.delete(session)
                db.commit()
        finally:
            db.close()

        # 4. Evict searcher cache
        if session_id in active_searchers:
            del active_searchers[session_id]

        return {"status": "success", "message": "Session and all associated data deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/documents")
def get_documents_endpoint(session_id: str):
    """Retrieve all documents associated with a session."""
    try:
        docs = get_session_documents(session_id)
        return {
            "documents": [
                {
                    "id": d.id,
                    "session_id": d.session_id,
                    "original_name": d.original_name,
                    "status": d.status,
                    "error_message": d.error_message,
                    "created_at": d.created_at.isoformat()
                }
                for d in docs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload a PDF or TXT file, link to a session, and start indexing."""
    # Ensure HUGGINGFACEHUB_API_TOKEN is configured before uploading/processing
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise HTTPException(
            status_code=400,
            detail="HUGGINGFACEHUB_API_TOKEN environment variable is not configured on the server. Hugging Face Access Token is required for free hosted embeddings."
        )

    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a .pdf or .txt file."
        )

    # 1. Ensure the session exists in the database
    get_or_create_session(session_id)

    # 2. Check for duplicate upload in the active session
    duplicate = check_duplicate_document(session_id, file.filename)
    if duplicate:
        return {
            "status": "duplicate",
            "document": {
                "id": duplicate.id,
                "session_id": duplicate.session_id,
                "original_name": duplicate.original_name,
                "status": duplicate.status,
                "created_at": duplicate.created_at.isoformat()
            },
            "message": "This file is already attached to this session."
        }

    # 3. Create document record
    document_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    saved_file_name = f"{document_id}{file_ext}"
    file_path = os.path.join(DOCUMENTS_DIR, saved_file_name)

    try:
        # Save physical file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Save metadata record to database
        doc = save_document(
            document_id=document_id,
            session_id=session_id,
            original_name=file.filename,
            file_path=file_path,
            status="uploaded"
        )

        # Trigger background processing
        background_tasks.add_task(process_document_task, document_id, file_path)

        return {
            "status": "success",
            "document": {
                "id": doc.id,
                "session_id": doc.session_id,
                "original_name": doc.original_name,
                "status": doc.status,
                "created_at": doc.created_at.isoformat()
            },
            "message": "File uploaded successfully. Processing started in the background."
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/{document_id}/retry")
def retry_document_endpoint(document_id: str, background_tasks: BackgroundTasks):
    """Retry processing a failed document."""
    # Ensure HUGGINGFACEHUB_API_TOKEN is configured before retrying
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise HTTPException(
            status_code=400,
            detail="HUGGINGFACEHUB_API_TOKEN environment variable is not configured on the server."
        )

    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if doc.status == "ready":
        return {"status": "success", "message": "Document is already fully processed and ready."}

    try:
        # Reset database status
        doc = update_document_status(document_id, "uploaded", None)
        
        # Trigger background task
        background_tasks.add_task(process_document_task, document_id, doc.file_path)

        return {
            "status": "success",
            "document": {
                "id": doc.id,
                "session_id": doc.session_id,
                "original_name": doc.original_name,
                "status": doc.status,
                "created_at": doc.created_at.isoformat()
            },
            "message": "Retry processing triggered in the background."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{document_id}")
def get_document_details(document_id: str):
    """Retrieve metadata and current processing status of a document."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "id": doc.id,
        "session_id": doc.session_id,
        "original_name": doc.original_name,
        "status": doc.status,
        "error_message": doc.error_message,
        "created_at": doc.created_at.isoformat()
    }


@app.delete("/api/documents/{document_id}")
def delete_document_endpoint(document_id: str):
    """Delete a document from database, disk storage, and vector index."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        # 1. Delete physical source file
        if os.path.exists(doc.file_path):
            os.remove(doc.file_path)

        # 2. Delete FAISS vector index folder
        index_path = os.path.join(VECTOR_STORES_DIR, document_id)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)

        # 3. Evict the searcher from memory cache so it rebuilds on next query
        session_id = doc.session_id
        if session_id in active_searchers:
            del active_searchers[session_id]

        # 4. Remove record from database
        delete_document_record(document_id)

        return {"status": "success", "message": "Document deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def query_document(request: QueryRequest):
    """Query all indexed documents in a session using LLM RAG."""
    # Ensure GROQ_API_KEY is configured
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY environment variable is not configured on the server."
        )

    # Ensure HUGGINGFACEHUB_API_TOKEN is configured
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise HTTPException(
            status_code=400,
            detail="HUGGINGFACEHUB_API_TOKEN environment variable is not configured on the server."
        )

    # 1. Fetch the merged searcher for this session
    searcher = get_or_load_session_searcher(request.session_id)

    try:
        # 2. Search using combined vector store
        search_result = searcher.search(request.question)
        answer = search_result["result"]
        sources = search_result["sources"]

        # Check if this is the first interaction in the session
        existing_history = get_history(session_id=request.session_id)
        title_updated = False
        new_title = None
        if not existing_history:
            # Generate a dynamic title based on the user's question
            new_title = generate_chat_title(request.question)
            update_session_title(request.session_id, new_title)
            title_updated = True
            logger.info(f"Updated session {request.session_id} title to: {new_title}")

        # 3. Save Q&A to database history
        save_qa(
            document_name="combined_session_docs",
            question=request.question,
            answer=answer,
            session_id=request.session_id
        )

        return {
            "session_id": request.session_id,
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "title_updated": title_updated,
            "new_title": new_title
        }
    except Exception as e:
        logger.error(f"Error querying document context for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
def get_qa_history(session_id: Optional[str] = None, document_name: Optional[str] = None, limit: int = 50):
    """Retrieve Q&A history, optionally filtered by session ID or document."""
    try:
        records = get_history(document_name=document_name, session_id=session_id, limit=limit)
        return {
            "history": [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "document_name": r.document_name,
                    "question": r.question,
                    "answer": r.answer,
                    "created_at": r.created_at.isoformat()
                }
                for r in records
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/history")
def clear_qa_history(session_id: Optional[str] = None, document_name: Optional[str] = None):
    """Delete history, optionally filtered by session ID or document."""
    try:
        delete_history(document_name=document_name, session_id=session_id)
        return {"status": "success", "message": "History cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
def legacy_list_documents():
    """List all successfully processed documents/vector stores available on disk (legacy compatibility)."""
    try:
        docs = []
        if os.path.exists(VECTOR_STORES_DIR):
            for name in os.listdir(VECTOR_STORES_DIR):
                path = os.path.join(VECTOR_STORES_DIR, name)
                if os.path.isdir(path):
                    if os.path.exists(os.path.join(path, "index.faiss")):
                        docs.append(name)
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
