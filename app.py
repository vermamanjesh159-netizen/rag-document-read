import streamlit as st
import tempfile
import os
from pathlib import Path
from ingest import process_document
from search_engine import DocumentSearcher
from database import init_db, save_qa, get_history, delete_history
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize DB tables on startup
init_db()

# Page configuration
st.set_page_config(
    page_title="AI Document Search",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("📚 AI-Powered Document Search")
st.markdown("Upload a document and ask questions about its content!")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'searcher' not in st.session_state:
    st.session_state.searcher = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_doc' not in st.session_state:
    st.session_state.current_doc = None

# Check for API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("""
    ⚠️ GROQ_API_KEY not found!

    Please create a `.env` file in the project root with:
    GROQ_API_KEY=your_key_here
    Get your API key from: https://console.groq.com
    """)
    st.stop()
else:
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "invalid format"
    st.sidebar.success(f"✅ API Key loaded: {masked_key}")

# Sidebar
with st.sidebar:
    st.header("📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt'],
        help="Upload PDF or TXT files"
    )

    if uploaded_file:
        if st.session_state.current_doc != uploaded_file.name:
            with st.spinner("🔄 Processing document..."):
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(uploaded_file.name).suffix
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    st.session_state.vectorstore = process_document(tmp_path)

                    if st.session_state.vectorstore:
                        st.session_state.searcher = DocumentSearcher(
                            st.session_state.vectorstore
                        )
                        st.session_state.current_doc = uploaded_file.name
                        st.session_state.messages = []
                        st.success(f"✅ Processed: {uploaded_file.name}")
                    else:
                        st.error("❌ Failed to process document")

                    os.unlink(tmp_path)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Display current document
    if st.session_state.current_doc:
        st.info(f"📌 Current document: **{st.session_state.current_doc}**")

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # --- Q&A History Panel ---
    st.header("🗄️ Q&A History")

    history_filter = st.checkbox("Filter by current document", value=True)
    doc_filter = st.session_state.current_doc if history_filter else None

    history_records = get_history(document_name=doc_filter, limit=20)

    if history_records:
        for record in history_records:
            with st.expander(f"🕐 {record.created_at.strftime('%b %d, %H:%M')} — {record.question[:50]}..."):
                st.markdown(f"**📄 Document:** {record.document_name}")
                st.markdown(f"**❓ Q:** {record.question}")
                st.markdown(f"**💬 A:** {record.answer}")
        if st.button("🗑️ Delete History"):
            delete_history(document_name=doc_filter)
            st.success("History deleted!")
            st.rerun()
    else:
        st.caption("No history yet.")

    st.divider()

    # Debug info
    with st.expander("🔧 Debug Info"):
        st.write(f"Python version: {os.sys.version}")
        st.write(f"Vector store loaded: {st.session_state.vectorstore is not None}")
        st.write(f"Messages: {len(st.session_state.messages)}")

# Main chat interface
if st.session_state.vectorstore and st.session_state.searcher:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.searcher.search(prompt)
                st.markdown(response)

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

        # 💾 Save to database
        save_qa(
            document_name=st.session_state.current_doc or "unknown",
            question=prompt,
            answer=response,
        )

else:
    # Welcome message when no document is loaded
    col1, col2 = st.columns(2)

    with col1:
        st.info("👈 Please upload a document from the sidebar to start asking questions!")

    with col2:
        st.markdown("""
        ### ✨ Features:
        - 📄 Upload PDF or TXT documents
        - 🔍 Ask questions about document content
        - 🤖 AI-powered answers based on your documents
        - 💬 Chat-like interface
        - 🗄️ Q&A history saved to database
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit + LangChain + Groq + SQLAlchemy")