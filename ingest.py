import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import with error handling
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    print(f"❌ Import error: {e}")
    raise


def process_document(file_path):
    """
    Process uploaded document and create vector store
    """
    try:
        print(f"📄 Processing document: {file_path}")

        # Load document based on file type
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            print("✅ PDF loader created")
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            print("✅ Text loader created")
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document(s)")
        print(f"📝 First 200 chars: {documents[0].page_content[:200]}...")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Document split into {len(chunks)} chunks")

        # Create local embeddings (free, no API key needed)
        print("🔄 Creating embeddings (loading local model, first run may take a moment)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("✅ Vector store created successfully")

        return vectorstore

    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
