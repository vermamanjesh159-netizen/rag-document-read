from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()


class DocumentSearcher:
    def __init__(self, vectorstore):
        """
        Initialize the document searcher with a vector store
        """
        self.vectorstore = vectorstore
        api_key = os.getenv("GROQ_API_KEY")

        # Initialize Groq LLM
        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=500,
        )

        # Custom prompt template
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer: """

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
        )

    def search(self, query):
        """
        Search for answer to query in the documents
        """
        try:
            print(f"🔍 Searching for: {query}")
            response = self.qa_chain.invoke(query)

            # Handle different response formats
            if isinstance(response, dict):
                result = response.get("result", str(response))
            else:
                result = str(response)

            print(f"✅ Got response: {result[:100]}...")
            return result

        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            return f"Error during search: {str(e)}"
