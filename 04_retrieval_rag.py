"""
Exercise 4: Building a Simple Retrieval System with LangChain

Objective:
- Build a basic RAG (Retrieval-Augmented Generation) pipeline.

Instructions:
1. Load GOOGLE_API_KEY.
2. Ingest a document (PDF or Web).
3. Split the document into small chunks (RecursiveCharacterTextSplitter).
4. Generate embeddings using GoogleGenerativeAIEmbeddings.
5. Store chunks in a Vector Store (e.g., Chroma or FAISS).
6. Create a Retriever from the Vector Store.
7. Use RetrievalQA (or the newer create_retrieval_chain) to answer questions based on the document.
8. Test with questions that require specific context from the loaded file.
"""

import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

class SafeGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts, **kwargs):
        texts = list(texts)
        vectors = super().embed_documents(texts, batch_size=1, **kwargs)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(vectors)}"
            )
        return vectors

# 1. Load and split the document
pdf_loader = PyPDFLoader("The Bhagavad Gita.pdf")
docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

# 2. Setup Embeddings and Vectorstore
# Correction: Switched to Google's dedicated text embedding model
persist_dir = "./chroma_db"

# Setup Embeddings
embeddings = SafeGoogleEmbeddings(
    model="gemini-embedding-2-preview",
    task_type="RETRIEVAL_DOCUMENT"
)

# Check if the local database already exists
if os.path.exists(persist_dir):
    print("Loading vector database from local storage...")
    vectorstore = Chroma(
        collection_name="gita_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir # <--- Loads from disk
    )
else:
    print("Local database not found. Reading PDF and generating embeddings...")
    # 1. Load and split the document
    pdf_loader = PyPDFLoader("The Bhagavad Gita.pdf")
    docs = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Strip empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    # Initialize Chroma WITH the persist_directory to save to disk
    vectorstore = Chroma(
        collection_name="gita_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir # <--- Saves to disk
    )

    # Batch and add documents to the local database
    print(f"Adding {len(valid_chunks)} chunks to Vector Store in batches...")
    batch_size = 50
    for i in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        time.sleep(1)

    print("Finished saving to local storage!")

# Create the retriever from the local database
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Setup LLM
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.0)
parser = StrOutputParser()
# 4. Create a ChatPromptTemplate
# The modern retrieval chain automatically injects retrieved docs into the 'context' variable
# and the user query into the 'input' variable.
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are Krishna, speaking with calm wisdom.\n\n"
        "Use the context below to answer the question.\n\n"
        "Context:\n{context}"
    )),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
result = chain.invoke("What is the meaning of life?")
print(result)
