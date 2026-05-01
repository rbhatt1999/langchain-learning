"""
Exercise 3: Working with Document Loaders and Text Splitters

Objective:
- Understand how to ingest external data and chunk it for LLM processing.

Instructions:
1. Load a PDF document (use a sample URL or local file).
2. Load a Web page using WebBaseLoader.
3. Implement two different splitting strategies:
   - CharacterTextSplitter (simple split).
   - RecursiveCharacterTextSplitter (context-aware split).
4. Compare the resulting chunks:
   - Total number of chunks.
   - Average chunk size.
   - Metadata preservation (check if 'source' or 'page' keys exist).
5. Print a sample chunk from each splitter to see the difference in text quality.
"""
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


load_dotenv()
pdf_loader = PyPDFLoader("The Bhagavad Gita.pdf")
docs = pdf_loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
character_chunks = text_splitter.split_documents(docs)
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
recursive_chunks = recursive_splitter.split_documents(docs)

print("Character Splitter Chunks:", len(character_chunks))
print("Recursive Splitter Chunks:", len(recursive_chunks))

def average_chunk_size(chunks):
   total_length = sum(len(chunk.page_content) for chunk in chunks)
   return total_length / len(chunks)

print("Average Character Splitter Chunk Size:", average_chunk_size(character_chunks))
print("Average Recursive Splitter Chunk Size:", average_chunk_size(recursive_chunks))

print("Character Metadata Sample:", character_chunks[0].metadata)
print("Recursive Metadata Sample:", recursive_chunks[0].metadata)

print("Character Chunk Sample:", character_chunks[0].page_content[:100])
print("Recursive Chunk Sample:", recursive_chunks[0].page_content[:100])
