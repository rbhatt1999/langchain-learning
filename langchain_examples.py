"""
LangChain Mastery: Core AI Orchestration Concepts
A collection of implementations exploring advanced LLM workflows including RAG, 
Memory, Multi-step Chains, and Autonomous Agents.
"""

import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool

# Note: Credentials and model IDs would typically be loaded from environment variables
# model_id = "meta-llama/llama-3-3-70b-instruct"

def example_json_parsing(llm):
    """Demonstrates extracting structured JSON from unstructured text."""
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template="Extract info about the movie {movie}.\n{format_instructions}",
        input_variables=["movie"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    return chain.invoke({"movie": "Inception"})

def example_rag_pipeline(embeddings):
    """Demonstrates a basic Retrieval-Augmented Generation (RAG) setup."""
    loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()

def example_lcel_chain(llm):
    """Demonstrates the modern LangChain Expression Language (LCEL)."""
    prompt = ChatPromptTemplate.from_template("Explain the concept of {topic} in one sentence.")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"topic": "Prompt Engineering"})

def example_agent_setup(llm):
    """Demonstrates setting up a ReAct Agent with custom tools."""
    def calculator(query):
        # Implementation of a safe calculation tool
        return eval(query)

    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Useful for mathematical calculations."
        )
    ]
    
    prompt = PromptTemplate.from_template("Solve: {input}\n{agent_scratchpad}")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    print("LangChain Implementations Loaded.")
    # In a real environment, you would initialize your LLM and run the examples:
    # llm = WatsonxLLM(...)
    # print(example_json_parsing(llm))
