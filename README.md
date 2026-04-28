# LangChain Mastery: Advanced AI Application Architecture

This repository documents my implementation and mastery of core LangChain concepts for building advanced AI applications. It serves as a comprehensive reference for modern LLM orchestration, from structured data extraction to autonomous agents.

## 🚀 Overview
LangChain is a powerful framework designed to simplify the creation of applications using Large Language Models (LLMs). This project focuses on the transition from simple prompting to building autonomous agents and retrieval systems.

---

## 🛠️ Key Implementations

### 1. Structured Data Extraction (JSON Output Parsing)
Implementing a robust chain that forces the LLM to return structured data instead of unstructured text. This is critical for building production-grade APIs.

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Define the structure and the chain
parser = JsonOutputParser()
prompt = PromptTemplate(
    template="Extract info about the movie {movie}.\n{format_instructions}",
    input_variables=["movie"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | llm | parser
result = chain.invoke({"movie": "Inception"})
```

### 2. Retrieval-Augmented Generation (RAG)
Building end-to-end pipelines that allow an AI to "read" and query external data (PDFs, Websites) using vector embeddings and similarity search.

- **Loaders**: `PyPDFLoader`, `WebBaseLoader`
- **Chunking**: `RecursiveCharacterTextSplitter` (balancing context vs. precision)
- **Vector Stores**: `ChromaDB` for storing and retrieving semantic embeddings.

```python
# Semantic search implementation
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What is LangChain?")
```

### 3. Modern Orchestration with LCEL
Utilizing **LangChain Expression Language (LCEL)** to build clean, maintainable, and readable multi-step AI logic using the pipe (`|`) operator.

```python
# Multi-step review processing chain
chain = (
    RunnablePassthrough.assign(sentiment=sentiment_chain)
    | RunnablePassthrough.assign(summary=summary_chain)
    | response_generation_chain
)
```

### 4. Conversation Memory Systems
Implementing stateful interactions that allow chatbots to maintain context across multiple turns using `ConversationBufferMemory`.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
conversation.invoke(input="Hi, I am a software engineer.")
conversation.invoke(input="What is my profession?") # AI remembers context
```

### 5. Autonomous ReAct Agents
Creating systems that leverage an LLM as a "reasoning engine" to autonomously use external tools (Calculators, Search APIs, Python REPL) to solve complex, multi-step problems.

- **Framework**: ReAct (Reasoning + Acting)
- **Executor**: `AgentExecutor` for managing the loop of Thought → Action → Observation.

```python
from langchain.agents import create_react_agent, AgentExecutor

# Defining tools and initializing the agent
tools = [CalculatorTool, WeatherTool]
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "What is the square root of 256?"})
```

---

## 📁 Repository Structure
- `README.md`: The single source of truth containing implementation details and conceptual architectural patterns.

## 🧠 Conclusion
This project demonstrates a deep understanding of the LangChain ecosystem, focusing on the ability to build intelligent, responsive, and tool-aware AI systems.

---
*Architecting the future of Generative AI.*
