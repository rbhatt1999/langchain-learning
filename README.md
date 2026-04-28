# LangChain Mastery: Advanced AI Application Implementations

This repository contains my implementations of core LangChain concepts for building advanced AI applications. It focuses on the architecture of LLM-powered systems, from simple prompting to building autonomous agents and retrieval systems.

## 🚀 Overview
LangChain is a powerful framework designed to simplify the creation of applications using Large Language Models (LLMs). This project documents my hands-on experience building various AI orchestration components in a clean, production-oriented style.

## 🛠️ Key Concepts Implemented
- **Model Integration**: Working with different foundational models and parameter tuning.
- **Prompt Engineering**: Using `PromptTemplates` and `ChatPromptTemplates` for dynamic input.
- **Output Parsing**: Extracting structured data (JSON) from unstructured AI responses.
- **RAG (Retrieval-Augmented Generation)**: Loading documents, splitting text, and using vector databases (ChromaDB) for context-aware QA.
- **Memory Systems**: Implementing conversation history to maintain state in chatbots.
- **Chains**: Building multi-step workflows using **LCEL (LangChain Expression Language)**.
- **Agents**: Creating **ReAct Agents** that can autonomously use tools to solve complex tasks.

## 📁 Repository Structure
- `langchain_examples.py`: A clean implementation file containing modular functions for each core LangChain concept.

## 🧠 Implementation Highlights
1. **Structured Data Extraction**: Using `JsonOutputParser` to transform LLM text into actionable data objects.
2. **Retrieval Pipelines**: Building end-to-end RAG workflows with document loaders and vector stores.
3. **Functional Chains**: Utilizing LCEL to compose readable and maintainable multi-step AI logic.
4. **Autonomous Agents**: Implementing ReAct agents that bridge the gap between reasoning and action using custom toolsets.

---
*Explorations in LangChain and Generative AI.*
