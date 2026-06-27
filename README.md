# 📄 RAG Document Assistant

> A lightweight Retrieval-Augmented Generation (RAG) system for querying private documents using natural language.

---

## 🚀 Overview

**RAG Document Assistant** is a production-style prototype that enables users to upload documents (PDFs) and interact with them through a conversational interface powered by LLMs.

The system retrieves relevant context from documents using semantic search and generates grounded answers based strictly on retrieved information.

It is designed as a minimal but realistic implementation of a **Retrieval-Augmented Generation pipeline**, demonstrating end-to-end architecture commonly used in modern LLM applications.

---

## 🎯 Key Capabilities

- 📄 Document ingestion (PDF support)
- 🔍 Semantic search over document chunks
- 🤖 Context-grounded question answering
- 📚 Source transparency (retrieved context visibility)
- ⚡ Lightweight, fast deployment (Streamlit-based UI)

---

## 🧠 Architecture

```text
Documents (PDF)
      ↓
Parsing & Chunking
      ↓
Embedding Generation
      ↓
Vector Store (ChromaDB)
      ↓
Similarity Retrieval
      ↓
Context Assembly
      ↓
LLM Response Generation
      ↓
User Interface (Streamlit)
