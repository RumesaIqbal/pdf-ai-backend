PDF Question Answering System – Backend

This repository contains the backend implementation of a PDF-based Question Answering system built using Retrieval-Augmented Generation (RAG). The backend handles PDF ingestion, text chunking, embedding generation, vector storage, similarity search, and answer generation using a Large Language Model.

🚀 Features

Upload and process PDF documents

Split documents into semantically meaningful chunks

Generate vector embeddings using Gemini Embedding models

Store and search embeddings using Supabase (pgvector)

Retrieve relevant chunks via vector similarity search

Generate context-aware answers using Gemini LLM

Session-based querying

Built with scalability and real-world constraints in mind

🧠 Architecture Overview

PDF Upload → Extract text from PDF

Chunking → Split text into manageable chunks

Embeddings → Convert chunks into vectors

Vector Store → Store embeddings in Supabase (pgvector)

Query Flow:

User question → embedding

Similarity search in Supabase

Top-k relevant chunks retrieved

Context + question passed to LLM

Final answer generated

🛠️ Tech Stack

Framework: FastAPI

LLM: Gemini 2.5 Flash (Free Tier)

Embeddings: Gemini text-embedding-004

Vector Database: Supabase (PostgreSQL + pgvector)

Deployment: Render
