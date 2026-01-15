PDF Question Answering System – Backend

This repository contains the backend implementation of a PDF-based Question Answering system built using Retrieval-Augmented Generation (RAG). The backend handles PDF ingestion, text chunking, embedding generation, vector storage, similarity search, and answer generation using a Large Language Model.

Architecture Overview

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

 Tech Stack

Framework: FastAPI

LLM: Gemini 2.5 Flash (Free Tier)

Embeddings: Gemini text-embedding-004

Vector Database: Supabase (PostgreSQL + pgvector)

Deployment: Render
