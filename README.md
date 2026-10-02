---
title: Nvidia Document Intelligence
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Nvidia 2024 Annual Report — AI Assistant

Ask any question about Nvidia's 2024 Annual Report.
Built with RAG — Retrieval Augmented Generation.

## Features
- Semantic search across 219 document chunks
- Source citations with every answer
- Confidence scoring (High/Medium/Low)
- Hallucination prevention — refuses out-of-scope questions
- 90% accuracy on 20-question evaluation suite

## Tech Stack
Python, Sentence Transformers, Qdrant, Groq API, Streamlit