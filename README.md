
## Pipeline Breakdown

| File                  | What it does                                      |
|-----------------------|---------------------------------------------------|
| `load_and_chunk.py`   | Loads PDF, splits into 219 overlapping chunks     |
| `embed_and_store.py`  | Converts chunks to vectors, stores in Qdrant      |
| `rag_pipeline.py`     | Retrieval + Groq LLM answer generation            |
| `evaluate.py`         | 20-question accuracy evaluation suite             |
| `app.py`              | Streamlit UI with citations and confidence scores |

## Tech Stack

| Tool                   | Purpose                    |
|------------------------|----------------------------|
| Python                 | Core language              |
| pypdf                  | PDF text extraction        |
| sentence-transformers  | Semantic embeddings        |
| Qdrant                 | Vector database            |
| Groq API               | LLM answer generation      |
| Streamlit              | Frontend UI                |
| Render                 | Deployment                 |

## Dataset

**Nvidia 2024 Annual Report**
- 174 pages
- ~98,000 words
- Real corporate financial and strategic data
- Chunked into 219 pieces with overlap for context preservation

## Research Motivation

Standard document QA systems fail on long corporate documents 
because they retrieve surface-level matches rather than 
semantically relevant context. This project investigates how 
careful chunking strategy, semantic similarity search, and 
grounded generation with explicit uncertainty quantification 
can produce reliable, citable answers from large real-world documents.

The "I don't know" handling and confidence scoring reflect 
a core research principle: a system that knows what it does 
not know is more trustworthy than one that always answers.

## Evaluation Journey

See `EVALUATION_JOURNEY.md` for the full breakdown of how 
accuracy was measured, what failed, and what was improved.

## Learning Notes

See `learning.md` for concepts studied while building this project.

## Author

**Om Shrirao**
B.Tech Chemical Engineering — VNIT Nagpur
GATE CS Qualified
GitHub: https://github.com/omshrirao-dev
