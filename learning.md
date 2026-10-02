═══════════════════════════════════════════════════════
WEEK 2 LEARNING LOG
═══════════════════════════════════════════════════════

Day 8 — RAG Blocks 1-2
Built: PDF loader + chunker with sliding window overlap
Learned: pypdf extraction, word-based chunking, overlap
         prevents boundary information loss
Key numbers: 174 pages, 649,593 chars, 98,280 words,
             219 chunks, 500 words each, 50 word overlap
Insight: chunk_size and overlap are design decisions —
         not magic numbers

───────────────────────────────────────────────────────

Day 9 — RAG Blocks 3-5
Built: Embeddings + Qdrant storage + retrieval
Learned: text → 384 numbers (vectors), same meaning =
         similar numbers, cosine similarity measures
         angle between vectors
Key concepts:
  - SentenceTransformer — local embedding model, no API cost
  - QdrantClient(":memory:") — in-RAM database
  - collection = table, point = row, payload = metadata
  - vector = searchable fingerprint
  - query_points() replaces .search() in newer Qdrant
Error fixed: client.search() deprecated → client.query_points()
Insight: embedding model must be same for storing AND
         searching — different model = different number
         space = wrong results

───────────────────────────────────────────────────────

Day 10 — RAG Block 6
Built: Complete RAG pipeline — question to answer
Learned: temperature=0.1 for factual answers,
         build_prompt() separates concerns,
         "ONLY the context" prevents hallucination,
         confidence scoring by retrieval score
Test results:
  Revenue question    → $60.9 billion ✓ score: 0.707
  Risks question      → 6 specific risks ✓ score: 0.506
  CEO question        → Jensen Huang ✓ score: 0.681
  Weather question    → "I don't know" ✓ score: 0.244
Key insight: low score = low confidence = correct refusal
             This is what prevents hallucination

───────────────────────────────────────────────────────

Day 11 — RAG Block 7
Built: Streamlit UI — browser-based chat interface
Learned: @st.cache_resource — load once reuse always,
         st.session_state — persists across reruns,
         st.chat_message — chat bubbles,
         st.expander — collapsible sections,
         confidence badges green/yellow/red by score,
         := walrus operator — assign and check together
Error fixed: torch DLL blocked by Windows Application
             Control policy on Python 3.11
             → switched to Python 3.14 where libs
             already allowed
Key insight: session_state is the only way to persist
             data across Streamlit reruns

═══════════════════════════════════════════════════════
WEEK 2 ERRORS LOG
═══════════════════════════════════════════════════════

ERROR 1 — client.search() deprecated
When: Block 5 retrieval
Why: Newer Qdrant removed .search() method
Fix: Changed to client.query_points().points
Lesson: Library APIs change — always check version docs

ERROR 2 — ModuleNotFoundError groq on Python 3.11
When: Running app.py with py -3.11
Why: Libraries installed on 3.14 not on 3.11
Fix: Used py -3.14 where all libraries already installed
Lesson: Each Python version has its own package list

ERROR 3 — torch DLL blocked by Windows
When: Installing sentence-transformers on Python 3.11
Why: Application Control policy blocked new DLL files
Fix: Used Python 3.14 where DLLs already approved
Lesson: Corporate/institutional machines have security
        policies — always check which Python version
        your libraries are installed on

═══════════════════════════════════════════════════════
RAG CONCEPTS MASTERED — WEEK 2
═══════════════════════════════════════════════════════

Chunking         → split docs into searchable pieces
Overlap          → prevent boundary information loss
Embeddings       → text → numbers representing meaning
Cosine similarity→ angle between vectors = similarity
Vector database  → stores and searches vectors fast
Retrieval        → find most relevant chunks by meaning
Prompt building  → inject context + question into LLM
Temperature      → 0.1 for factual, 1.0 for creative
Confidence score → retrieval score tells you how sure
Hallucination    → prevented by "ONLY context" + score
Session state    → persist data across Streamlit reruns
Cache resource   → load heavy models once not every time

═══════════════════════════════════════════════════════
CURRENT STATUS
═══════════════════════════════════════════════════════

Week: Week 2
Phase: Phase 1 — RAG

Blocks complete:
✓ Block 1 — PDF loader
✓ Block 2 — Chunker with overlap
✓ Block 3 — Embeddings
✓ Block 4 — Qdrant storage
✓ Block 5 — Retrieval
✓ Block 6 — LLM generation + citations
✓ Block 7 — Streamlit UI

Remaining:
○ Block 8 — Evaluation suite (20 questions)
○ Block 9 — Deploy to Hugging Face Spaces

GitHub: github.com/omshrirao-dev/week2-rag
Live chatbot: https://week1-ai-chatbot-1.onrender.com
Next: Block 8 — evaluation suite
═══════════════════════════════════════════════════════