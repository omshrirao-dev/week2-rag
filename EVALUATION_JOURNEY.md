# RAG Evaluation Journey — From 55% to 90%

## Project
Smart Document Intelligence System built on Nvidia 2024 Annual Report (174 pages).
Evaluation suite: 20 questions measuring RAG accuracy.

## The Journey

### Run 1 — 55% accuracy
**What broke:** Most failed answers were empty strings.
**Root cause:** The 120b parameter model silently returned
None for certain prompts. No error — just empty content.
**How diagnosed:** Added debug logging to print full answer
for every failed question. Saw empty strings immediately.
**Fix:** Added null check — if content is None try fallback model.
**Lesson:** Always log actual outputs not just pass/fail scores.

---

### Run 2 — 70% accuracy
**What broke:** Questions with correct answers still failing.
**Root cause:** Keyword matching evaluation was wrong.
Example: Question asked about chip architecture.
RAG answered "Blackwell" — correct answer.
Evaluation checked for keyword "cuda" — false failure.
**How diagnosed:** Printed expected keywords vs actual answer
for every failure. Saw the mismatch immediately.
**Fix:** Updated keywords to match what document actually says.
**Lesson:** Evaluation methodology can be wrong even when
RAG output is correct. Test your tests.

---

### Run 3 — 75% accuracy
**What broke:** Last 5 questions all failed with 429 errors.
**Root cause:** Hit daily token limit on 120b parameter model.
200,000 tokens per day — evaluation suite consumed all of it.
**How diagnosed:** Error message said exactly: rate limit
exceeded, used 197,884 of 200,000 tokens.
**Fix:** Switched to smaller cheaper model for evaluation.
**Lesson:** Match model size to task complexity.
Evaluation is simple reading — does not need 120b parameters.

---

### Run 4 — 90% accuracy
**Three improvements made simultaneously:**

**Improvement 1 — LLM-as-judge scoring**
Problem: keyword matching misses semantically correct answers.
Solution: send both expected and generated answer to a judge LLM.
Ask it: "do these mean the same thing? PASS or FAIL"
This is the industry standard evaluation method used by
Anthropic, OpenAI, and Google for LLM evaluation.
Impact: caught multiple correct answers that keywords missed.

**Improvement 2 — Smaller model**
Problem: 120b model hits daily token limits during evaluation.
Solution: switched to qwen/qwen3.8-27b — much lower token cost.
Impact: no more rate limit errors across 20 questions.

**Improvement 3 — Chunk truncation**
Problem: 10 chunks × 500 words = 5000 word prompt.
Exceeds smaller model context window limit.
Solution: truncate each chunk to 300 words before sending.
Impact: prompts stay within token limits. No 413 errors.

---

## Final Results

| Run | Accuracy | Main Problem | Fix Applied |
|-----|----------|--------------|-------------|
| 1   | 55%      | Empty answers | Null check + fallback |
| 2   | 70%      | Wrong keywords | Updated evaluation |
| 3   | 75%      | Rate limits | Smaller model |
| 4   | 90%      | Evaluation methodology | LLM-as-judge |

---

## Key Takeaways

1. Log everything — empty answers look like passes without logging
2. Test your evaluation — false negatives hide real accuracy
3. Match model to task — big models waste tokens on simple tasks
4. LLM-as-judge > keyword matching for semantic evaluation
5. Diagnosis loop > writing more code when accuracy is low

---

## Files

- load_and_chunk.py — PDF loading and chunking
- embed_and_store.py — embeddings and Qdrant storage
- rag_pipeline.py — complete RAG pipeline
- app.py — Streamlit UI
- evaluate.py — evaluation suite with LLM-as-judge scoring

---

## Tech Stack
Python, sentence-transformers, Qdrant, Groq API,
Streamlit, pypdf