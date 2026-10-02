import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

# ─────────────────────────────────────────
# 20 EVALUATION QUESTIONS
# ─────────────────────────────────────────

EVAL_QUESTIONS = [
    {"id": 1, "question": "What was Nvidia's total revenue in fiscal year 2024?", "expected": "Nvidia's total revenue was around 60.9 billion dollars"},
    {"id": 2, "question": "Who is the CEO of Nvidia?", "expected": "Jensen Huang is the CEO of Nvidia"},
    {"id": 3, "question": "What is Nvidia's primary product?", "expected": "Nvidia's primary products are GPUs and graphics processors"},
    {"id": 4, "question": "What was Nvidia's data center revenue in 2024?", "expected": "Nvidia's data center revenue was around 47.5 billion dollars"},
    {"id": 5, "question": "What chip architecture does Nvidia use?", "expected": "Nvidia uses Blackwell and CUDA architecture"},
    {"id": 6, "question": "What does NVIDIA stand for?", "expected": "The document does not explain what NVIDIA stands for"},
    {"id": 7, "question": "What are the main business segments of Nvidia?", "expected": "Nvidia's main segments are compute and networking and graphics"},
    {"id": 8, "question": "What was Nvidia's gross margin in 2024?", "expected": "Nvidia's gross margin was around 72 to 73 percent"},
    {"id": 9, "question": "What risks does Nvidia face from export controls?", "expected": "US government export controls restrict Nvidia chip sales to China"},
    {"id": 10, "question": "What GPU chips does Nvidia sell for AI workloads?", "expected": "Nvidia sells H100 and other GPUs for AI training"},
    {"id": 11, "question": "How much did Nvidia's revenue grow year over year?", "expected": "Nvidia revenue grew over 120 percent year over year"},
    {"id": 12, "question": "What is Nvidia Omniverse?", "expected": "Nvidia Omniverse is a simulation and digital twin platform"},
    {"id": 13, "question": "Where is Nvidia headquartered?", "expected": "Nvidia is headquartered in Santa Clara California"},
    {"id": 14, "question": "What is CUDA used for?", "expected": "CUDA is used for parallel computing and GPU programming"},
    {"id": 15, "question": "What was Nvidia's operating income in 2024?", "expected": "Nvidia's operating income was around 32 billion dollars"},
    {"id": 16, "question": "What industries does Nvidia serve?", "expected": "Nvidia serves gaming, data center, automotive, and healthcare industries"},
    {"id": 17, "question": "What are Nvidia's AI computing systems?", "expected": "Nvidia has DGX systems and other AI computing platforms"},
    {"id": 18, "question": "How does Nvidia describe its competitive position?", "expected": "Nvidia describes itself as a leader in accelerated computing and AI"},
    {"id": 19, "question": "What is the weather in Santa Clara today?", "expected": "The document does not contain weather information"},
    {"id": 20, "question": "What is Nvidia's stock price right now?", "expected": "The document does not contain current stock price information"},
]


# ─────────────────────────────────────────
# SETUP RAG PIPELINE
# ─────────────────────────────────────────

def setup_pipeline():
    from load_and_chunk import load_pdf, chunk_text

    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=api_key)

    print("Loading and chunking PDF...")
    text = load_pdf("nvidia_annual_report_2024.pdf")
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    print("Embedding chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=False)

    print("Storing in Qdrant...")
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="nvidia_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={"text": chunk["text"], "chunk_id": chunk["id"]}
        ))

    client.upsert(collection_name="nvidia_docs", points=points)
    print(f"Pipeline ready. {len(chunks)} chunks indexed.\n")
    return embedding_model, groq_client, client


# ─────────────────────────────────────────
# LLM AS JUDGE
# Your idea — compare meaning not keywords
# ─────────────────────────────────────────

def judge_answer(question, expected, generated, groq_client):
    prompt = f"""You are evaluating a RAG system answer.

Question: {question}
Expected answer: {expected}
Generated answer: {generated}

Does the generated answer correctly address the question
with the same meaning as the expected answer?
If both say the document does not contain the information
that is also a PASS.

Reply with only one word: PASS or FAIL"""

    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        verdict = response.choices[0].message.content
        if verdict:
            return "pass" in verdict.lower()
        return False

    except Exception as e:
        print(f"    JUDGE ERROR: {str(e)}")
        return False


# ─────────────────────────────────────────
# EVALUATE ONE QUESTION
# ─────────────────────────────────────────

def evaluate_question(question_data, embedding_model, groq_client, qdrant_client):

    question = question_data["question"]
    expected = question_data["expected"]

    # Retrieve 5 chunks — keeps prompt within token limits
    question_vector = embedding_model.encode([question])[0]
    results = qdrant_client.query_points(
        collection_name="nvidia_docs",
        query=question_vector.tolist(),
        limit=5
    ).points

    retrieved = []
    for result in results:
        retrieved.append({
            "text": result.payload["text"],
            "score": result.score
        })

    # Build prompt — truncate each chunk to 300 words
    # Keeps total prompt under token limits
    context = ""
    for i, chunk in enumerate(retrieved):
        chunk_text = " ".join(chunk["text"].split()[:300])
        context += f"\n[Source {i+1}]\n{chunk_text}\n"

    prompt = f"""You are an AI assistant analyzing the Nvidia 2024 Annual Report.
Answer using ONLY the context below.
If the answer is not in the context — say "I don't have enough information."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    # Generate using cheaper model
    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )
        content = response.choices[0].message.content
        generated = content if content else "no answer returned"

    except Exception as e:
        print(f"    API ERROR: {str(e)}")
        return {
            "id": question_data["id"],
            "question": question,
            "generated": f"API error: {str(e)}",
            "passed": False,
            "retrieval_score": retrieved[0]["score"]
        }

    # Judge using LLM — your idea
    passed = judge_answer(question, expected, generated, groq_client)

    return {
        "id": question_data["id"],
        "question": question,
        "generated": generated,
        "passed": passed,
        "retrieval_score": retrieved[0]["score"]
    }


# ─────────────────────────────────────────
# RUN FULL EVALUATION
# ─────────────────────────────────────────

def run_evaluation():

    print("=" * 60)
    print("RAG EVALUATION SUITE — 20 Questions")
    print("Scoring: LLM-as-judge (your idea)")
    print("Model: qwen/qwen3.8-27b")
    print("Chunks: 5 per question, 300 words each")
    print("=" * 60)

    embedding_model, groq_client, qdrant_client = setup_pipeline()

    results = []
    passed = 0
    failed = 0

    for q in EVAL_QUESTIONS:
        print(f"Testing Q{q['id']}: {q['question'][:50]}...")

        result = evaluate_question(
            q, embedding_model, groq_client, qdrant_client
        )

        results.append(result)

        if result["passed"]:
            passed += 1
            print(f"  ✓ PASS (retrieval score: {result['retrieval_score']:.3f})")
        else:
            failed += 1
            print(f"  ✗ FAIL (retrieval score: {result['retrieval_score']:.3f})")
            print(f"    Answer: {result['generated'][:150]}")

    accuracy = (passed / len(EVAL_QUESTIONS)) * 100

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total questions:  {len(EVAL_QUESTIONS)}")
    print(f"Passed:           {passed}")
    print(f"Failed:           {failed}")
    print(f"Accuracy:         {accuracy:.1f}%")
    print("Scoring method:   LLM-as-judge")
    print("=" * 60)

    if failed > 0:
        print("\nFAILED QUESTIONS:")
        for r in results:
            if not r["passed"]:
                print(f"\nQ{r['id']}: {r['question']}")
                print(f"Answer: {r['generated'][:200]}")

    return accuracy, results


if __name__ == "__main__":
    accuracy, results = run_evaluation()