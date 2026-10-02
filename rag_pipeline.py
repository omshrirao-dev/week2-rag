import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Load API key
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")

# ─────────────────────────────────────────
# SETUP — models and clients
# ─────────────────────────────────────────

print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=api_key)
print("Ready.")

# ─────────────────────────────────────────
# BLOCK 6 — GENERATE
# Retrieve relevant chunks + send to LLM
# ─────────────────────────────────────────

def build_prompt(question, retrieved_chunks):
    """
    Build the prompt that goes to the LLM.
    Context = the retrieved chunks.
    Question = what the user asked.
    
    This is RAG in one function:
    Retrieval Augmented Generation
    = normal question + relevant document context
    """
    
    # Combine retrieved chunks into one context block
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"\n[Source {i+1} — Score: {chunk['score']:.2f}]\n"
        context += chunk["text"]
        context += "\n"
    
    # The prompt tells LLM:
    # 1. Only use the context provided
    # 2. Cite which source you used
    # 3. Say if you don't know
    prompt = f"""You are an AI assistant analyzing the Nvidia 2024 Annual Report.

Answer the question using ONLY the context provided below.
If the answer is not in the context — say "I don't have enough information to answer this from the document."
Always mention which source number you used.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    return prompt


def generate_answer(question, retrieved_chunks):
    """Send question + context to Groq LLM and get answer"""
    
    prompt = build_prompt(question, retrieved_chunks)
    
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=512,
        temperature=0.1  # low temperature = more factual, less creative
    )
    
    return response.choices[0].message.content


# ─────────────────────────────────────────
# FULL PIPELINE — all blocks together
# ─────────────────────────────────────────

def run_rag_pipeline():
    """Run the complete RAG pipeline from PDF to answer"""
    
    # Import our previous blocks
    from load_and_chunk import load_pdf, chunk_text
    
    # Step 1 + 2: Load and chunk
    print("\n[1/5] Loading and chunking PDF...")
    text = load_pdf("nvidia_annual_report_2024.pdf")
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    # Step 3: Embed
    print("[2/5] Embedding chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    
    # Step 4: Store in Qdrant
    print("[3/5] Storing in Qdrant...")
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
            payload={
                "text": chunk["text"],
                "chunk_id": chunk["id"]
            }
        ))
    
    client.upsert(collection_name="nvidia_docs", points=points)
    print(f"[4/5] Stored {len(points)} chunks. Ready for questions.")
    
    # Step 5: Question answering loop
    print("\n[5/5] RAG system ready. Ask questions about Nvidia.")
    print("Type 'quit' to exit.\n")
    print("="*60)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() == 'quit':
            print("Goodbye.")
            break
            
        if not question:
            print("Please enter a question.")
            continue
        
        # Retrieve relevant chunks
        question_vector = embedding_model.encode([question])[0]
        results = client.query_points(
            collection_name="nvidia_docs",
            query=question_vector.tolist(),
            limit=5
        ).points
        
        retrieved = []
        for result in results:
            retrieved.append({
                "text": result.payload["text"],
                "chunk_id": result.payload["chunk_id"],
                "score": result.score
            })
        
        # Generate answer
        print("\nSearching document...")
        answer = generate_answer(question, retrieved)
        
        print(f"\nAnswer:\n{answer}")
        print(f"\nBest match score: {retrieved[0]['score']:.3f}")
        print("="*60)


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    run_rag_pipeline()