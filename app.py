import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ─────────────────────────────────────────
# SETUP — load environment
# ─────────────────────────────────────────

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

# ─────────────────────────────────────────
# CACHED RESOURCES — load once, reuse always
# Without cache: 30 second reload on every message
# With cache: loads once at startup, instant after
# ─────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load embedding model and Groq client once"""
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=api_key)
    return embedding_model, groq_client


@st.cache_resource
def build_rag_index():
    """Load PDF, chunk, embed, store — runs once on startup"""

    from load_and_chunk import load_pdf, chunk_text

    embedding_model, _ = load_models()

    # Block 1 + 2: Load and chunk PDF
    text = load_pdf("nvidia_annual_report_2024.pdf")
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    # Block 3: Embed all chunks
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=False)

    # Block 4: Store in Qdrant
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
    return client, chunks


# ─────────────────────────────────────────
# RAG CORE — retrieve and answer
# ─────────────────────────────────────────

def retrieve_and_answer(question, client, embedding_model, groq_client):
    """Full RAG pipeline: question → retrieve → generate → return"""

    # Block 5: Retrieve relevant chunks
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

    # Block 6: Build prompt with retrieved context
    context = ""
    for i, chunk in enumerate(retrieved):
        context += f"\n[Source {i+1} — Score: {chunk['score']:.2f}]\n"
        context += chunk["text"]
        context += "\n"

    prompt = f"""You are an AI assistant analyzing the Nvidia 2024 Annual Report.

Answer the question using ONLY the context provided below.
If the answer is not in the context — say "I don't have enough information to answer this from the document."
Always mention which source number you used.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    # Block 6: Generate answer via Groq
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.1
    )

    answer = response.choices[0].message.content
    top_score = retrieved[0]["score"]

    return answer, retrieved, top_score


# ─────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────

# Page config — must be first st. command
st.set_page_config(
    page_title="Nvidia Document Intelligence",
    page_icon="🤖",
    layout="wide"
)

# Title and subtitle
st.title("🤖 Nvidia 2024 Annual Report — AI Assistant")
st.caption("Ask any question about Nvidia's 2024 Annual Report. Answers include source citations and confidence scores.")

# Load everything — cached so runs only once
with st.spinner("Loading document and building index... (first load takes ~30 seconds)"):
    embedding_model, groq_client = load_models()
    qdrant_client, chunks = build_rag_index()

st.success(f"✅ Ready. {len(chunks)} document chunks indexed.")

# ─────────────────────────────────────────
# SIDEBAR — document info and sample questions
# ─────────────────────────────────────────

with st.sidebar:
    st.header("📄 Document Info")
    st.write("**Source:** Nvidia 2024 Annual Report")
    st.write("**Pages:** 174")
    st.write(f"**Chunks indexed:** {len(chunks)}")
    st.write("**Embedding model:** all-MiniLM-L6-v2")
    st.write("**Vector dimensions:** 384")
    st.write("**LLM:** Groq API")
    st.write("**Vector DB:** Qdrant")

    st.divider()

    st.header("💡 Sample Questions")
    st.write("• What was Nvidia's revenue in 2024?")
    st.write("• Who is the CEO of Nvidia?")
    st.write("• What are the main risks facing Nvidia?")
    st.write("• How does the data center business perform?")
    st.write("• What does Nvidia say about AI opportunities?")
    st.write("• What is Nvidia's gross margin?")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────

# Initialize chat history in session state
# session_state persists across Streamlit reruns
# Without this — history resets on every message
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        # Show confidence for assistant messages
        if message["role"] == "assistant" and "score" in message:
            score = message["score"]
            if score > 0.6:
                st.success(f"Confidence: {score:.2f} — High")
            elif score > 0.4:
                st.warning(f"Confidence: {score:.2f} — Medium")
            else:
                st.error(f"Confidence: {score:.2f} — Low")

# Chat input — sticky bar at bottom of page
if question := st.chat_input("Ask a question about Nvidia's 2024 Annual Report..."):

    # Show user message immediately
    with st.chat_message("user"):
        st.write(question)

    # Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            answer, retrieved, top_score = retrieve_and_answer(
                question, qdrant_client, embedding_model, groq_client
            )

        # Show answer
        st.write(answer)

        # Show confidence badge
        if top_score > 0.6:
            st.success(f"Confidence: {top_score:.2f} — High")
        elif top_score > 0.4:
            st.warning(f"Confidence: {top_score:.2f} — Medium")
        else:
            st.error(f"Confidence: {top_score:.2f} — Low (question may be outside document scope)")

        # Show source chunks in collapsible section
        with st.expander("📚 View source chunks used to generate this answer"):
            for i, chunk in enumerate(retrieved[:3]):
                st.write(f"**Source {i+1}** — similarity score: {chunk['score']:.3f}")
                st.write(chunk["text"][:300] + "...")
                st.divider()

    # Save assistant message to history with score
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "score": top_score
    })