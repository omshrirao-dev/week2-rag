import os
from pypdf import PdfReader

# ─────────────────────────────────────────
# BLOCK 1 — LOAD
# Read the PDF and extract all text
# ─────────────────────────────────────────

def load_pdf(filepath):
    """Load PDF and extract all text page by page"""
    
    print(f"Loading PDF: {filepath}")
    
    reader = PdfReader(filepath)
    total_pages = len(reader.pages)
    
    print(f"Total pages found: {total_pages}")
    
    # Extract text from every page
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n[Page {i+1}]\n{text}"
    
    print(f"Total characters extracted: {len(full_text)}")
    return full_text


# ─────────────────────────────────────────
# BLOCK 2 — CHUNK
# Split text into smaller overlapping pieces
# ─────────────────────────────────────────

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks with overlap.
    
    chunk_size: how many words per chunk
    overlap: how many words to repeat between chunks
    
    Why overlap? So important information at chunk 
    boundaries is not lost. If a sentence spans two 
    chunks — overlap ensures context is preserved.
    """
    
    # Split into words
    words = text.split()
    total_words = len(words)
    
    print(f"Total words in document: {total_words}")
    
    chunks = []
    start = 0
    chunk_number = 0
    
    while start < total_words:
        # Get chunk_size words starting from current position
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "id": f"chunk_{chunk_number}",
            "text": chunk_text,
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words)
        })
        
        chunk_number += 1
        
        # Move forward by chunk_size minus overlap
        # This creates the overlapping effect
        start += chunk_size - overlap
    
    print(f"Total chunks created: {len(chunks)}")
    print(f"Chunk size: {chunk_size} words")
    print(f"Overlap: {overlap} words")
    
    return chunks


# ─────────────────────────────────────────
# TEST IT — run this file directly
# ─────────────────────────────────────────

if __name__ == "__main__":
    
    # Load the PDF
    pdf_path = "nvidia_annual_report_2024.pdf"
    text = load_pdf(pdf_path)
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    # Show first 3 chunks so we can see what they look like
    print("\n" + "="*50)
    print("FIRST 3 CHUNKS PREVIEW")
    print("="*50)
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} (ID: {chunk['id']})")
        print(f"Words: {chunk['word_count']}")
        print(f"Preview: {chunk['text'][:200]}...")
        print("-"*30)
    
    print(f"\nTotal chunks ready for embedding: {len(chunks)}")

    