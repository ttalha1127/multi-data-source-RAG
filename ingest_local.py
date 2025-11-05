import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DATA_DIR = "data"
FAISS_DIR = "faiss_index"

# Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Orcalo pc 8\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


# Step 1: Extract text from selected pages using OCR
def extract_selected_text_ocr(pdf_path, start_page=2, end_page=39):
    pdf = fitz.open(pdf_path)
    pages_text = []

    total_pages = len(pdf)
    start = max(0, start_page)
    end = min(total_pages, end_page)

    print(f"🔹 Processing pages {start+1} to {end} (Total pages in PDF: {total_pages})")

    for i in range(start, end):
        page = pdf.load_page(i)
        pix = page.get_pixmap(dpi=300)
        img = Image.open(BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        pages_text.append(text)
        print(f" OCR done for page {i+1}/{end}")

    pdf.close()
    return "\n".join(pages_text)

# Step 2: Split text into chunks
def split_text_into_chunks(text, chunk_size=800, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    print(f"🔹 Total chunks created: {len(chunks)}")
    return chunks

# Step 3: Create embeddings and store in FAISS
def create_embeddings(chunks, source_name="GLOBAL-TOURISM.pdf"):
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(page_content=chunk, metadata={"source": source_name}) for chunk in chunks]
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_DIR)
    print(f"🔹 Embeddings created and saved in FAISS folder: {FAISS_DIR}")
    return vectorstore

# Step 4: Load FAISS for querying
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    print(f"🔹 FAISS vectorstore loaded from {FAISS_DIR}")
    return vectorstore


# Step 5: Test retrieval (optional)
def test_query_faiss(query, top_k=3):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=top_k)
    
    print(f"\n🔹 Top {top_k} results for query: '{query}'")
    for i, doc in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Source:", doc.metadata.get("source"))
        print("Text:", doc.page_content[:500], "...")

# Full pipeline runner
def run_full_pipeline(pdf_filename="GLOBAL-TOURISM.pdf"):
    pdf_path = os.path.join(DATA_DIR, pdf_filename)
    print("🔹 Starting OCR extraction...")
    text = extract_selected_text_ocr(pdf_path, start_page=2, end_page=39)

    # Save OCR text to file
    output_txt_path = os.path.join(DATA_DIR, pdf_filename.replace(".pdf", ".txt"))
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n OCR text saved to: {output_txt_path}")

    # Split text into chunks
    print("\n🔹 Splitting text into chunks...")
    chunks = split_text_into_chunks(text)

    # Save chunks to file
    chunks_output_path = os.path.join(DATA_DIR, pdf_filename.replace(".pdf", "-chunks.txt"))
    with open(chunks_output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"--- Chunk {i} ---\n{chunk}\n\n")
    print(f" Text chunks saved to: {chunks_output_path}")

    # Create embeddings and store in FAISS
    print("\n🔹 Creating embeddings and storing in FAISS...")
    create_embeddings(chunks, source_name=pdf_filename)

# Standalone run
if __name__ == "__main__":
    run_full_pipeline()
