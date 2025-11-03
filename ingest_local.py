import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.documents import Document

# Setup
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
    return "\n".join(pages_text)  # Combine all pages into one string for a single text file

# Step 2: Create embeddings and store in FAISS
def create_embeddings(text, file_name="GLOBAL-TOURISM.txt"):
    # Wrap text in a Document
    doc = Document(page_content=text, metadata={"source": file_name})
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents([doc], embeddings)
    # Save FAISS locally
    vectorstore.save_local(FAISS_DIR)
    print(f" Embeddings created and saved in FAISS folder: {FAISS_DIR}")


# Run the flow
if __name__ == "__main__":
    pdf_path = os.path.join(DATA_DIR, "GLOBAL-TOURISM.pdf")
    print("🔹 Starting OCR extraction...")
    text = extract_selected_text_ocr(pdf_path, start_page=2, end_page=39)

    # # Optional: preview first 1000 chars
    # print("\n🔹 Sample Extracted Text:\n")
    # print(text[:1000])

    # Save OCR text to a single file
    output_txt_path = os.path.join(DATA_DIR, "GLOBAL-TOURISM.txt")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n OCR text saved to: {output_txt_path}")

    # Create embeddings and store in FAISS
    print("\n🔹 Creating embeddings and storing in FAISS...")
    create_embeddings(text, file_name="GLOBAL-TOURISM.txt")
