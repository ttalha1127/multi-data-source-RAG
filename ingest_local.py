# ingest_local.py

import os
import pickle
import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Config
DATA_DIR = "data"
FAISS_DIR = "faiss_index"
os.makedirs(FAISS_DIR, exist_ok=True)

# Initialize Gemini embeddings
# embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def build_index_from_pdfs(pdf_paths):
    # 1. Load PDFs using PyPDFLoader
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    print(f"Loaded {len(docs)} documents from PDFs.")
    return docs


#     # 2. Split documents into manageable chunks
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = splitter.split_documents(docs)

#     # 3. Generate embeddings
#     texts = [chunk.page_content for chunk in chunks]
#     embeddings = embedder.embed_documents(texts)
#     embed_dim = len(embeddings[0])

#     # 4. Build and save FAISS index
#     index = faiss.IndexFlatL2(embed_dim)
#     index.add(embeddings)
#     faiss.write_index(index, os.path.join(FAISS_DIR, "index.faiss"))

#     # 5. Save metadata (chunks)
#     with open(os.path.join(FAISS_DIR, "docs.pkl"), "wb") as f:
#         pickle.dump(chunks, f)

#     print(f"FAISS index built and saved to {FAISS_DIR}")

# if __name__ == "__main__":
#     pdfs = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
#     if pdfs:
#         build_index_from_pdfs(pdfs)
#     else:
#         print("No PDF files found in the data directory.")
