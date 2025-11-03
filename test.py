from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Replace this with your key (string)
API_KEY = "AIzaSyDPBapzIvKrLK85ZZLUCmgfk1ny31jn-T4"

# Small dummy text
dummy_text = ["Hello world! This is a test to check embeddings."]

embeddings_client = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=API_KEY  # Changed from api_key to google_api_key
)

# Try embedding the dummy text
result = embeddings_client.embed_documents(dummy_text)
print("✅ Embedding works! Here's a preview:")
print(result[0][:8], "...")  # Show first few numbers only