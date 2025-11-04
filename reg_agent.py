from ingest_local import load_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
vectorstore = load_vectorstore()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0
)
# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
query = "tell me the tourism development policy six points actions ? "
docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])

# Step 2: Ask Gemini LLM with context
prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"
response = llm.invoke(prompt)

# Print answer
print("\n🔹 Answer:")
print(response.content)