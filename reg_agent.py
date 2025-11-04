from ingest_local import load_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os
import requests

# 1️⃣ Load environment and retriever
load_dotenv()
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# 2️⃣ Define crypto price tool
@tool(description="Get the current price of a cryptocurrency by symbol (e.g., BTC, ETH)")
def get_crypto_price(symbol: str) -> str:
    """Fetch current cryptocurrency price from CoinMarketCap"""
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY}
    params = {"symbol": symbol.upper(), "convert": "USD"}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return f"Error fetching price for {symbol.upper()}"

    data = response.json()
    try:
        crypto_data = data["data"][symbol.upper()]
        price = crypto_data["quote"]["USD"]["price"]
        return f"The current price of {symbol.upper()} is ${price:.2f} USD."
    except KeyError:
        return f"Could not find price for {symbol.upper()}"

# 3️⃣ Define PDF retriever tool
@tool(description="Search PDF knowledge base for relevant information")
def search_pdf(query: str) -> str:
    """Retrieve text from FAISS vectorstore"""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the PDF knowledge base."
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Context from PDF:\n{context}"

# 4️⃣ Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 5️⃣ Create reasoning agent with both tools

system_prompt = """
You are a smart assistant that can access two tools:
1. 'get_crypto_price' — use when the user asks about cryptocurrencies, coins, or prices.
2. 'search_pdf' — use when the user asks about general information or topics covered in the PDF knowledge base.

Rules:
- If the query is about crypto or a coin symbol (like BTC, ETH), use get_crypto_price.
- If the query is about policies, tourism, or any other general info, use search_pdf.
- If the query is unrelated to crypto or the PDF knowledge base, respond with:
  "I can only answer questions about cryptocurrency prices or from the PDF data."
"""

agent = create_agent(
    llm,
    tools=[get_crypto_price, search_pdf],
    system_prompt=system_prompt
)

# 6️⃣ User interaction
print("\n💬 Smart Assistant is ready!")
print("Ask something like:")

while True:
    user_input = input("\nYou: ")   # Take any question from the user
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    try:
        #  Correct: wrap user input in 'messages'
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": user_input}
            ]
        })
        #  Print the final assistant message
        print("Agent:", result["messages"][-1].content)

    except Exception as e:
        print("Error:", str(e))

