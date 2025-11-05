from ingest_local import load_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os
import requests
from mongo_conn import db
import re
import json

# 1 Load environment and retriever
load_dotenv()
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2 Define crypto price tool
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

# 3 Define PDF retriever tool
@tool(description="Search PDF knowledge base for relevant information")
def search_pdf(query: str) -> str:
    """Retrieve text from FAISS vectorstore"""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the PDF knowledge base."
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Context from PDF:\n{context}"

# 4 Define MongoDB search tool
@tool(description="Search and fetch data from MongoDB and show the query dynamically")
def search_mongodb(query: str) -> str:
    """
    Convert user natural language query into MongoDB query dynamically,
    show the query in db.collection.find({...}) format, and fetch results.
    """
    try:
        # Dynamically pick the first collection
        collections = db.list_collection_names()
        if not collections:
            return "No collections found in the database."
        collection_name = collections[0]
        collection = db[collection_name]

        # Define field mapping for LLM
        field_info = """
        The collection has the following fields:
        passenger_id, name, age, gender, ticket_class,
        from_station, to_station, train_number, train_name,
        journey_date, booking_status, coach_number, seat_number,
        fare, pnr_number, booking_date
        """

        # Prompt LLM to generate MongoDB query and projection
        llm_for_query = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        query_prompt = f"""
        You are a MongoDB expert. Convert the user's natural language query
        into a MongoDB find query for the collection '{collection_name}'.
        
        Return a JSON with two keys:
        - "filter": the filter criteria for find()
        - "projection": which fields to include (only include relevant fields based on user query)
        
        Available fields: {field_info}

        User query: "{query}"

        Examples:
        - If user asks "show me names of female passengers", return:
          {{"filter": {{"gender": "female"}}, "projection": {{"name": 1}}}}

        - If user asks "booking dates for Priya Patel", return:
          {{"filter": {{"name": "Priya Patel"}}, "projection": {{"booking_date": 1, "name": 1}}}}

        - If user asks "age of all passengers", return:
          {{"filter": {{}}, "projection": {{"name": 1, "age": 1}}}}

        Now generate the JSON for the current query:
        """
        
        mongo_query_str = llm_for_query.invoke(query_prompt).content.strip()
        mongo_query_str = re.sub(r'```json\n?|\n?```', '', mongo_query_str).strip()

        # Parse JSON
        query_data = json.loads(mongo_query_str)
        mongo_filter = query_data.get("filter", {})
        projection = query_data.get("projection", {"_id": 0})

        # Ensure _id is excluded unless specifically requested
        if "_id" not in projection:
            projection["_id"] = 0

        # Show MongoDB query
        mongo_command_str = f"db.{collection_name}.find({json.dumps(mongo_filter, indent=2)}, {json.dumps(projection, indent=2)})"

        # Execute query
        results = list(collection.find(mongo_filter, projection).limit(10))

        if not results:
            return f"No results found for your query."

        # Format the MongoDB query for display
        clean_query = f"db.{collection_name}.find({json.dumps(mongo_filter)}, {json.dumps(projection)})"

        # FIXED: Correct indentation for the for loop
        formatted_results = []
        for result in results:
            relevant_fields = {k: v for k, v in result.items() if k != "_id"}
            if relevant_fields:
                result_str = ", ".join([f"{k}: {v}" for k, v in relevant_fields.items()])
                formatted_results.append(result_str)

        if not formatted_results:
            return "No relevant data found."

        return f"""🔍 **Search Results:**

**Query Used:** `{clean_query}`

**Data Found:**
{chr(10).join(formatted_results)}"""

    except Exception as e:  # Make sure this except is properly aligned
        return f"Error executing MongoDB query: {str(e)}"

# 5 Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 6 Create reasoning agent with all tools
system_prompt = """
You are a smart assistant that can access three tools:
1. 'get_crypto_price' — use when the user asks about cryptocurrencies, coins, or prices.
2. 'search_pdf' — use when the user asks about general information or topics covered in the PDF knowledge base.
3. 'search_mongodb' — use when the user asks to search, fetch, or query data from the database.

Rules:
- If the query is about crypto or a coin symbol (like BTC, ETH), use get_crypto_price.
- If the query is about policies, tourism, or any other general info, use search_pdf.
- If the query is about database records, searching data, or fetching information from MongoDB, use search_mongodb.
- When using search_mongodb, the tool will automatically show only relevant fields.
- If the query is unrelated to these tools, respond with:
  "I can only answer questions about cryptocurrency prices, PDF knowledge base, or database queries."
"""

agent = create_agent(
    llm,
    tools=[get_crypto_price, search_pdf, search_mongodb],
    system_prompt=system_prompt
)

# 7 User interaction
print("\n Smart Assistant is ready!")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    try:
        # Invoke agent
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        # Extract the final response
        final_response = ""
        
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            # Find the last AI message
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content:
                    final_response = msg.content
                    break
                elif isinstance(msg, dict) and msg.get('content'):
                    final_response = msg['content']
                    break
        
        if not final_response:
            final_response = str(result)
        
        print("\n Assistant:", final_response)

    except Exception as e:
        print("Error:", e)