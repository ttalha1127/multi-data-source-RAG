from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from mongo_conn import db
import json, re, ast

@tool(description="Search and fetch data from MongoDB and show the query dynamically")
def search_mongodb(query: str) -> str:
    try:
        # Get the first collection dynamically
        collection_name = db.list_collection_names()[0]
        collection = db[collection_name]

        # LLM generates MongoDB query JSON
        llm_for_query = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        query_prompt = f"""
        Convert the following natural language query to a MongoDB JSON filter.
        Collection fields:
        passenger_id, name, age, gender, ticket_class, from_station,
        to_station, train_number, train_name, journey_date,
        booking_status, coach_number, seat_number, fare, pnr_number, booking_date.

        User query: "{query}"
        Only return the JSON filter.
        """
        mongo_query_str = llm_for_query.invoke(query_prompt).content.strip()
        import re, json
        mongo_query_str = re.sub(r'```json\n?|\n?```', '', mongo_query_str).strip()
        mongo_query = json.loads(mongo_query_str)

        # Make all string fields case-insensitive regex
        for k, v in mongo_query.items():
            if isinstance(v, str):
                mongo_query[k] = {"$regex": v, "$options": "i"}

        # Show the MongoDB query
        mongo_command_str = f"db.{collection_name}.find({json.dumps(mongo_query, indent=2)})"

        # Execute query
        results = list(collection.find(mongo_query, {"_id": 0}).limit(10))
        if not results:
            return f"🔍 MongoDB Query:\n{mongo_command_str}\n\nNo results found."

        formatted_results = "\n".join([str(r) for r in results])
        return f"🔍 MongoDB Query:\n{mongo_command_str}\n\n✅ Query Results:\n{formatted_results}"

    except Exception as e:
        return f"Error executing MongoDB query: {str(e)}"



# -----------------------
# Setup agent
# -----------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

agent = create_agent(
    llm,
    tools=[search_mongodb],
    system_prompt="""You are a smart assistant that answers all database queries using the search_mongodb tool.
    The database contains passenger data including names, ages, coach numbers, stations, train numbers, fares, and PNR details."""
)

print("💬 Assistant ready!")

# -----------------------
# User interaction
# -----------------------
while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        # Get the response
        last_message = result["messages"][-1]
        print("Agent:", last_message.content)

    except Exception as e:
        print("Error:", e)
