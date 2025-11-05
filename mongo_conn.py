from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]



#getting db name dynamically as well
# print(client.get_database().name)


print(" Connected to:", db.name)
print(" Collections:", db.list_collection_names())
