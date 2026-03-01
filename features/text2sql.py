from os import environ
import json
from dotenv import load_dotenv
import sqlite3
from langchain_groq import ChatGroq


load_dotenv()
api_key_groq = environ['API_GROG']
config = json.load(open("configuration.json"))
model_name = config["model_name"]


llm = ChatGroq(
    model=model_name,
    temperature=0,
    max_tokens=500,
    api_key=api_key_groq
)

schema = """
Table: hotels
Columns:
    id INTEGER PRIMARY KEY
    name TEXT
    address TEXT
    price INTEGER
    free_cancellation BOOL

Table: flight_schedules
Columns:
    id INTEGER PRIMARY KEY
    airplane_id INTEGER
    departure_time TIME
    arrival_time TIME
    price INTEGER
)

Table: flights
Columns:
    airplane_id INTEGER PRIMARY KEY
    airplane_name TEXT
    cabin_kg INTEGER
    baggage_kg INTEGER
    beverages TEXT (Free, Paid, Not Available)
    entertainment TEXT (Free, Paid, Not Available)

The tables flight_schedule and flights can be joined.
"""

# Execute the SQL query
def execute_query(sql_query):
    # Create connection
    conn = sqlite3.connect('database/travel.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Execute query
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results


def generate_sql_query(query, schema):
    prompt = f"""
    Given the following database schema:
    {schema}

    Translate the following query into SQL:
    "{query}"

    Rules:
    - Only generate SELECT queries, ready to execute. Do not add anything not part of the SQL.
    - Do not use INSERT, UPDATE, DELETE, DROP, ALTER.
    - Return only executable SQL.
    - Do not include explanations or markdown.
    """
    
    response = llm.invoke(prompt)
    response = response.model_dump()
    
    generated_sql = response["content"].strip().replace("```sql", "").replace("```", "")
    reasoning_sql = response.get("additional_kwargs", {}).get("reasoning_content", "")
    return generated_sql, reasoning_sql, response


def hotels_flights_text2sql(query: str):
    generated_sql, reasoning_sql, response = generate_sql_query(query, schema)

    return generated_sql, reasoning_sql, response
