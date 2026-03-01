from os import environ
from dotenv import load_dotenv
import json
import os
import csv
from typing import Literal
from pydantic import BaseModel
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import Tool, StructuredTool
from features.validator import ChatQuery
from features.chat_memory import update_memory, prepare_memory
from features.rag import activities_rag
from features.text2sql import hotels_flights_text2sql, execute_query


# Read the environment
load_dotenv()
api_key_groq = environ['API_GROG']
config = json.load(open("configuration.json"))
model_name = config["model_name"]
temperature = config["temperature"]
max_tokens = config["max_tokens"]
window_buffer_memory = config["window_buffer_memory"]


def load_llm(model_name):
    # Load LLM
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key_groq
    )
    # llm = ChatOpenAI(model="gpt-4o-mini", api_key=TOKEN_OPENAI, temperature=1, max_tokens=50)

    return llm


# Load llm
llm = load_llm(model_name)


# Create tools
def tool_activities_rag(question: str) -> str:
    answer = product_rag.invoke(question)
    return answer


def tool_hotels_flights_generate_sql(query: str):
    generated_sql, reasoning_sql, response = hotels_flights_text2sql(query)
    return generated_sql


def tools_hotels_flights_execute_sql(generated_sql: str):
    query_result = execute_query(generated_sql)
    return str(query_result)


def tools_booking(name: str, subject: Literal["activity", "hotel", "flight"], price: int, remarks: str=""):
    result = "Failed."
    try:
        result = "Failed."
        with open("database/booking_status.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([name, subject, f"Rp {price:,}", remarks])
        result = "Done."
    except:
        pass
    return result


# Create tools
class BookingInput(BaseModel):
    name: str
    subject: Literal["activity", "hotel", "flight"]
    price: int
    remarks: str = ""

tools = [
    Tool(
        name="tool_activities_rag",
        func=tool_activities_rag,
        description="Contains the information of activities and attractions.",
    ),
    Tool(
        name="tool_hotels_flights_generate_sql",
        func=tool_hotels_flights_generate_sql,
        description="Use this tool to answer questions about hotels and flights. It converts natural language into SQL queries. The argument 'query' will accept the message to convert to the SQL.",
    ),
    Tool(
        name="tools_hotels_flights_execute_sql",
        func=tools_hotels_flights_execute_sql,
        description="Use this tool to execute SQL about hotels and flights. It accepts the generated SQL output from the tool 'tool_hotels_flights_generate_sql' and returns the list in string format.",
    ),
    StructuredTool.from_function(
        func=tools_booking,
        name="tools_booking",
        description="Use this tool to book (1) activities, (2) hotels, and (3) flights. This tool accepts 4 arguments: name, subject, price, and remarks. 'Subject' must be 'activity', 'hotel', or 'flight'. 'Name' is the name of the subject. Add additional information in the 'remakrs', for example, hotel contains the address and free cancellation. fligh 'remarks' contains departure time and arrival time. For activities, the 'remarks' is empty string. If any of the 4 argument is not provided, then find it using the tool 'tools_hotels_flights_execute_sql'.",
        args_schema=BookingInput,
    )
]


# Load the RAG
product_rag = activities_rag(llm)


def sales_agent():
    system_prompt = """
    You are a travel assistant.
    If you think that the latest question is a follow-up question referring to the previous chat history, create a new question enhanced by the chat history before calling any tools.
    You have access to the following tools:\n{tools}. When you need to use a tool, call it exactly by name: {tool_names}.
    If you do not know the answer, say that the answer is not provided. 
    The answer must be less than 150 words.
    """

    # Create prompt template
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent


# Create the agent
sales_agent_executor = sales_agent()


def ask_llm(query_body: ChatQuery, ai_agent=True):
    if ai_agent == True:
        # Prepare the memory history
        messages = prepare_memory(query_body.history_memory)
        current_chat_length = len(messages)
        messages = messages[:min(window_buffer_memory, current_chat_length)]
        messages.append(("user", query_body.query))

        # Invoke the LLM
        response = sales_agent_executor.invoke({"messages": messages})

        # Get the Answer
        output = response['messages'][-1].content
        
        # Get the reasoning
        reasoning = ""
        for chat_i in range(current_chat_length+1, len(response['messages'])):
            try:
                reasoning += (
                    response['messages'][chat_i].__class__.__name__ + ": "
                    + response['messages'][chat_i].content
                    + response['messages'][chat_i].additional_kwargs.get("reasoning_content", "") + "\n"
                )
            except:
                pass
        
        # Update memory
        chat_history = update_memory(query_body.history_memory, query_body.query, output)
        result = {"output": output, "history": chat_history, "reasoning": reasoning}

    else:
        pass

    return result


def input_query(query_body: ChatQuery):
    # Input the query and history to llm
    response = ask_llm(query_body)
    
    return response
