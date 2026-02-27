from os import environ
from dotenv import load_dotenv
import json
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import Tool
from features.validator import ChatQuery
from features.chat_memory import update_memory, prepare_memory
from features.rag import activities_rag


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
    """
    Contains the information of activities and attractions.
    """
    answer = product_rag.invoke(question)
    return answer

def general_knowledge(query: str) -> str:
    return llm.predict(query)

# Create tools
tools = [
    Tool(
        name="tool_activities_rag",
        func=tool_activities_rag,
        description="Contains the information of activities and attractions.",
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
    The answer must be less than 80 words.
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
        messages.append(("user", query_body.query))
        current_chat_length = len(messages)

        # Invoke the LLM
        response = sales_agent_executor.invoke({"messages": messages})

        # Get the Answer
        output = response['messages'][-1].content
        
        # Get the reasoning
        reasoning = ""
        for chat_i in range(current_chat_length, len(response['messages'])):
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
