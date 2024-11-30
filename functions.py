from os import environ
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory,
)
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from validator import ChatQuery

# Read API KEY
load_dotenv()
TOKEN_OPENAI = environ['TOKEN_OPENAI']
TOKEN_HF = environ['TOKEN_HF']


def load_llm():
    # Load LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=TOKEN_OPENAI, temperature=1, max_tokens=50)

    return llm


# Load llm
llm = load_llm()


def choose_memory(input_memory: str, memory_arg: int):
    if input_memory == "Buffer":
        memory = ConversationBufferMemory(llm=llm)
    elif input_memory == "Buffer Window":
        memory = ConversationBufferWindowMemory(llm=llm, k=memory_arg)
    elif input_memory == "Summary":
        memory = ConversationSummaryMemory(llm=llm)
    elif input_memory == "Summary Buffer":
        memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=memory_arg)
    else: #input_memory == "Token Buffer":
        memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=memory_arg)

    return memory


@tool
def saving(earn: int, spend: int) -> int:
    """Calculate the saving from earning and spending"""
    return earn - spend


@tool
def investing(capital: int, tenure: float, interest: float) -> int:
    """Calculate the final value of investment given the capital, interest rate per annum, and tenure"""
    invest = capital * (1 + interest)**tenure
    return invest


def create_agent(memory):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer not more than 20 words."),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Create the agent with tools
    tools = [saving, investing]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

    return agent_executor


def ask_llm(query_body: ChatQuery, llm):
    # Assign memory
    memory = choose_memory(query_body.memory, query_body.memory_arg)
    
    # Add chat history
    history_input = query_body.history_input
    history_output = query_body.history_output
    if history_input is not None:
        for inp, outp in zip(history_input, history_output):
            memory.save_context({"input": inp}, {"output": outp})

    # Ask the llm
    agent_executor = create_agent(memory)
    response = agent_executor.invoke({"input": query_body.query})
    
    chat_history = ""
    for message in memory.chat_memory.messages:
        role = type(message).__name__.replace("Message", "")
        content = message.content
        chat_history.append(f"{role}: {content}\n")
    chat_history = chat_history[:-1]
    

    # If without agent
    # conversation = ConversationChain(llm=llm, memory=memory)
    # response = conversation(query_body.query)

    return response


def input_query(query_body: ChatQuery):
    # Input the query and history to llm
    response = ask_llm(query_body, llm)
    
    return response
