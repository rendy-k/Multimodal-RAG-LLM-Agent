from os import environ
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory,
)
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


# Create tools
@tool
def investing(capital: int, tenure: float, dividend_yield: float) -> int:
    """Calculate the final value of investment given the capital, interest rate per annum, and tenure.
    If the dividend yield is in percentage (%), than divide it with 100 to convert it to be decimal."""
    invest = capital * (1 + dividend_yield)**tenure
    return invest

@tool
def calculate_dy(current_price: int, dividend: int) -> float:
    """Calculate dividend yield based on the dividend and and current price."""
    dividend_yield = round(dividend/current_price, 3)
    return dividend_yield

@tool
def entry_criteria(dividend_yield: float, current_price: int, yearly_payout: bool) -> str:
    """Decide whether to buy the stock (entry) based on the dividend yield, current price, and yearly payout.
    If the dividend yield is in percentage (%), than divide it with 100 to convert it to be decimal."""
    if yearly_payout == True:
        if dividend_yield >= 0.04 or current_price <= 1000:
            decision = "buy the stock"
        else:
            decision = "do not buy the stock"
    else:
        if dividend_yield >= 0.6 and current_price <= 3000:
            decision = "buy the stock"
        else:
            decision = "do not buy the stock"
    
    return decision


def create_agent():
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer not more than 30 words. All number must be in decimal., never use percentage (%)"),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Create the agent with tools
    tools = [investing, calculate_dy, entry_criteria]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


def convert_history_string(history):
    history_input = []
    history_output = []
    
    if history != "":
        # Convert history to input and output
        history_list = history.replace(
            "\nHuman: ", "---split---Human: "
        ).replace("\nAI: ", "---split---AI: ").replace("\nSystem: ", "---split---System: ")
        history_list = history_list.split("---split---")

        for m in history_list:
            if m[:7] == "Human: ":
                history_input.append(m[7:])
            elif m[:4] == "AI: ":
                history_output.append(m[4:])
            elif m[:8] == "System: ":
                history_input.append(m[8:])
                history_output.append("")

    return history_input, history_output


def add_history_to_memory(history_memory, memory):
    # Convert history in string to list
    history_input, history_output = convert_history_string(history_memory)

    # Add chat history to memory
    if len(history_input) > 0:
        for inp, outp in zip(history_input, history_output):
            memory.save_context({"input": inp}, {"output": outp})
    
        # Load memory variables
        loaded_memory = memory.load_memory_variables({})
        # Convert history in string to list after loading memory variables
        history_input, history_output = convert_history_string(loaded_memory['history'])
        # Add chat history to memory
        memory.clear()
        if len(history_input) > 0:
            for inp, outp in zip(history_input, history_output):
                memory.save_context({"input": inp}, {"output": outp})
    
    return memory


def ask_llm(query_body: ChatQuery):
    # Assign memory
    memory = choose_memory(query_body.memory, query_body.memory_arg)
    
    memory = add_history_to_memory(query_body.history_memory, memory)

    # Ask the llm
    agent_executor = create_agent()
    response = agent_executor.invoke(
        {"input": query_body.query, "history": memory.chat_memory.messages}
    )

    # Answer
    output = response["output"]
    
    # Convert memory to string
    chat_history = ""
    for message in memory.chat_memory.messages:
        role = type(message).__name__.replace("Message", "")
        content = message.content
        chat_history = chat_history + f"{role}: {content}\n"
    chat_history = chat_history[:-1]

    # Add new chat history
    chat_history = chat_history + "\nHuman: " + query_body.query + "\nAI: " + output
    
    result = {"output": output, "history": chat_history}

    # Use this if without agent
    # conversation = ConversationChain(llm=llm, memory=memory)
    # response = conversation(query_body.query)

    return result


def input_query(query_body: ChatQuery):
    # Input the query and history to llm
    response = ask_llm(query_body)
    
    return response
