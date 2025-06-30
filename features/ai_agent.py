from os import environ
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain_core.tools import tool
from features.validator import ChatQuery
from features.chat_memory import choose_memory, add_history_to_memory, convert_memory_to_string

# Read API KEY
load_dotenv()
api_key_groq = environ['API_GROG']


def load_llm():
    # Load LLM
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=1,
        max_tokens=50,
        api_key=api_key_groq
    )
    # llm = ChatOpenAI(model="gpt-4o-mini", api_key=TOKEN_OPENAI, temperature=1, max_tokens=50)

    return llm


# Load llm
llm = load_llm()


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
            ("system", f"Answer not more than 50 words. All number must be in decimal, never use percentage (%)"),
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


# Create the agent
agent_executor = create_agent()


def ask_llm(query_body: ChatQuery, ai_agent=True):
    if ai_agent == True:
        # Assign memory
        memory = choose_memory(llm, query_body.memory, query_body.memory_arg)
        
        memory = add_history_to_memory(query_body.history_memory, memory)

        # Ask the llm
        response = agent_executor.invoke(
            {"input": query_body.query, "history": memory.chat_memory.messages}
        )

        # Get the Answer
        output = response["output"]
        
        # Convert memory to string
        chat_history = convert_memory_to_string(memory, query_body.query, output)
        
        result = {"output": output, "history": chat_history}

    else:
        # Use this if without agent
        conversation = ConversationChain(llm=llm, memory=memory)
        response = conversation(query_body.query)

    return result


def input_query(query_body: ChatQuery):
    # Input the query and history to llm
    response = ask_llm(query_body)
    
    return response
