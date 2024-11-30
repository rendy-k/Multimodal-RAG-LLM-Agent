import streamlit as st
import requests
from validator import ChatQuery

URL = "http://localhost:8000/query/"

# Request the LLM
def request_llm(body: ChatQuery):
    response = requests.post(URL, json=body)
    response = response.json()

    return response


# Show history
@st.dialog("History in the memory")
def show_history_memory():
    if "history_memory" in st.session_state:
        st.markdown(st.session_state.history_memory)

def set_session_state():
    # Prepare the LLM model
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    if "history_memory" not in st.session_state:
        st.session_state.history_memory = ""


def main():
    # Sidebar
    with st.sidebar:
        # Button to show history
        show_history = st.button("Show history")

        # Setting the LLM       
        with st.form("setting"):
            temperature = st.number_input("Temperature", value=1.0, step=0.1)
            max_tokens = st.number_input("Maximum tokens", value=100, step=1)
            memory = st.selectbox(
                    "Conversation memory",
                    ["Buffer", "Buffer Window", "Summary", "Summary Buffer", "Token"],
                    index=0
                )
            memory_arg = st.number_input(
                    "Memory arguments", value=50, step=1,
                    help="Buffer Window: k\n\nSummary Buffer: max_token_limit\n\nToken: max_token_limit"
                )
            save_setting = st.form_submit_button("Save setting")

    # Application main interface
    st.title("AI Chatbot")
    st.markdown("The application provides a chatbot with conversation memory. In the backend, it has an AI Agent to call tools/functions. The tools are fine-tuned LLM, Hugging Face Pipeline, custom function, and wrapped AI.")

    # Display chats
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ask a question
    if query := st.chat_input("Ask a question"):
        # Append user question to history
        st.session_state.history.append({"role": "user", "content": query})
        # Add user question
        with st.chat_message("user"):
            st.markdown(query)

        # Prepare the input and output history
        history_input = [
            message["content"] for message in st.session_state.history[:-1] if message["role"]=="user"
        ] # The last content in the query is not history
        
        history_output = [
            message["content"] for message in st.session_state.history if message["role"]=="assistant"
        ]
        
        # Request the LLM
        body = {
            "query": query,
            "history_input": history_input,
            "history_output": history_output,
            "temperature": round(temperature, 3),
            "max_tokens": max_tokens,
            "memory": memory,
            "memory_arg": memory_arg,
        }

        response = request_llm(body)
        with st.chat_message("assistant"):
            st.write(response["response"])

        # Append assistant answer to history
        st.session_state.history.append({"role": "assistant", "content": response["response"]})

        # Extract the latest history memory
        st.session_state.history_memory = response["history"]

    # Show history if the button is clicked
    if show_history:
        show_history_memory()


if __name__ == "__main__":
    set_session_state()

    main()
    