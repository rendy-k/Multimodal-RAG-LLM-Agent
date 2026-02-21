import streamlit as st
import requests
from features.validator import ChatQuery

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
        st.write(st.session_state.history_memory.replace("\n", "\n\n"))

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

        # Button to show history
        show_reasoning = st.button("Show reasonings")

        # Setting the LLM       
        st.markdown("### Status")
        


    # Application main interface
    st.title("Travel Chatbot")

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

       
        # Request the LLM
        body = {
            "query": query,
            "history_memory": st.session_state.history_memory,
        }

        response = request_llm(body)
        with st.chat_message("assistant"):
            st.write(response["output"])

        # Append assistant answer to history
        st.session_state.history.append({"role": "assistant", "content": response["output"]})

        # Extract the latest history memory
        st.session_state.history_memory = response["history"]

    # Show history if the button is clicked
    if show_history:
        show_history_memory()
    
    if show_reasoning:
        pass


if __name__ == "__main__":
    set_session_state()

    main()
    