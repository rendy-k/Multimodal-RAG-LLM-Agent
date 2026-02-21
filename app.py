import streamlit as st
import requests
import pandas as pd
import json
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

# Show reasoning
@st.dialog("AI reasoning")
def show_ai_reasoning():
    if "ai_reasoning" in st.session_state:
        st.write(st.session_state.ai_reasoning.replace("\n", "\n\n"))

# Load booking data
def load_booking_data():
    booking_data = pd.read_csv("database/booking_status.csv")
    booking_data = booking_data.to_json(orient="records")
    booking_data = json.loads(booking_data)
    
    status_text = ""
    for item in booking_data:
        for key, value in item.items():
            if value is not None:
                if key != "name":
                    status_text += f"- {key}: {value}\n"
                else:
                    status_text += f"**{key}: {value}**\n"
        status_text += "\n\n"
    st.session_state.booking = status_text


def set_session_state():
    # Prepare the LLM model
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    if "history_memory" not in st.session_state:
        st.session_state.history_memory = ""

    if "ai_reasoning" not in st.session_state:
        st.session_state.ai_reasoning = ""

    if "booking" not in st.session_state:
        st.session_state.booking = ""


def main():
    # Sidebar
    with st.sidebar:
        # Button to show history
        show_history = st.button("Show history")

        # Button to show history
        show_reasoning = st.button("Show reasoning")

        # Status       
        st.markdown("### Status")
        with st.expander("Booking status"):
            st.markdown(st.session_state.booking)

        refresh_status = st.button("Refresh")

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

        # Append AI reasonings
        st.session_state.ai_reasoning += (
            f"HUMAN: {query}\n\nAI REASONING: "
            + response["reasoning"]
            + "\n\nAI: " + {response['output']}
            + "\n\n==========\n\n"
        )

    # Show history if the button is clicked
    if show_history:
        show_history_memory()
    
    # Show reasoning if the button is click
    if show_reasoning:
        show_ai_reasoning()

    # Refresh loading status data
    if refresh_status:
        load_booking_data()

if __name__ == "__main__":
    set_session_state()

    main()
    