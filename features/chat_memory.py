from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory,
)


def choose_memory(llm, input_memory: str, memory_arg: int):
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


def convert_memory_to_string(memory, query, output):
    chat_history = ""
    for message in memory.chat_memory.messages:
        role = type(message).__name__.replace("Message", "")
        content = message.content
        chat_history = chat_history + f"{role}: {content}\n"
    chat_history = chat_history[:-1]

    # Add new chat history
    chat_history = chat_history + "\nHuman: " + query + "\nAI: " + output
    
    return chat_history
