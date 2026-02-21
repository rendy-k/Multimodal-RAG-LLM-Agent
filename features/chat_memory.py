from langchain_core.runnables.history import RunnableWithMessageHistory
import re


def prepare_memory(history_memory):
    history_memory = "\n" + history_memory
    pattern = r'(Human|AI):\s*(.*?)(?=(\nHuman:|\nAI:|$))'
    matches = re.findall(pattern, history_memory, flags=re.DOTALL)

    # Map roles and strip extra spaces
    conversation = []
    role_map = {'Human': 'user', 'AI': 'assistant'}
    
    for role, content, _ in matches:
        conversation.append((role_map[role], content.strip()))
    
    return conversation


def update_memory(history_memory, query, output):
    if history_memory != "":
        history_memory += "\n"
    human_message = f"Human: {query}"
    ai_message = f"\nAI: {output}"

    string_memory = history_memory + human_message + ai_message
    return string_memory
