from pydantic import BaseModel

class ChatQuery(BaseModel):
    query: str
    history_input: list
    history_output: list
    temperature: float
    max_tokens: int
    memory: str
    memory_arg: int
    