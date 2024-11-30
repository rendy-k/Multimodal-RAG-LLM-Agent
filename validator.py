from pydantic import BaseModel

class ChatQuery(BaseModel):
    query: str
    history_memory: str
    temperature: float
    max_tokens: int
    memory: str
    memory_arg: int
    