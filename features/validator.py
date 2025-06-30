from pydantic import BaseModel
from typing import Optional

class ChatQuery(BaseModel):
    query: str
    history_memory: Optional[str] = ""
    temperature: Optional[float] = 0.5
    max_tokens: Optional[int] = 30
    memory: Optional[str] = "Buffer Window"
    memory_arg: Optional[int] = 3
    