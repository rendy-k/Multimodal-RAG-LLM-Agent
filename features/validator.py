from pydantic import BaseModel
from typing import Optional

class ChatQuery(BaseModel):
    query: str
    history_memory: Optional[str] = ""
    memory: Optional[str] = "Buffer Window"
    memory_arg: Optional[int] = 3
    