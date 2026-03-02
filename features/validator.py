from pydantic import BaseModel
from typing import Optional

class ChatQuery(BaseModel):
    query: str
    history_memory: Optional[str] = ""

    