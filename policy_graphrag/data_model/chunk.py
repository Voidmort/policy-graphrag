from typing import Any
from pydantic import BaseModel

class Chunk(BaseModel):
    id: str
    content: str
    document_id: str
    file_name: str
    


