from pydantic import BaseModel


class ContentID(BaseModel):
    id: str
    content: str
