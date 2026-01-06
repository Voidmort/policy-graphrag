from typing import Any
from .content_id import ContentID


class Document(ContentID):
    name: str
    text_unit_ids: list[str]
    attributes: dict[str, Any] | None = None
