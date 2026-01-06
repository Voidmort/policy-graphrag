from typing import Any, List
from pydantic import BaseModel


class Entity(BaseModel):
    """A protocol for an entity in the system."""

    id: str | None = None
    name: str

    type: str | None = None
    """Type of the entity (can be any string, optional)."""

    description: str | None = None
    """Description of the entity (optional)."""
