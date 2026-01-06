import json

from typing import Any, List
from pydantic import BaseModel

from .. import utils


class Edge(BaseModel):
    source: str
    """The source entity name."""

    target: str
    """The target entity name."""

    weight: float | None = 1.0
    """The edge weight."""

    description: str | None = None
    """A description of the relationship (optional)."""

    text_unit_ids: list[str] = []
    """List of text unit IDs in which the relationship appears (optional)."""

    rank: int | None = 1
    """Rank of the relationship, used for sorting (optional). Higher rank indicates more important relationship. This can be based on centrality or other metrics."""

    attributes: dict[str, Any] | None = None
    """Additional attributes associated with the relationship (optional). To be included in the search prompt"""

    @property
    def id(self):
        return utils.compute_mdhash_id(f"{self.source}-{self.description}-{self.target}")
    
    @property
    def source_id(self):
        return utils.compute_mdhash_id(self.source)
    
    @property
    def target_id(self):
        return utils.compute_mdhash_id(self.target)
    
    @property
    def edge_string(self):
        data = {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "description": self.description,
            "rank": self.rank
        }
        return json.dumps(data, ensure_ascii=False)