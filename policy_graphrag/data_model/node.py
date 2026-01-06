from typing import Any, List
from pydantic import BaseModel
from .. import utils


class Cluster(BaseModel):
    level: int
    cluster: int


class Node(BaseModel):
    name: str
    entity_type: str
    description: str
    text_unit_ids: list[str] = []
    clusters: List[Cluster] = []
    rank: int = 0

    @property
    def id(self):
        return utils.compute_mdhash_id(self.name)
