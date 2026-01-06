from pydantic import BaseModel, Field
from typing import Literal


class QueryParam(BaseModel):
    mode: Literal["graph", "naive"] = "graph"
    only_need_context: bool = False
    level: int = 2
    top_k: int = 10
    threshold: float = 0.8

    # local search
    local_max_token_for_text_unit: int = Field(4000, description="12000 * 0.33")
    local_max_token_for_local_context: int = Field(4800, description="12000 * 0.4")
    local_max_token_for_community_report: int = Field(3200, description="12000 * 0.27")
    local_community_single_one: bool = False

    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    global_special_community_map_llm_kwargs: dict = Field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
