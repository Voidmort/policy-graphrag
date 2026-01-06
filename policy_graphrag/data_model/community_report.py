from __future__ import annotations
from typing import List
from pydantic import BaseModel
from policy_graphrag.utils import community_report_json_to_str


class Finding(BaseModel):
    summary: str
    explanation: str


class ReportJson(BaseModel):
    title: str
    summary: str
    rating: float
    rating_explanation: str
    findings: List[Finding]


class CommunityReport(BaseModel):
    community_id: str = ""
    title: str
    level: int
    edges: List[List[str]]
    nodes: List[str]
    text_unit_ids: List[str]
    occurrence: float
    summary: str = ""
    rating: float = 0
    rating_explanation: str = ""
    findings: List[Finding] = []
    sub_communities: List = []

    @property
    def report_string(self):
        return community_report_json_to_str(self.model_dump())
