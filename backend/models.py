from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime


class SourceRef(BaseModel):
    url: str
    title: str = ""
    snippet: str = "" 

class CellValue(BaseModel):
    value: Any
    sources: list[SourceRef] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Entity(BaseModel):
    id: str
    cells: dict[str, CellValue]  # column_name to CellValue

    def get_name(self) -> str:
        cell = self.cells.get("name")
        return str(cell.value) if cell else self.id


class SearchPlan(BaseModel):
    entity_type: str
    columns: list[str]        
    search_queries: list[str] 
    rationale: str
    rules: dict = Field(default_factory=dict)


class ScrapedPage(BaseModel):
    url: str
    title: str = ""
    content: str


class RankingInfo(BaseModel):
    total: int                        # total number of constraints extracted
    labels: list[str]                 # human-readable constraint descriptions
    scores: dict[str, int]           # entity_id is n constraints satisfied


# Final output
class EntityTable(BaseModel):
    query: str
    entity_type: str
    columns: list[str]
    entities: list[Entity]
    sources_consulted: list[str]
    search_queries_used: list[str]
    rounds_completed: int
    ranking: Optional[RankingInfo] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
