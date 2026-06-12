from datetime import datetime
from pydantic import BaseModel


class ProjectCreate(BaseModel):
    name: str


class ProjectResponse(BaseModel):
    id: int
    name: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ProjectOverview(BaseModel):
    id: int
    name: str
    status: str
    character_count: int
    dialogue_count: int
    chapter_count: int


class UploadResult(BaseModel):
    project_id: int
    characters: int
    dialogues: int
    chapters: int
