from typing import Optional

from pydantic import BaseModel


class IssueDto(BaseModel):
    id: int = None
    description: str
    categoryId: int
    location: dict[str, float]
    status: str
    priority: str
    reporterId: str
    assigneeId: Optional[int] = None
    createdAt: int
    updatedAt: int
    documentPath: Optional[str]
    contentType: Optional[str]
    actualityStatus: str = None
    isDescriptionByVoice: bool
    voiceDescriptionId: int


class ResponseDto(BaseModel):
    prediction: float
