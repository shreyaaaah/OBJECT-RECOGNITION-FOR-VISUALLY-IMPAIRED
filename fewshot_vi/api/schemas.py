"""Pydantic schemas for the REST API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DirectionSchema(BaseModel):
    horizontal: str
    vertical: str
    pan: float


class DetectionItemSchema(BaseModel):
    box: List[float] = Field(..., description="[x1, y1, x2, y2]")
    name: Optional[str] = None
    object_id: Optional[int] = None
    score: float
    margin: float
    decision: str
    direction: DirectionSchema
    distance_m: float
    distance_phrase: str
    spoken_phrase: str


class BeepSchema(BaseModel):
    enabled: bool = False
    interval_ms: int = 500
    freq_hz: int = 440
    pan: float = 0.0


class FindStatusSchema(BaseModel):
    target: Optional[str] = None
    found: bool = False
    distance_m: Optional[float] = None
    direction: Optional[DirectionSchema] = None
    spoken_phrase: Optional[str] = None
    beep: BeepSchema = Field(default_factory=BeepSchema)


class RecognizeResponse(BaseModel):
    detections: List[DetectionItemSchema] = Field(default_factory=list)
    spoken_announcements: List[str] = Field(default_factory=list)
    guidance_hint: Optional[str] = None
    find_status: Optional[FindStatusSchema] = None
    fps: Optional[float] = None


class ObjectItemSchema(BaseModel):
    id: int
    name: str
    created_at: float
    embedding_count: int
    confuser_group: Optional[int] = None
    last_seen_ts: Optional[float] = None
    last_seen_phrase: Optional[str] = None
    location: Optional[str] = None


class RenameObjectRequest(BaseModel):
    name: str


class ConfuserRequest(BaseModel):
    confuser_name: str


class BatchRegisterItem(BaseModel):
    name: str
    confuser_of: Optional[str] = None
    # Base64 video or images
    video_base64: Optional[str] = None
    images_base64: Optional[List[str]] = None


class MemoryItemSchema(BaseModel):
    object_id: int
    name: str
    last_seen_ts: Optional[float] = None
    time_ago: Optional[str] = None
    location: Optional[str] = None
    confidence: Optional[float] = None
    spoken_answer: str


class AdapterStatusResponse(BaseModel):
    active: bool
    num_classes: int
    total_embeddings: int
    in_dim: int
    out_dim: int
    margin: float
    intra_class_sim: float
    inter_class_sim: float
