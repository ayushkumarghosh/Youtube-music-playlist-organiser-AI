"""Core application models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.constants import MOOD_LABELS, PROMPT_VERSION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunScope(StrEnum):
    ALL_PLAYLISTS = "all_playlists"
    SINGLE_PLAYLIST = "single_playlist"


class RunStatus(StrEnum):
    PREVIEWED = "previewed"
    APPLIED = "applied"
    FAILED = "failed"


class MoodLabel(StrEnum):
    HAPPY = "Happy / Feel-good"
    SAD = "Sad / Emotional"
    ROMANTIC = "Romantic / Love"
    CHILL = "Chill / Relaxing"
    ENERGETIC = "Energetic / Hype"
    DARK = "Dark / Intense"


@dataclass(slots=True)
class PlaylistSummary:
    playlist_id: str
    title: str
    description: str
    privacy_status: str
    item_count: int


@dataclass(slots=True)
class PlaylistItemRecord:
    playlist_item_id: str
    playlist_id: str
    playlist_title: str
    video_id: str
    title: str
    description: str
    channel_title: str
    position: int


@dataclass(slots=True)
class SongCandidate:
    video_id: str
    title: str
    channel_title: str
    description: str
    source_playlists: list[str] = field(default_factory=list)
    source_playlist_ids: list[str] = field(default_factory=list)
    source_positions: list[int] = field(default_factory=list)

    @property
    def metadata_hash(self) -> str:
        payload = {
            "video_id": self.video_id,
            "title": self.title,
            "channel_title": self.channel_title,
            "description": self.description,
            "source_playlists": self.source_playlists,
            "source_positions": self.source_positions,
        }
        return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class MoodClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_music: bool = Field(description="Whether the item appears to be a song/music track.")
    mood: MoodLabel | None = Field(
        default=None,
        description="Best-fit mood when the item is music and there is enough metadata.",
    )
    confidence: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=300)
    model_name: str = Field(default="")
    prompt_version: str = Field(default=PROMPT_VERSION)

    @field_validator("mood")
    @classmethod
    def validate_mood(cls, value: MoodLabel | None, info: Any) -> MoodLabel | None:
        is_music = info.data.get("is_music")
        if is_music and value is None:
            raise ValueError("Music rows must have a mood.")
        return value


class MoodClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_music: bool
    mood: MoodLabel | None = None
    confidence: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=300)

    @field_validator("mood")
    @classmethod
    def validate_music_mood(cls, value: MoodLabel | None, info: Any) -> MoodLabel | None:
        is_music = info.data.get("is_music")
        if is_music and value is None:
            raise ValueError("Music rows must include a mood.")
        return value


class BatchMoodClassificationItem(MoodClassificationResponse):
    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)


class BatchMoodClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[BatchMoodClassificationItem]


class ApprovedAssignment(BaseModel):
    video_id: str
    final_mood: MoodLabel | None
    source_scope: RunScope
    override_applied: bool = False


class SetupSettings(BaseModel):
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = ""
    google_client_secrets_file: str = ""
    session_secret: str = ""

    def masked(self) -> "SetupSettings":
        return SetupSettings(
            azure_openai_endpoint=self.azure_openai_endpoint,
            azure_openai_api_key="********" if self.azure_openai_api_key else "",
            azure_openai_deployment=self.azure_openai_deployment,
            google_client_secrets_file=self.google_client_secrets_file,
            session_secret="********" if self.session_secret else "",
        )

    def is_complete(self) -> bool:
        return all(
            [
                self.azure_openai_endpoint.strip(),
                self.azure_openai_api_key.strip(),
                self.azure_openai_deployment.strip(),
                self.google_client_secrets_file.strip(),
            ]
        )


class RunItemView(BaseModel):
    video_id: str
    title: str
    channel_title: str
    description: str
    source_playlists: list[str]
    source_positions: list[int]
    suggested_mood: MoodLabel | None = None
    final_mood: MoodLabel | None = None
    confidence: int
    reason: str
    is_music: bool
    default_included: bool
    override_applied: bool = False


class RunSummary(BaseModel):
    total_candidates: int
    classified_count: int
    default_included_count: int
    excluded_count: int


class RunDetail(BaseModel):
    run_id: str
    status: RunStatus
    scope: RunScope
    source_playlist_id: str | None = None
    source_playlist_title: str | None = None
    created_at: str
    summary: RunSummary
    items: list[RunItemView]

    @property
    def mood_labels(self) -> list[str]:
        return list(MOOD_LABELS)
