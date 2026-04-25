"""Core application models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


def normalize_mood_labels(values: Any) -> list[MoodLabel]:
    if values in (None, "", []):
        return []
    if isinstance(values, (MoodLabel, str)):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        raise TypeError("Moods must be a mood label or a list of mood labels.")

    normalized_values: set[str] = set()
    for value in values:
        if value in (None, ""):
            continue
        mood = value if isinstance(value, MoodLabel) else MoodLabel(str(value))
        normalized_values.add(mood.value)

    return [MoodLabel(label) for label in MOOD_LABELS if label in normalized_values]


def serialize_mood_labels(values: Any) -> str:
    return json.dumps([mood.value for mood in normalize_mood_labels(values)])


def deserialize_mood_labels(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        decoded = raw
    return [mood.value for mood in normalize_mood_labels(decoded)]


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
    moods: list[MoodLabel] = Field(
        default_factory=list,
        description="Strong-fit moods when the item is music and there is enough metadata.",
    )
    confidence: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=300)
    model_name: str = Field(default="")
    prompt_version: str = Field(default=PROMPT_VERSION)

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_mood_field(cls, data: Any) -> Any:
        if isinstance(data, dict) and "moods" not in data and "mood" in data:
            legacy_mood = data.get("mood")
            data = dict(data)
            data["moods"] = [] if legacy_mood in (None, "") else [legacy_mood]
        return data

    @field_validator("moods", mode="before")
    @classmethod
    def normalize_moods(cls, value: Any) -> list[MoodLabel]:
        return normalize_mood_labels(value)

    @model_validator(mode="after")
    def validate_moods(self) -> "MoodClassification":
        if self.is_music and not self.moods:
            raise ValueError("Music rows must have at least one mood.")
        if not self.is_music and self.moods:
            raise ValueError("Non-music rows must not include moods.")
        return self

    @property
    def mood(self) -> MoodLabel | None:
        return self.moods[0] if self.moods else None


class MoodClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_music: bool
    moods: list[MoodLabel] = Field(default_factory=list)
    confidence: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_mood_field(cls, data: Any) -> Any:
        if isinstance(data, dict) and "moods" not in data and "mood" in data:
            legacy_mood = data.get("mood")
            data = dict(data)
            data["moods"] = [] if legacy_mood in (None, "") else [legacy_mood]
        return data

    @field_validator("moods", mode="before")
    @classmethod
    def normalize_moods(cls, value: Any) -> list[MoodLabel]:
        return normalize_mood_labels(value)

    @model_validator(mode="after")
    def validate_music_moods(self) -> "MoodClassificationResponse":
        if self.is_music and not self.moods:
            raise ValueError("Music rows must include at least one mood.")
        if not self.is_music and self.moods:
            raise ValueError("Non-music rows must not include moods.")
        return self

    @property
    def mood(self) -> MoodLabel | None:
        return self.moods[0] if self.moods else None


class BatchMoodClassificationItem(MoodClassificationResponse):
    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)


class BatchMoodClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[BatchMoodClassificationItem]


class ApprovedAssignment(BaseModel):
    video_id: str
    final_moods: list[MoodLabel] = Field(default_factory=list)
    source_scope: RunScope
    override_applied: bool = False

    @field_validator("final_moods", mode="before")
    @classmethod
    def normalize_final_moods(cls, value: Any) -> list[MoodLabel]:
        return normalize_mood_labels(value)

    @property
    def final_mood(self) -> MoodLabel | None:
        return self.final_moods[0] if self.final_moods else None


class SetupSettings(BaseModel):
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = ""
    google_client_secrets_json: str = ""
    session_secret: str = ""
    app_base_url: str = ""

    def masked(self) -> "SetupSettings":
        return SetupSettings(
            azure_openai_endpoint=self.azure_openai_endpoint,
            azure_openai_api_key="********" if self.azure_openai_api_key else "",
            azure_openai_deployment=self.azure_openai_deployment,
            google_client_secrets_json="********" if self.google_client_secrets_json else "",
            session_secret="********" if self.session_secret else "",
            app_base_url=self.app_base_url,
        )

    def is_complete(self) -> bool:
        return all(
            [
                self.azure_openai_endpoint.strip(),
                self.azure_openai_api_key.strip(),
                self.azure_openai_deployment.strip(),
                self.google_client_secrets_json.strip(),
                self.session_secret.strip(),
            ]
        )


class RunItemView(BaseModel):
    video_id: str
    title: str
    channel_title: str
    description: str
    source_playlists: list[str]
    source_positions: list[int]
    suggested_moods: list[MoodLabel] = Field(default_factory=list)
    final_moods: list[MoodLabel] = Field(default_factory=list)
    confidence: int
    reason: str
    is_music: bool
    default_included: bool
    override_applied: bool = False

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_mood_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        upgraded = dict(data)
        if "suggested_moods" not in upgraded and "suggested_mood" in upgraded:
            legacy_suggested = upgraded.get("suggested_mood")
            upgraded["suggested_moods"] = [] if legacy_suggested in (None, "") else [legacy_suggested]
        if "final_moods" not in upgraded and "final_mood" in upgraded:
            legacy_final = upgraded.get("final_mood")
            upgraded["final_moods"] = [] if legacy_final in (None, "") else [legacy_final]
        return upgraded

    @field_validator("suggested_moods", "final_moods", mode="before")
    @classmethod
    def normalize_moods(cls, value: Any) -> list[MoodLabel]:
        return normalize_mood_labels(value)

    @property
    def suggested_mood(self) -> MoodLabel | None:
        return self.suggested_moods[0] if self.suggested_moods else None

    @property
    def final_mood(self) -> MoodLabel | None:
        return self.final_moods[0] if self.final_moods else None


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
