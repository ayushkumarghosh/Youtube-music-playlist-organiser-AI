"""Core application models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.constants import BUILT_IN_CATEGORY_SETS, CATEGORY_SET_MOOD_ID, MOOD_LABELS, PROMPT_VERSION


SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_label(value: str) -> str:
    slug = SLUG_PATTERN.sub("-", value.strip().lower()).strip("-")
    return slug or "label"


class RunScope(StrEnum):
    ALL_PLAYLISTS = "all_playlists"
    SINGLE_PLAYLIST = "single_playlist"
    SELECTED_PLAYLISTS = "selected_playlists"


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


class CategoryLabelDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(min_length=1, max_length=90)
    name: str = Field(min_length=1, max_length=90)
    description: str = Field(default="", max_length=300)

    @field_validator("slug", "name", "description", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("slug")
    @classmethod
    def normalize_slug(cls, value: str) -> str:
        return slugify_label(value)


class CategorySetDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=90)
    description: str = Field(default="", max_length=300)
    labels: list[CategoryLabelDefinition] = Field(min_length=1)
    source: str = Field(default="builtin")
    prompt: str = Field(default="", max_length=1500)
    archived: bool = False
    created_at: str = ""
    updated_at: str = ""

    @field_validator("id", "name", "description", "source", "prompt", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return slugify_label(value)

    @model_validator(mode="after")
    def validate_labels(self) -> "CategorySetDefinition":
        slugs = [label.slug for label in self.labels]
        names = [label.name.casefold() for label in self.labels]
        if len(set(slugs)) != len(slugs):
            raise ValueError("Category label slugs must be unique.")
        if len(set(names)) != len(names):
            raise ValueError("Category label names must be unique.")
        return self

    def label_for_slug(self, slug: str) -> CategoryLabelDefinition | None:
        normalized = slugify_label(slug)
        return next((label for label in self.labels if label.slug == normalized), None)

    def label_names_for_slugs(self, slugs: list[str]) -> list[str]:
        names = []
        for slug in slugs:
            label = self.label_for_slug(slug)
            if label is not None:
                names.append(label.name)
        return names

    @property
    def definition_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"created_at", "updated_at"})
        return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class CategoryAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category_id: str = Field(min_length=1)
    label_slugs: list[str] = Field(default_factory=list)
    confidence: int = Field(default=0, ge=0, le=100)
    reason: str = Field(default="", max_length=300)

    @field_validator("category_id", mode="before")
    @classmethod
    def normalize_category_id(cls, value: Any) -> str:
        return slugify_label(str(value))

    @field_validator("label_slugs", mode="before")
    @classmethod
    def normalize_label_input(cls, value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise TypeError("label_slugs must be a string or list of strings.")
        return [slugify_label(str(item)) for item in value if str(item).strip()]

    @field_validator("label_slugs")
    @classmethod
    def dedupe_labels(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))


def category_set_from_data(data: dict[str, Any]) -> CategorySetDefinition:
    return CategorySetDefinition.model_validate(data)


def built_in_category_sets() -> list[CategorySetDefinition]:
    return [category_set_from_data(data) for data in BUILT_IN_CATEGORY_SETS]


def default_mood_category_set() -> CategorySetDefinition:
    return built_in_category_sets()[0]


def find_category_set(category_sets: list[CategorySetDefinition], category_id: str) -> CategorySetDefinition | None:
    normalized = slugify_label(category_id)
    return next((category for category in category_sets if category.id == normalized), None)


def category_sets_hash(category_sets: list[CategorySetDefinition]) -> str:
    payload = [category.model_dump(mode="json", exclude={"created_at", "updated_at"}) for category in category_sets]
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


MOOD_CATEGORY = default_mood_category_set()
MOOD_NAME_TO_SLUG = {label.name: label.slug for label in MOOD_CATEGORY.labels}
MOOD_SLUG_TO_NAME = {label.slug: label.name for label in MOOD_CATEGORY.labels}


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
        text = value.value if isinstance(value, MoodLabel) else str(value)
        text = MOOD_SLUG_TO_NAME.get(slugify_label(text), text)
        mood = MoodLabel(text)
        normalized_values.add(mood.value)

    return [MoodLabel(label) for label in MOOD_LABELS if label in normalized_values]


def mood_assignment_from_values(values: Any, *, confidence: int = 0, reason: str = "") -> CategoryAssignment:
    moods = normalize_mood_labels(values)
    return CategoryAssignment(
        category_id=CATEGORY_SET_MOOD_ID,
        label_slugs=[MOOD_NAME_TO_SLUG[mood.value] for mood in moods],
        confidence=confidence,
        reason=reason,
    )


def mood_values_from_assignment(assignment: CategoryAssignment | None) -> list[MoodLabel]:
    if assignment is None or assignment.category_id != CATEGORY_SET_MOOD_ID:
        return []
    return normalize_mood_labels([MOOD_SLUG_TO_NAME.get(slug, slug) for slug in assignment.label_slugs])


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

    def to_category_classification(self) -> "SongCategoryClassification":
        assignment = mood_assignment_from_values(self.moods, confidence=self.confidence, reason=self.reason)
        return SongCategoryClassification(
            is_music=self.is_music,
            assignments=[assignment] if self.is_music else [],
            model_name=self.model_name,
            prompt_version=self.prompt_version,
        )


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


class SongCategoryClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_music: bool
    assignments: list[CategoryAssignment] = Field(default_factory=list)
    model_name: str = Field(default="")
    prompt_version: str = Field(default=PROMPT_VERSION)

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_moods(cls, data: Any) -> Any:
        if isinstance(data, dict) and "assignments" not in data and "moods" in data:
            data = dict(data)
            data["assignments"] = [
                mood_assignment_from_values(
                    data.get("moods", []),
                    confidence=int(data.get("confidence", 0) or 0),
                    reason=str(data.get("reason", "")),
                ).model_dump(mode="json")
            ]
        return data

    @property
    def moods(self) -> list[MoodLabel]:
        return mood_values_from_assignment(self.assignment_for(CATEGORY_SET_MOOD_ID))

    @property
    def confidence(self) -> int:
        if not self.assignments:
            return 0
        return max(assignment.confidence for assignment in self.assignments)

    @property
    def reason(self) -> str:
        reasons = [assignment.reason for assignment in self.assignments if assignment.reason]
        return reasons[0] if reasons else "No category matched."

    def assignment_for(self, category_id: str) -> CategoryAssignment | None:
        normalized = slugify_label(category_id)
        return next((assignment for assignment in self.assignments if assignment.category_id == normalized), None)


class BatchCategoryClassificationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    is_music: bool
    assignments: list[CategoryAssignment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_non_music(self) -> "BatchCategoryClassificationItem":
        if not self.is_music and any(assignment.label_slugs for assignment in self.assignments):
            raise ValueError("Non-music rows must not include category labels.")
        return self


class BatchCategoryClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[BatchCategoryClassificationItem]


class CustomCategoryProposalLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=90)
    description: str = Field(default="", max_length=300)

    @field_validator("name", "description", mode="before")
    @classmethod
    def strip_text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()


class CustomCategoryProposalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[CustomCategoryProposalLabel] = Field(min_length=2, max_length=12)

    @model_validator(mode="after")
    def validate_unique_names(self) -> "CustomCategoryProposalResponse":
        names = [label.name.casefold() for label in self.labels]
        if len(names) != len(set(names)):
            raise ValueError("Generated custom category labels must be unique.")
        return self


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
    suggested_assignments: list[CategoryAssignment] = Field(default_factory=list)
    final_assignments: list[CategoryAssignment] = Field(default_factory=list)
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
        if "suggested_assignments" not in upgraded:
            legacy_suggested = upgraded.get("suggested_moods", upgraded.get("suggested_mood"))
            if legacy_suggested is not None:
                upgraded["suggested_assignments"] = [
                    mood_assignment_from_values(
                        legacy_suggested,
                        confidence=int(upgraded.get("confidence", 0) or 0),
                        reason=str(upgraded.get("reason", "")),
                    ).model_dump(mode="json")
                ]
        if "final_assignments" not in upgraded:
            legacy_final = upgraded.get("final_moods", upgraded.get("final_mood"))
            if legacy_final is not None:
                upgraded["final_assignments"] = [
                    mood_assignment_from_values(
                        legacy_final,
                        confidence=int(upgraded.get("confidence", 0) or 0),
                        reason=str(upgraded.get("reason", "")),
                    ).model_dump(mode="json")
                ]
        if "final_assignments" not in upgraded and "suggested_assignments" in upgraded:
            upgraded["final_assignments"] = upgraded["suggested_assignments"]
        return upgraded

    @property
    def suggested_moods(self) -> list[MoodLabel]:
        return mood_values_from_assignment(self.suggested_assignment_for(CATEGORY_SET_MOOD_ID))

    @property
    def final_moods(self) -> list[MoodLabel]:
        return mood_values_from_assignment(self.final_assignment_for(CATEGORY_SET_MOOD_ID))

    @property
    def suggested_mood(self) -> MoodLabel | None:
        return self.suggested_moods[0] if self.suggested_moods else None

    @property
    def final_mood(self) -> MoodLabel | None:
        return self.final_moods[0] if self.final_moods else None

    def suggested_assignment_for(self, category_id: str) -> CategoryAssignment | None:
        normalized = slugify_label(category_id)
        return next((assignment for assignment in self.suggested_assignments if assignment.category_id == normalized), None)

    def final_assignment_for(self, category_id: str) -> CategoryAssignment | None:
        normalized = slugify_label(category_id)
        return next((assignment for assignment in self.final_assignments if assignment.category_id == normalized), None)

    def suggested_label_slugs(self, category_id: str) -> list[str]:
        assignment = self.suggested_assignment_for(category_id)
        return assignment.label_slugs if assignment is not None else []

    def final_label_slugs(self, category_id: str) -> list[str]:
        assignment = self.final_assignment_for(category_id)
        return assignment.label_slugs if assignment is not None else []

    def suggested_label_names(self, category: CategorySetDefinition) -> list[str]:
        return category.label_names_for_slugs(self.suggested_label_slugs(category.id))

    def final_label_names(self, category: CategorySetDefinition) -> list[str]:
        return category.label_names_for_slugs(self.final_label_slugs(category.id))


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
    category_sets: list[CategorySetDefinition] = Field(default_factory=lambda: [default_mood_category_set()])

    @model_validator(mode="before")
    @classmethod
    def default_legacy_category_sets(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("category_sets"):
            data = dict(data)
            data["category_sets"] = [default_mood_category_set().model_dump(mode="json")]
        return data

    @property
    def mood_labels(self) -> list[str]:
        return list(MOOD_LABELS)

    def category_for_id(self, category_id: str) -> CategorySetDefinition | None:
        return find_category_set(self.category_sets, category_id)
