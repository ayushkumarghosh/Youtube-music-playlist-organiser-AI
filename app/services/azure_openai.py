"""Azure OpenAI classification service."""

from __future__ import annotations

import asyncio
from hashlib import sha256
import json
from typing import Any

from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from app.constants import (
    CLASSIFICATION_BATCH_MAX_SONGS,
    CLASSIFICATION_BATCH_MIN_SONGS,
    CLASSIFICATION_DESCRIPTION_CHAR_LIMIT,
    CLASSIFICATION_INPUT_SOFT_TOKEN_BUDGET,
    CLASSIFICATION_RETRY_ATTEMPTS,
    CLASSIFICATION_MAX_OUTPUT_TOKENS,
    CLASSIFICATION_OUTPUT_TOKEN_RESERVE_PER_SONG,
    PROMPT_VERSION,
    SYSTEM_REASONING_EFFORT,
)
from app.db import Database
from app.models import (
    BatchCategoryClassificationItem,
    BatchCategoryClassificationResponse,
    BatchMoodClassificationResponse,
    CategoryAssignment,
    CategorySetDefinition,
    CustomCategoryProposalResponse,
    MoodClassification,
    SetupSettings,
    SongCandidate,
    SongCategoryClassification,
    category_sets_hash,
    default_mood_category_set,
    mood_assignment_from_values,
    utc_now,
)


SYSTEM_PROMPT = """You classify YouTube videos into user-selected playlist categories in bulk.
Use only the metadata and category definitions provided.
Return JSON that matches the schema exactly.
Keep the response compact and valid JSON only.
Rules:
- If the item does not look like music/song content, set is_music to false and assignments to [].
- If the metadata is too weak or ambiguous, set is_music to false and explain through empty assignments.
- For music, evaluate every provided category set independently.
- Use category_id values exactly as provided.
- Use only label_slugs from the provided category definitions.
- Each assignment must include category_id, label_slugs, confidence, and reason.
- label_slugs may contain more than one strong-fit label, but keep labels selective.
- If a category has no clearly supported label for a music item, include that category with label_slugs [].
- Confidence must be an integer from 0 to 100.
- reason must be concise, grounded in metadata, and preferably under 12 words.
- Return exactly one result item for every input song.
- Do not omit, duplicate, or invent video_id values.
- Do not include markdown, prose, or code fences.
"""

CUSTOM_CATEGORY_PROMPT = """You design reusable playlist labels for a user-defined music organization category.
Return JSON that matches the schema exactly.
Rules:
- Use the user's category name, free-form prompt, and requested target count.
- Create concise playlist label names that are useful for classifying songs.
- Each label needs a short guidance description.
- Labels must be distinct, non-overlapping where practical, and understandable without the prompt.
- Do not include markdown, prose, or code fences.
"""


class AzureClassificationError(RuntimeError):
    """Raised when Azure OpenAI validation or connectivity fails."""


def normalize_category_sets(category_sets: list[CategorySetDefinition] | None) -> list[CategorySetDefinition]:
    return category_sets or [default_mood_category_set()]


def build_cache_key(
    candidate: SongCandidate,
    category_sets: list[CategorySetDefinition] | None = None,
) -> str:
    raw = f"{candidate.video_id}:{candidate.metadata_hash}:{category_sets_hash(normalize_category_sets(category_sets))}:{PROMPT_VERSION}"
    return sha256(raw.encode("utf-8")).hexdigest()


def serialize_candidate_for_batch(candidate: SongCandidate) -> dict[str, Any]:
    return {
        "video_id": candidate.video_id,
        "title": candidate.title,
        "channel_title": candidate.channel_title,
        "description": candidate.description[:CLASSIFICATION_DESCRIPTION_CHAR_LIMIT],
        "source_playlists": candidate.source_playlists,
        "source_positions": candidate.source_positions,
    }


def serialize_category_for_batch(category: CategorySetDefinition) -> dict[str, Any]:
    return {
        "id": category.id,
        "name": category.name,
        "description": category.description,
        "prompt": category.prompt,
        "labels": [
            {
                "slug": label.slug,
                "name": label.name,
                "description": label.description,
            }
            for label in category.labels
        ],
    }


def estimate_serialized_tokens(payload: Any) -> int:
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return max(1, (len(serialized) + 3) // 4)


class AzureOpenAIClassifier:
    def __init__(self, settings: SetupSettings, db: Database) -> None:
        self.settings = settings
        self.db = db
        base_url = settings.azure_openai_endpoint.rstrip("/") + "/openai/v1/"
        self.async_client = AsyncOpenAI(
            api_key=settings.azure_openai_api_key,
            base_url=base_url,
        )
        self.sync_client = OpenAI(
            api_key=settings.azure_openai_api_key,
            base_url=base_url,
        )

    def probe(self) -> None:
        try:
            profile = self._build_attempt_profiles(batch_size=1)[0]
            response = self.sync_client.responses.parse(
                **self._build_batch_request_kwargs(
                    [
                        SongCandidate(
                            video_id="probe-video",
                            title="Probe song",
                            channel_title="Probe artist",
                            description="Warm upbeat song for connectivity verification.",
                            source_playlists=["Probe"],
                            source_playlist_ids=["probe-playlist"],
                            source_positions=[0],
                        )
                    ],
                    profile,
                    [default_mood_category_set()],
                )
            )
            if response.output_parsed is None:
                raise AzureClassificationError("Azure OpenAI returned an empty probe response.")
        except Exception as exc:  # pragma: no cover - network error path
            raise AzureClassificationError(
                "Unable to use the configured Azure OpenAI deployment with the current Responses API settings."
            ) from exc

    def propose_custom_category_labels(
        self,
        name: str,
        prompt: str,
        target_count: int,
    ) -> CustomCategoryProposalResponse:
        try:
            response = self.sync_client.responses.parse(
                model=self.settings.azure_openai_deployment,
                instructions=CUSTOM_CATEGORY_PROMPT,
                input=json.dumps(
                    {
                        "category_name": name,
                        "prompt": prompt,
                        "target_label_count": target_count,
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                reasoning={"effort": SYSTEM_REASONING_EFFORT},
                text_format=CustomCategoryProposalResponse,
                max_output_tokens=max(CLASSIFICATION_MAX_OUTPUT_TOKENS, target_count * 80),
                text={"verbosity": "low"},
                truncation="disabled",
                store=False,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise AzureClassificationError("Azure OpenAI returned an empty custom category proposal.")
            return parsed
        except Exception as exc:  # pragma: no cover - network error path
            raise AzureClassificationError("Custom category proposal failed.") from exc

    async def classify_candidates(
        self,
        candidates: list[SongCandidate],
        category_sets: list[CategorySetDefinition] | None = None,
    ) -> dict[str, SongCategoryClassification]:
        category_sets = normalize_category_sets(category_sets)
        cached_results: dict[str, SongCategoryClassification] = {}
        uncached_candidates: list[SongCandidate] = []

        for candidate in candidates:
            cache_key = build_cache_key(candidate, category_sets)
            cached = self.db.load_cached_classification(cache_key)
            if cached:
                cached_results[candidate.video_id] = self._coerce_cached_classification(cached)
            else:
                uncached_candidates.append(candidate)

        if not uncached_candidates:
            return cached_results

        batches = self.pack_candidate_batches(uncached_candidates)
        fresh_results: dict[str, SongCategoryClassification] = {}
        processed_uncached = 0

        for batch_index, batch_candidates in enumerate(batches, start=1):
            try:
                try:
                    batch_results = await self._classify_batch_with_recovery(batch_candidates, category_sets)
                except TypeError:
                    batch_results = await self._classify_batch_with_recovery(batch_candidates)
            except AzureClassificationError as exc:
                raise AzureClassificationError(
                    f"Classification failed after {processed_uncached} of {len(uncached_candidates)} uncached songs. "
                    f"Batch {batch_index}/{len(batches)} with {len(batch_candidates)} songs failed: {exc}"
                ) from exc
            fresh_results.update(
                {
                    video_id: self._coerce_any_classification(classification)
                    for video_id, classification in batch_results.items()
                }
            )
            processed_uncached += len(batch_candidates)

        return {**cached_results, **fresh_results}

    def pack_candidate_batches(self, candidates: list[SongCandidate]) -> list[list[SongCandidate]]:
        if not candidates:
            return []
        if len(candidates) <= CLASSIFICATION_BATCH_MIN_SONGS:
            return [candidates]

        batches: list[list[SongCandidate]] = []
        current_batch: list[SongCandidate] = []
        current_tokens = 0

        for candidate in candidates:
            estimated_tokens = estimate_serialized_tokens(serialize_candidate_for_batch(candidate))
            candidate_fits_current = (
                bool(current_batch)
                and len(current_batch) >= CLASSIFICATION_BATCH_MIN_SONGS
                and (
                    len(current_batch) >= CLASSIFICATION_BATCH_MAX_SONGS
                    or current_tokens + estimated_tokens > CLASSIFICATION_INPUT_SOFT_TOKEN_BUDGET
                )
            )

            if candidate_fits_current:
                batches.append(current_batch)
                current_batch = [candidate]
                current_tokens = estimated_tokens
                continue

            current_batch.append(candidate)
            current_tokens += estimated_tokens

        if current_batch:
            batches.append(current_batch)
        return batches

    async def _classify_batch_with_recovery(
        self,
        candidates: list[SongCandidate],
        category_sets: list[CategorySetDefinition] | None = None,
    ) -> dict[str, SongCategoryClassification]:
        category_sets = normalize_category_sets(category_sets)
        last_error: Exception | None = None
        for attempt, profile in enumerate(self._build_attempt_profiles(len(candidates)), start=1):
            try:
                try:
                    response = await self._request_batch_response(candidates, profile, category_sets)
                except TypeError:
                    response = await self._request_batch_response(candidates, profile)
                items_by_id = self._validate_batch_response(candidates, response, category_sets)
                return self._persist_batch_results(candidates, items_by_id, category_sets)
            except (AzureClassificationError, ValidationError, Exception) as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt < CLASSIFICATION_RETRY_ATTEMPTS:
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue

        if len(candidates) == 1:
            raise AzureClassificationError(
                f"Single-song fallback failed for video_id={candidates[0].video_id}."
            ) from last_error

        midpoint = len(candidates) // 2
        left_results = await self._classify_batch_with_recovery(candidates[:midpoint], category_sets)
        right_results = await self._classify_batch_with_recovery(candidates[midpoint:], category_sets)
        return {**left_results, **right_results}

    def _build_attempt_profiles(self, batch_size: int) -> list[dict[str, Any]]:
        base_output_budget = max(
            CLASSIFICATION_MAX_OUTPUT_TOKENS,
            batch_size * CLASSIFICATION_OUTPUT_TOKEN_RESERVE_PER_SONG,
        )
        return [
            {
                "reasoning": {"effort": SYSTEM_REASONING_EFFORT},
                "max_output_tokens": base_output_budget,
                "verbosity": "low",
            },
            {
                "reasoning": {"effort": "low"},
                "max_output_tokens": base_output_budget * 2,
                "verbosity": "low",
            },
            {
                "reasoning": {"effort": "low"},
                "max_output_tokens": base_output_budget * 3,
                "verbosity": "low",
            },
        ]

    async def _request_batch_response(
        self,
        candidates: list[SongCandidate],
        profile: dict[str, Any],
        category_sets: list[CategorySetDefinition] | None = None,
    ) -> BatchCategoryClassificationResponse:
        response = await self.async_client.responses.parse(
            **self._build_batch_request_kwargs(candidates, profile, category_sets)
        )
        parsed = response.output_parsed
        if parsed is None:
            raise AzureClassificationError("Azure OpenAI returned an empty batch classification.")
        return parsed

    def _build_batch_request_kwargs(
        self,
        candidates: list[SongCandidate],
        profile: dict[str, Any],
        category_sets: list[CategorySetDefinition] | None = None,
    ) -> dict[str, Any]:
        category_sets = normalize_category_sets(category_sets)
        song_payload = [serialize_candidate_for_batch(candidate) for candidate in candidates]
        category_payload = [serialize_category_for_batch(category) for category in category_sets]
        return {
            "model": self.settings.azure_openai_deployment,
            "instructions": SYSTEM_PROMPT,
            "input": json.dumps(
                {"songs": song_payload, "category_sets": category_payload},
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            "reasoning": profile["reasoning"],
            "text_format": BatchCategoryClassificationResponse,
            "max_output_tokens": profile["max_output_tokens"],
            "text": {"verbosity": profile["verbosity"]},
            "truncation": "disabled",
            "store": False,
        }

    def _validate_batch_response(
        self,
        candidates: list[SongCandidate],
        response: BatchCategoryClassificationResponse | BatchMoodClassificationResponse,
        category_sets: list[CategorySetDefinition] | None = None,
    ) -> dict[str, BatchCategoryClassificationItem]:
        category_sets = normalize_category_sets(category_sets)
        allowed_categories = {category.id: category for category in category_sets}
        requested_ids = [candidate.video_id for candidate in candidates]
        requested_id_set = set(requested_ids)
        response_items = [self._coerce_response_item(item) for item in response.items]
        response_ids = [item.video_id for item in response_items]
        response_id_set = set(response_ids)

        duplicate_ids = sorted({video_id for video_id in response_ids if response_ids.count(video_id) > 1})
        missing_ids = sorted(requested_id_set - response_id_set)
        extra_ids = sorted(response_id_set - requested_id_set)

        if duplicate_ids or missing_ids or extra_ids or len(response_items) != len(candidates):
            raise AzureClassificationError(
                "Batch response validation failed. "
                f"missing_ids={missing_ids[:5]} extra_ids={extra_ids[:5]} duplicate_ids={duplicate_ids[:5]}"
            )

        normalized_items: dict[str, BatchCategoryClassificationItem] = {}
        for item in response_items:
            assignments_by_category: dict[str, CategoryAssignment] = {}
            for assignment in item.assignments:
                category = allowed_categories.get(assignment.category_id)
                if category is None:
                    raise AzureClassificationError(
                        f"Batch response included unknown category_id={assignment.category_id}."
                    )
                allowed_label_slugs = {label.slug for label in category.labels}
                unknown_labels = sorted(set(assignment.label_slugs) - allowed_label_slugs)
                if unknown_labels:
                    raise AzureClassificationError(
                        f"Batch response included unknown labels for {category.id}: {unknown_labels[:5]}"
                    )
                if assignment.category_id in assignments_by_category:
                    raise AzureClassificationError(
                        f"Batch response duplicated category_id={assignment.category_id}."
                    )
                assignments_by_category[assignment.category_id] = assignment

            if item.is_music:
                for category in category_sets:
                    assignments_by_category.setdefault(
                        category.id,
                        CategoryAssignment(
                            category_id=category.id,
                            label_slugs=[],
                            confidence=0,
                            reason="No confident label.",
                        ),
                    )
            else:
                assignments_by_category = {}

            normalized_items[item.video_id] = item.model_copy(
                update={"assignments": list(assignments_by_category.values())}
            )

        return normalized_items

    def _persist_batch_results(
        self,
        candidates: list[SongCandidate],
        items_by_id: dict[str, BatchCategoryClassificationItem],
        category_sets: list[CategorySetDefinition],
    ) -> dict[str, SongCategoryClassification]:
        classifications: dict[str, SongCategoryClassification] = {}
        for candidate in candidates:
            response_item = items_by_id[candidate.video_id]
            classification = SongCategoryClassification(
                is_music=response_item.is_music,
                assignments=response_item.assignments,
                model_name=self.settings.azure_openai_deployment,
                prompt_version=PROMPT_VERSION,
            )
            cache_key = build_cache_key(candidate, category_sets)
            self.db.save_cached_classification(
                cache_key=cache_key,
                video_id=candidate.video_id,
                metadata_hash=candidate.metadata_hash,
                prompt_version=PROMPT_VERSION,
                payload=classification.model_dump(mode="json"),
                updated_at=utc_now(),
            )
            classifications[candidate.video_id] = classification
        return classifications

    def _coerce_cached_classification(self, cached: dict[str, Any]) -> SongCategoryClassification:
        if "assignments" in cached:
            return SongCategoryClassification.model_validate(cached)
        return MoodClassification.model_validate(cached).to_category_classification()

    def _coerce_any_classification(self, classification: Any) -> SongCategoryClassification:
        if isinstance(classification, SongCategoryClassification):
            return classification
        if isinstance(classification, MoodClassification):
            return classification.to_category_classification()
        if hasattr(classification, "assignments"):
            if hasattr(classification, "model_dump"):
                return SongCategoryClassification.model_validate(classification.model_dump(mode="json"))
            return SongCategoryClassification.model_validate(classification)
        if hasattr(classification, "moods"):
            return MoodClassification(
                is_music=bool(classification.is_music),
                moods=classification.moods,
                confidence=int(getattr(classification, "confidence", 0) or 0),
                reason=str(getattr(classification, "reason", "")),
                model_name=str(getattr(classification, "model_name", self.settings.azure_openai_deployment)),
                prompt_version=str(getattr(classification, "prompt_version", PROMPT_VERSION)),
            ).to_category_classification()
        return SongCategoryClassification.model_validate(classification)

    def _coerce_response_item(self, item: Any) -> BatchCategoryClassificationItem:
        if isinstance(item, BatchCategoryClassificationItem):
            return item
        if hasattr(item, "assignments"):
            return BatchCategoryClassificationItem.model_validate(item.model_dump(mode="json"))
        if hasattr(item, "moods"):
            assignment = mood_assignment_from_values(
                getattr(item, "moods"),
                confidence=int(getattr(item, "confidence", 0) or 0),
                reason=str(getattr(item, "reason", "")),
            )
            return BatchCategoryClassificationItem(
                video_id=item.video_id,
                is_music=bool(item.is_music),
                assignments=[assignment] if item.is_music else [],
            )
        return BatchCategoryClassificationItem.model_validate(item)
