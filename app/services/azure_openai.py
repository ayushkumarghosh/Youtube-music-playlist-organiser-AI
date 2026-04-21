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
    BatchMoodClassificationItem,
    BatchMoodClassificationResponse,
    MoodClassification,
    SetupSettings,
    SongCandidate,
    utc_now,
)


SYSTEM_PROMPT = """You classify YouTube videos into mood playlists for songs in bulk.
Use only the metadata provided.
Return JSON that matches the schema exactly.
Keep the response compact and valid JSON only.
Rules:
- If the item does not look like music/song content, set is_music to false and mood to null.
- If the metadata is too weak or ambiguous, set is_music to false and explain why.
- When it is music, choose exactly one mood:
  1. Happy / Feel-good
  2. Sad / Emotional
  3. Romantic / Love
  4. Chill / Relaxing
  5. Energetic / Hype
  6. Dark / Intense
- Confidence must be an integer from 0 to 100.
- reason must be concise, grounded in the metadata, and preferably under 12 words.
- Return exactly one result item for every input song.
- Do not omit, duplicate, or invent video_id values.
- Do not include markdown, prose, or code fences.
"""


class AzureClassificationError(RuntimeError):
    """Raised when Azure OpenAI validation or connectivity fails."""


def build_cache_key(candidate: SongCandidate) -> str:
    raw = f"{candidate.video_id}:{candidate.metadata_hash}:{PROMPT_VERSION}"
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
                )
            )
            if response.output_parsed is None:
                raise AzureClassificationError("Azure OpenAI returned an empty probe response.")
        except Exception as exc:  # pragma: no cover - network error path
            raise AzureClassificationError(
                "Unable to use the configured Azure OpenAI deployment with the current Responses API settings."
            ) from exc

    async def classify_candidates(
        self,
        candidates: list[SongCandidate],
    ) -> dict[str, MoodClassification]:
        cached_results: dict[str, MoodClassification] = {}
        uncached_candidates: list[SongCandidate] = []

        for candidate in candidates:
            cache_key = build_cache_key(candidate)
            cached = self.db.load_cached_classification(cache_key)
            if cached:
                cached_results[candidate.video_id] = MoodClassification.model_validate(cached)
            else:
                uncached_candidates.append(candidate)

        if not uncached_candidates:
            return cached_results

        batches = self.pack_candidate_batches(uncached_candidates)
        fresh_results: dict[str, MoodClassification] = {}
        processed_uncached = 0

        for batch_index, batch_candidates in enumerate(batches, start=1):
            try:
                batch_results = await self._classify_batch_with_recovery(batch_candidates)
            except AzureClassificationError as exc:
                raise AzureClassificationError(
                    f"Classification failed after {processed_uncached} of {len(uncached_candidates)} uncached songs. "
                    f"Batch {batch_index}/{len(batches)} with {len(batch_candidates)} songs failed: {exc}"
                ) from exc
            fresh_results.update(batch_results)
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
    ) -> dict[str, MoodClassification]:
        last_error: Exception | None = None
        for attempt, profile in enumerate(self._build_attempt_profiles(len(candidates)), start=1):
            try:
                response = await self._request_batch_response(candidates, profile)
                items_by_id = self._validate_batch_response(candidates, response)
                return self._persist_batch_results(candidates, items_by_id)
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
        left_results = await self._classify_batch_with_recovery(candidates[:midpoint])
        right_results = await self._classify_batch_with_recovery(candidates[midpoint:])
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
    ) -> BatchMoodClassificationResponse:
        response = await self.async_client.responses.parse(
            **self._build_batch_request_kwargs(candidates, profile)
        )
        parsed = response.output_parsed
        if parsed is None:
            raise AzureClassificationError("Azure OpenAI returned an empty batch classification.")
        return parsed

    def _build_batch_request_kwargs(
        self,
        candidates: list[SongCandidate],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        song_payload = [serialize_candidate_for_batch(candidate) for candidate in candidates]
        return {
            "model": self.settings.azure_openai_deployment,
            "instructions": SYSTEM_PROMPT,
            "input": json.dumps({"songs": song_payload}, ensure_ascii=True, separators=(",", ":")),
            "reasoning": profile["reasoning"],
            "text_format": BatchMoodClassificationResponse,
            "max_output_tokens": profile["max_output_tokens"],
            "text": {"verbosity": profile["verbosity"]},
            "truncation": "disabled",
            "store": False,
        }

    def _validate_batch_response(
        self,
        candidates: list[SongCandidate],
        response: BatchMoodClassificationResponse,
    ) -> dict[str, BatchMoodClassificationItem]:
        requested_ids = [candidate.video_id for candidate in candidates]
        requested_id_set = set(requested_ids)
        response_ids = [item.video_id for item in response.items]
        response_id_set = set(response_ids)

        duplicate_ids = sorted({video_id for video_id in response_ids if response_ids.count(video_id) > 1})
        missing_ids = sorted(requested_id_set - response_id_set)
        extra_ids = sorted(response_id_set - requested_id_set)

        if duplicate_ids or missing_ids or extra_ids or len(response.items) != len(candidates):
            raise AzureClassificationError(
                "Batch response validation failed. "
                f"missing_ids={missing_ids[:5]} extra_ids={extra_ids[:5]} duplicate_ids={duplicate_ids[:5]}"
            )

        return {item.video_id: item for item in response.items}

    def _persist_batch_results(
        self,
        candidates: list[SongCandidate],
        items_by_id: dict[str, BatchMoodClassificationItem],
    ) -> dict[str, MoodClassification]:
        classifications: dict[str, MoodClassification] = {}
        for candidate in candidates:
            response_item = items_by_id[candidate.video_id]
            classification = MoodClassification(
                is_music=response_item.is_music,
                mood=response_item.mood,
                confidence=response_item.confidence,
                reason=response_item.reason,
                model_name=self.settings.azure_openai_deployment,
                prompt_version=PROMPT_VERSION,
            )
            cache_key = build_cache_key(candidate)
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
