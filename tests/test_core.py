from __future__ import annotations

import asyncio
import json
from pathlib import Path

from googleapiclient.errors import HttpError
import pytest

from app.db import Database
from app.models import (
    BatchMoodClassificationItem,
    BatchMoodClassificationResponse,
    MoodClassification,
    MoodLabel,
    PlaylistItemRecord,
    RunScope,
    SetupSettings,
    SongCandidate,
)
from app.services.azure_openai import AzureClassificationError, AzureOpenAIClassifier, build_cache_key
from app.services.organizer import OrganizerService, dedupe_candidates
from app.services.youtube import (
    YouTubeService,
    YouTubeSyncError,
    build_managed_playlist_title,
    is_managed_playlist,
)


def build_temp_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.initialize()
    return db


def make_candidate(index: int, *, description: str = "desc") -> SongCandidate:
    return SongCandidate(
        video_id=f"video-{index}",
        title=f"Song {index}",
        channel_title=f"Artist {index}",
        description=description,
        source_playlists=["A"],
        source_playlist_ids=["p1"],
        source_positions=[index],
    )


def make_http_error(status: int, reason: str, message: str) -> HttpError:
    response = type("Resp", (), {"status": status, "reason": message})()
    content = json.dumps(
        {
            "error": {
                "message": message,
                "errors": [{"reason": reason}],
            }
        }
    ).encode("utf-8")
    return HttpError(response, content, uri="https://youtube.googleapis.com/test")


def test_managed_playlist_detection_and_naming() -> None:
    title = build_managed_playlist_title(RunScope.ALL_PLAYLISTS, "Happy / Feel-good")
    assert title == "Mood [All] - Happy / Feel-good"
    assert is_managed_playlist(title, "plain description")
    assert is_managed_playlist("Custom playlist", "[yt-mood-organizer-managed] managed")
    assert not is_managed_playlist("Road trip", "normal playlist")


def test_dedupe_candidates_keeps_source_context() -> None:
    items = [
        PlaylistItemRecord(
            playlist_item_id="pi1",
            playlist_id="p1",
            playlist_title="A",
            video_id="v1",
            title="Song One",
            description="desc",
            channel_title="Artist",
            position=0,
        ),
        PlaylistItemRecord(
            playlist_item_id="pi2",
            playlist_id="p2",
            playlist_title="B",
            video_id="v1",
            title="Song One",
            description="desc",
            channel_title="Artist",
            position=4,
        ),
    ]
    candidates = dedupe_candidates(items)
    assert len(candidates) == 1
    assert candidates[0].source_playlists == ["A", "B"]
    assert candidates[0].source_positions == [0, 4]


def test_schema_contract_rejects_invalid_mood() -> None:
    with pytest.raises(Exception):
        BatchMoodClassificationItem.model_validate(
            {
                "video_id": "v1",
                "is_music": True,
                "mood": "Unclear",
                "confidence": 50,
                "reason": "bad mood",
            }
        )


def test_schema_contract_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        BatchMoodClassificationResponse.model_validate(
            {
                "items": [
                    {
                        "video_id": "v1",
                        "is_music": True,
                        "mood": MoodLabel.HAPPY,
                        "confidence": 50,
                        "reason": "valid",
                        "extra_field": "nope",
                    }
                ]
            }
        )


def test_cache_key_changes_with_metadata() -> None:
    one = SongCandidate(
        video_id="v1",
        title="Song",
        channel_title="Artist",
        description="One",
        source_playlists=["A"],
        source_playlist_ids=["p1"],
        source_positions=[0],
    )
    two = SongCandidate(
        video_id="v1",
        title="Song",
        channel_title="Artist",
        description="Two",
        source_playlists=["A"],
        source_playlist_ids=["p1"],
        source_positions=[0],
    )
    assert build_cache_key(one) != build_cache_key(two)


def test_classifier_uses_cache(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    settings = SetupSettings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="secret",
        azure_openai_deployment="gpt-5.4",
        google_client_secrets_file="client.json",
        session_secret="secret",
    )
    classifier = AzureOpenAIClassifier(settings, db)
    candidate = make_candidate(1, description="A bright summer track")
    calls = {"count": 0}

    async def fake_request(batch_candidates: list[SongCandidate], profile):
        calls["count"] += 1
        return BatchMoodClassificationResponse(
            items=[
                BatchMoodClassificationItem(
                    video_id=batch_candidates[0].video_id,
                    is_music=True,
                    mood=MoodLabel.HAPPY,
                    confidence=88,
                    reason="Upbeat title and artist context.",
                )
            ]
        )

    classifier._request_batch_response = fake_request  # type: ignore[method-assign]
    first = asyncio.run(classifier.classify_candidates([candidate]))
    second = asyncio.run(classifier.classify_candidates([candidate]))
    assert first[candidate.video_id].mood == MoodLabel.HAPPY
    assert second[candidate.video_id].mood == MoodLabel.HAPPY
    assert calls["count"] == 1


def test_batch_packer_keeps_1200_songs_together(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    classifier = AzureOpenAIClassifier(
        SetupSettings(
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret",
            azure_openai_deployment="gpt-5.4",
            google_client_secrets_file="client.json",
            session_secret="secret",
        ),
        db,
    )
    batches = classifier.pack_candidate_batches([make_candidate(index) for index in range(1200)])
    assert len(batches) == 1
    assert len(batches[0]) == 1200


def test_batch_packer_splits_large_inputs_and_keeps_remainder(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    classifier = AzureOpenAIClassifier(
        SetupSettings(
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret",
            azure_openai_deployment="gpt-5.4",
            google_client_secrets_file="client.json",
            session_secret="secret",
        ),
        db,
    )
    batches = classifier.pack_candidate_batches([make_candidate(index) for index in range(2501)])
    assert len(batches) == 2
    assert len(batches[0]) >= 1000
    assert len(batches[1]) == 501
    assert sum(len(batch) for batch in batches) == 2501


def test_batch_request_kwargs_use_canonical_responses_shape(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    classifier = AzureOpenAIClassifier(
        SetupSettings(
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret",
            azure_openai_deployment="gpt-5.4",
            google_client_secrets_file="client.json",
            session_secret="secret",
        ),
        db,
    )
    candidate = make_candidate(7, description="A bright and punchy anthem")
    profile = classifier._build_attempt_profiles(batch_size=1)[0]

    kwargs = classifier._build_batch_request_kwargs([candidate], profile)

    assert kwargs["instructions"]
    assert kwargs["input"].startswith("{\"songs\":[")
    assert kwargs["text"] == {"verbosity": "low"}
    assert kwargs["truncation"] == "disabled"
    assert kwargs["store"] is False
    assert kwargs["text_format"] is BatchMoodClassificationResponse


def test_cached_songs_are_excluded_from_batch_request_but_returned(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    settings = SetupSettings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="secret",
        azure_openai_deployment="gpt-5.4",
        google_client_secrets_file="client.json",
        session_secret="secret",
    )
    classifier = AzureOpenAIClassifier(settings, db)
    candidates = [make_candidate(index) for index in range(3)]

    cached_classification = MoodClassification(
        is_music=True,
        mood=MoodLabel.CHILL,
        confidence=77,
        reason="Cached result",
        model_name="gpt-5.4",
        prompt_version="test",
    )
    cached_candidate = candidates[0]
    db.save_cached_classification(
        cache_key=build_cache_key(cached_candidate),
        video_id=cached_candidate.video_id,
        metadata_hash=cached_candidate.metadata_hash,
        prompt_version=cached_classification.prompt_version,
        payload=cached_classification.model_dump(mode="json"),
        updated_at="now",
    )

    seen_video_ids: list[str] = []

    async def fake_batch(batch_candidates: list[SongCandidate]):
        seen_video_ids.extend(candidate.video_id for candidate in batch_candidates)
        return {
            candidate.video_id: MoodClassification(
                is_music=True,
                mood=MoodLabel.HAPPY,
                confidence=90,
                reason="Fresh result",
                model_name="gpt-5.4",
                prompt_version="test",
            )
            for candidate in batch_candidates
        }

    classifier._classify_batch_with_recovery = fake_batch  # type: ignore[method-assign]
    results = asyncio.run(classifier.classify_candidates(candidates))
    assert set(results.keys()) == {candidate.video_id for candidate in candidates}
    assert seen_video_ids == [candidates[1].video_id, candidates[2].video_id]
    assert results[cached_candidate.video_id].reason == "Cached result"


def test_validate_batch_response_rejects_missing_duplicate_and_extra_ids(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    classifier = AzureOpenAIClassifier(
        SetupSettings(
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret",
            azure_openai_deployment="gpt-5.4",
            google_client_secrets_file="client.json",
            session_secret="secret",
        ),
        db,
    )
    candidates = [make_candidate(index) for index in range(2)]
    bad_response = BatchMoodClassificationResponse(
        items=[
            BatchMoodClassificationItem(
                video_id=candidates[0].video_id,
                is_music=True,
                mood=MoodLabel.HAPPY,
                confidence=80,
                reason="ok",
            ),
            BatchMoodClassificationItem(
                video_id="extra-id",
                is_music=False,
                mood=None,
                confidence=15,
                reason="extra",
            ),
            BatchMoodClassificationItem(
                video_id=candidates[0].video_id,
                is_music=True,
                mood=MoodLabel.CHILL,
                confidence=60,
                reason="dup",
            ),
        ]
    )
    with pytest.raises(AzureClassificationError):
        classifier._validate_batch_response(candidates, bad_response)


def test_split_on_failure_recovers_by_recursing_into_smaller_batches(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    settings = SetupSettings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="secret",
        azure_openai_deployment="gpt-5.4",
        google_client_secrets_file="client.json",
        session_secret="secret",
    )
    classifier = AzureOpenAIClassifier(settings, db)
    candidates = [make_candidate(index) for index in range(4)]
    request_sizes: list[int] = []

    async def fake_request(batch_candidates: list[SongCandidate], profile):
        request_sizes.append(len(batch_candidates))
        if len(batch_candidates) > 1:
            raise AzureClassificationError("force split")
        candidate = batch_candidates[0]
        return BatchMoodClassificationResponse(
            items=[
                BatchMoodClassificationItem(
                    video_id=candidate.video_id,
                    is_music=True,
                    mood=MoodLabel.HAPPY,
                    confidence=90,
                    reason="single fallback",
                )
            ]
        )

    classifier._request_batch_response = fake_request  # type: ignore[method-assign]
    results = asyncio.run(classifier._classify_batch_with_recovery(candidates))
    assert set(results.keys()) == {candidate.video_id for candidate in candidates}
    assert any(size == 1 for size in request_sizes)


def test_youtube_request_retries_transient_http_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = build_temp_db(tmp_path)
    service = YouTubeService(
        SetupSettings(
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret",
            azure_openai_deployment="gpt-5.4",
            google_client_secrets_file="client.json",
            session_secret="secret",
        ),
        db,
    )
    monkeypatch.setattr("app.services.youtube.time.sleep", lambda _: None)
    attempts = {"count": 0}

    class FakeRequest:
        def execute(self):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise make_http_error(409, "SERVICE_UNAVAILABLE", "The operation was aborted.")
            return {"ok": True}

    result = service._execute_request(FakeRequest, "adding a test video")

    assert result == {"ok": True}
    assert attempts["count"] == 3


def test_youtube_request_raises_sync_error_for_non_retryable_http_error(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    service = YouTubeService(
        SetupSettings(
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret",
            azure_openai_deployment="gpt-5.4",
            google_client_secrets_file="client.json",
            session_secret="secret",
        ),
        db,
    )

    class FakeRequest:
        def execute(self):
            raise make_http_error(403, "insufficientPermissions", "Forbidden")

    with pytest.raises(YouTubeSyncError):
        service._execute_request(FakeRequest, "adding a test video")


class FakeYouTubeService:
    def __init__(self) -> None:
        self.playlists = [
            type("Playlist", (), {"playlist_id": "source-1", "title": "Chill Mix", "item_count": 2})(),
            type("Playlist", (), {"playlist_id": "source-2", "title": "Gym", "item_count": 1})(),
        ]
        self.items = {
            "source-1": [
                PlaylistItemRecord("i1", "source-1", "Chill Mix", "v1", "Song A", "desc", "Artist A", 0),
                PlaylistItemRecord("i2", "source-1", "Chill Mix", "v2", "Song B", "desc", "Artist B", 1),
            ],
            "source-2": [
                PlaylistItemRecord("i3", "source-2", "Gym", "v3", "Song C", "desc", "Artist C", 0),
                PlaylistItemRecord("i4", "source-2", "Gym", "v1", "Song A", "desc", "Artist A", 1),
            ],
        }
        self.reconciled: dict[str, list[str]] = {}

    def get_source_playlists(self, scope: RunScope, selected_playlist_id: str | None = None):
        if scope == RunScope.ALL_PLAYLISTS:
            return self.playlists
        return [playlist for playlist in self.playlists if playlist.playlist_id == selected_playlist_id]

    def list_playlist_items(self, playlist_id: str, playlist_title: str):
        return self.items[playlist_id]

    def ensure_managed_playlists(self, scope: RunScope, source_playlist_title: str | None):
        titles = {}
        for mood in [
            "Happy / Feel-good",
            "Sad / Emotional",
            "Romantic / Love",
            "Chill / Relaxing",
            "Energetic / Hype",
            "Dark / Intense",
        ]:
            titles[mood] = type(
                "Playlist",
                (),
                {
                    "playlist_id": f"managed-{mood}",
                    "title": build_managed_playlist_title(scope, mood, source_playlist_title),
                },
            )()
        return titles

    def reconcile_playlist(self, playlist_id: str, desired_video_ids: list[str]):
        self.reconciled[playlist_id] = desired_video_ids
        return {"deletes": 0, "inserts": len(desired_video_ids), "updates": 0}


class FakeClassifier:
    async def classify_candidates(self, candidates: list[SongCandidate]):
        return {
            "v1": type(
                "Classification",
                (),
                {"is_music": True, "mood": MoodLabel.CHILL, "confidence": 82, "reason": "Soft vibe"},
            )(),
            "v2": type(
                "Classification",
                (),
                {"is_music": False, "mood": None, "confidence": 25, "reason": "Looks like a podcast clip"},
            )(),
            "v3": type(
                "Classification",
                (),
                {"is_music": True, "mood": MoodLabel.ENERGETIC, "confidence": 91, "reason": "Gym energy"},
            )(),
        }


def test_preview_all_playlists_and_non_music_exclusion(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    organizer = OrganizerService(db, FakeYouTubeService(), FakeClassifier())
    run = organizer.create_preview(RunScope.ALL_PLAYLISTS)
    assert run.summary.total_candidates == 3
    assert run.summary.default_included_count == 2
    excluded = {item.video_id for item in run.items if not item.default_included}
    assert excluded == {"v2"}


def test_preview_single_playlist_and_apply_override(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    youtube = FakeYouTubeService()
    organizer = OrganizerService(db, youtube, FakeClassifier())
    run = organizer.create_preview(RunScope.SINGLE_PLAYLIST, "source-1")
    assert run.source_playlist_title == "Chill Mix"
    summary = organizer.apply_run(
        run.run_id,
        {
            "v1": "",
            "v2": "Happy / Feel-good",
        },
    )
    assert summary["total_assignments"] == 1
    happy_playlist = "managed-Happy / Feel-good"
    chill_playlist = "managed-Chill / Relaxing"
    assert youtube.reconciled[happy_playlist] == ["v2"]
    assert youtube.reconciled[chill_playlist] == []
