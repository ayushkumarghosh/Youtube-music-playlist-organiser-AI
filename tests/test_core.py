from __future__ import annotations

import asyncio
import json
from pathlib import Path

from googleapiclient.errors import HttpError
import pytest

from app.db import Database
from app.models import (
    BatchCategoryClassificationItem,
    BatchCategoryClassificationResponse,
    BatchMoodClassificationItem,
    BatchMoodClassificationResponse,
    CategoryAssignment,
    CategoryLabelDefinition,
    CategorySetDefinition,
    MoodClassification,
    MoodLabel,
    PlaylistItemRecord,
    RunScope,
    RunStatus,
    SetupSettings,
    SongCandidate,
    SongCategoryClassification,
    built_in_category_sets,
)
from app.security import EncryptedStateError, decrypt_json, encrypt_json
from app.services.settings import SettingsService
from app.services.azure_openai import AzureClassificationError, AzureOpenAIClassifier, build_cache_key
from app.services.organizer import OrganizerService, dedupe_candidates
from app.services.youtube import (
    GOOGLE_SCOPES,
    YouTubeService,
    YouTubeSyncError,
    build_managed_playlist_title,
    extract_managed_playlist_mood,
    is_managed_playlist,
)


GOOGLE_CLIENT_SECRETS_JSON = '{"web":{"client_id":"client-id","client_secret":"client-secret","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token"}}'


def build_temp_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    db.initialize()
    return db


def test_settings_are_loaded_from_env_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = build_temp_db(tmp_path)
    db.save_settings(
        SetupSettings(
            azure_openai_endpoint="https://stored.openai.azure.com",
            azure_openai_api_key="stored",
            azure_openai_deployment="stored",
            google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
            session_secret="stored",
        )
    )
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRETS_JSON", GOOGLE_CLIENT_SECRETS_JSON)
    monkeypatch.setenv("SESSION_SECRET", "env-secret")

    settings = SettingsService(db).get_settings()

    assert settings.azure_openai_endpoint == "https://env.openai.azure.com"
    assert settings.azure_openai_api_key == "env-key"
    assert settings.azure_openai_deployment == "env-deployment"
    assert settings.session_secret == "env-secret"


def test_settings_validation_reports_missing_and_invalid_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = build_temp_db(tmp_path)
    for key in [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "GOOGLE_CLIENT_SECRETS_JSON",
        "SESSION_SECRET",
    ]:
        monkeypatch.delenv(key, raising=False)

    errors = SettingsService(db).validate()
    assert "AZURE_OPENAI_ENDPOINT is required." in errors
    assert "GOOGLE_CLIENT_SECRETS_JSON is required." in errors

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deployment")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRETS_JSON", "{")
    monkeypatch.setenv("SESSION_SECRET", "env-secret")

    assert SettingsService(db).validate() == ["GOOGLE_CLIENT_SECRETS_JSON must be valid JSON."]


def test_youtube_oauth_uses_narrow_playlist_write_scope() -> None:
    assert GOOGLE_SCOPES == ["https://www.googleapis.com/auth/youtube.force-ssl"]


def test_encrypted_browser_state_round_trip_and_rejects_invalid_token() -> None:
    token = encrypt_json({"token": "abc", "nested": {"ok": True}}, "session-secret")

    assert decrypt_json(token, "session-secret") == {"token": "abc", "nested": {"ok": True}}

    with pytest.raises(EncryptedStateError):
        decrypt_json(token, "wrong-secret")

    with pytest.raises(EncryptedStateError):
        decrypt_json("not-a-fernet-token", "session-secret")


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
    assert title == "Happy / Feel-good"
    assert is_managed_playlist(title, "[vibeshelf-managed] managed")
    assert is_managed_playlist(title, "[yt-mood-organizer-managed] managed")
    assert is_managed_playlist("Custom playlist", "[yt-mood-organizer-managed] managed")
    assert not is_managed_playlist("Road trip", "normal playlist")


def test_extract_managed_playlist_mood_supports_new_and_legacy_formats() -> None:
    assert (
        extract_managed_playlist_mood(
            "Happy / Feel-good",
            "[vibeshelf-managed] Managed by VibeShelf. Scope: All playlists. Mood: Happy / Feel-good.",
        )
        == "Happy / Feel-good"
    )
    assert (
        extract_managed_playlist_mood(
            "Happy / Feel-good",
            "[yt-mood-organizer-managed] Managed by VibeShelf. Scope: All playlists. Mood: Happy / Feel-good.",
        )
        == "Happy / Feel-good"
    )
    assert (
        extract_managed_playlist_mood(
            "Mood [All] - Happy / Feel-good",
            "legacy description",
        )
        == "Happy / Feel-good"
    )


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
                "moods": ["Unclear"],
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
                        "moods": [MoodLabel.HAPPY],
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
        google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
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
                    moods=[MoodLabel.HAPPY],
                    confidence=88,
                    reason="Upbeat title and artist context.",
                )
            ]
        )

    classifier._request_batch_response = fake_request  # type: ignore[method-assign]
    first = asyncio.run(classifier.classify_candidates([candidate]))
    second = asyncio.run(classifier.classify_candidates([candidate]))
    assert first[candidate.video_id].moods == [MoodLabel.HAPPY]
    assert second[candidate.video_id].moods == [MoodLabel.HAPPY]
    assert calls["count"] == 1


def test_batch_packer_keeps_1200_songs_together(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    classifier = AzureOpenAIClassifier(
        SetupSettings(
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret",
            azure_openai_deployment="gpt-5.4",
            google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
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
            google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
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
            google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
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
    assert kwargs["text_format"] is BatchCategoryClassificationResponse
    assert "\"category_sets\"" in kwargs["input"]


def test_cached_songs_are_excluded_from_batch_request_but_returned(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    settings = SetupSettings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="secret",
        azure_openai_deployment="gpt-5.4",
        google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
        session_secret="secret",
    )
    classifier = AzureOpenAIClassifier(settings, db)
    candidates = [make_candidate(index) for index in range(3)]

    cached_classification = MoodClassification(
        is_music=True,
        moods=[MoodLabel.CHILL],
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
                moods=[MoodLabel.HAPPY],
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
            google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
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
                moods=[MoodLabel.HAPPY],
                confidence=80,
                reason="ok",
            ),
            BatchMoodClassificationItem(
                video_id="extra-id",
                is_music=False,
                moods=[],
                confidence=15,
                reason="extra",
            ),
            BatchMoodClassificationItem(
                video_id=candidates[0].video_id,
                is_music=True,
                moods=[MoodLabel.CHILL],
                confidence=60,
                reason="dup",
            ),
        ]
    )
    with pytest.raises(AzureClassificationError):
        classifier._validate_batch_response(candidates, bad_response)


def test_category_batch_response_rejects_unknown_label(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    classifier = AzureOpenAIClassifier(
        SetupSettings(
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret",
            azure_openai_deployment="gpt-5.4",
            google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
            session_secret="secret",
        ),
        db,
    )
    candidate = make_candidate(1)
    activity = next(category for category in built_in_category_sets() if category.id == "activity")
    response = BatchCategoryClassificationResponse(
        items=[
            BatchCategoryClassificationItem(
                video_id=candidate.video_id,
                is_music=True,
                assignments=[
                    CategoryAssignment(
                        category_id="activity",
                        label_slugs=["not-a-real-label"],
                        confidence=80,
                        reason="bad label",
                    )
                ],
            )
        ]
    )

    with pytest.raises(AzureClassificationError):
        classifier._validate_batch_response([candidate], response, [activity])


def test_split_on_failure_recovers_by_recursing_into_smaller_batches(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    settings = SetupSettings(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="secret",
        azure_openai_deployment="gpt-5.4",
        google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
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
                    moods=[MoodLabel.HAPPY],
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
            google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
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
            google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
            session_secret="secret",
        ),
        db,
    )

    class FakeRequest:
        def execute(self):
            raise make_http_error(403, "insufficientPermissions", "Forbidden")

    with pytest.raises(YouTubeSyncError):
        service._execute_request(FakeRequest, "adding a test video")


def test_youtube_revoke_token_posts_refresh_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = build_temp_db(tmp_path)
    service = YouTubeService(
        SetupSettings(
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret",
            azure_openai_deployment="gpt-5.4",
            google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
            session_secret="secret",
        ),
        db,
        {"token": "access-token", "refresh_token": "refresh-token"},
    )
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.data, timeout))
        return FakeResponse()

    monkeypatch.setattr("app.services.youtube.urlopen", fake_urlopen)

    service.revoke_token()

    assert calls == [
        (
            "https://oauth2.googleapis.com/revoke",
            b"token=refresh-token",
            10,
        )
    ]


def test_delete_authorized_youtube_data_clears_runs_cache_and_tokens(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    db.save_token_payload("google", {"token": "abc"})
    db.save_cached_classification(
        cache_key="cache-key",
        video_id="video-1",
        metadata_hash="hash",
        prompt_version="prompt",
        payload={"ok": True},
        updated_at="now",
    )
    db.save_run(
        run_id="run-1",
        status=RunStatus.PREVIEWED,
        scope=RunScope.SELECTED_PLAYLISTS,
        source_playlist_id="playlist-1",
        source_playlist_title="Road Trip",
        created_at="now",
        summary_json={
            "total_candidates": 1,
            "classified_count": 1,
            "default_included_count": 1,
            "excluded_count": 0,
        },
        items=[
            {
                "video_id": "video-1",
                "title": "Song",
                "channel_title": "Artist",
                "description": "desc",
                "source_playlists": ["Road Trip"],
                "source_positions": [0],
                "suggested_moods": [MoodLabel.HAPPY],
                "final_moods": [MoodLabel.HAPPY],
                "confidence": 90,
                "reason": "reason",
                "is_music": True,
                "default_included": True,
            }
        ],
    )

    db.delete_authorized_youtube_data()

    assert db.load_token_payload("google") is None
    assert db.load_cached_classification("cache-key") is None
    assert db.get_run("run-1") is None


def test_custom_category_crud_and_archive(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    category = CategorySetDefinition(
        id="custom-intent",
        name="Listening Intent",
        description="Custom playlist category.",
        source="custom",
        prompt="Organize songs by why I would listen to them.",
        labels=[
            CategoryLabelDefinition(slug="sing-along", name="Sing Along", description="Hooky vocal songs."),
            CategoryLabelDefinition(slug="background", name="Background", description="Low-friction listening."),
        ],
    )

    saved = db.save_custom_category_set(category)

    assert saved.id == "custom-intent"
    assert [item.id for item in db.list_custom_category_sets()] == ["custom-intent"]
    assert db.get_custom_category_set("custom-intent").labels[0].name == "Sing Along"  # type: ignore[union-attr]

    db.archive_custom_category_set("custom-intent")

    assert db.list_custom_category_sets() == []
    assert db.list_custom_category_sets(include_archived=True)[0].archived is True


def test_reconcile_playlist_only_appends_missing_videos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = build_temp_db(tmp_path)
    service = YouTubeService(
        SetupSettings(
            azure_openai_endpoint="https://example.openai.azure.com",
            azure_openai_api_key="secret",
            azure_openai_deployment="gpt-5.4",
            google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
            session_secret="secret",
        ),
        db,
    )
    inserted_video_ids: list[str] = []

    class FakeRequest:
        def __init__(self, video_id: str) -> None:
            self.video_id = video_id

        def execute(self):
            inserted_video_ids.append(self.video_id)
            return {"id": f"playlist-item-{self.video_id}"}

    class FakePlaylistItems:
        def insert(self, part: str, body: dict[str, object]):
            video_id = str(body["snippet"]["resourceId"]["videoId"])  # type: ignore[index]
            return FakeRequest(video_id)

    class FakeYouTubeClient:
        def playlistItems(self):
            return FakePlaylistItems()

    monkeypatch.setattr(service, "_client", lambda: FakeYouTubeClient())
    monkeypatch.setattr(
        service,
        "_fetch_playlist_records",
        lambda youtube, playlist_id: [
            {"playlist_item_id": "existing-v1", "video_id": "v1", "position": 0},
            {"playlist_item_id": "legacy-track", "video_id": "legacy", "position": 1},
        ],
    )

    counts = service.reconcile_playlist("managed-happy", ["v1", "v2", "v2", "v3"])

    assert counts == {"deletes": 0, "inserts": 2, "updates": 0}
    assert inserted_video_ids == ["v2", "v3"]


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

    def get_source_playlists(
        self,
        scope: RunScope,
        selected_playlist_id: str | None = None,
        selected_playlist_ids: list[str] | None = None,
    ):
        if scope == RunScope.ALL_PLAYLISTS:
            return self.playlists
        if scope == RunScope.SELECTED_PLAYLISTS:
            selected_ids = set(selected_playlist_ids or [])
            return [playlist for playlist in self.playlists if playlist.playlist_id in selected_ids]
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
                {
                    "is_music": True,
                    "moods": [MoodLabel.HAPPY, MoodLabel.CHILL],
                    "confidence": 82,
                    "reason": "Soft vibe",
                },
            )(),
            "v2": type(
                "Classification",
                (),
                {"is_music": False, "moods": [], "confidence": 25, "reason": "Looks like a podcast clip"},
            )(),
            "v3": type(
                "Classification",
                (),
                {"is_music": True, "moods": [MoodLabel.ENERGETIC], "confidence": 91, "reason": "Gym energy"},
            )(),
        }


class FakeCategoryYouTubeService(FakeYouTubeService):
    def ensure_managed_playlists(
        self,
        scope: RunScope,
        source_playlist_title: str | None,
        category_sets=None,
    ):
        result = {}
        for category in category_sets:
            for label in category.labels:
                key = f"{category.id}:{label.slug}"
                result[key] = type(
                    "Playlist",
                    (),
                    {
                        "playlist_id": f"managed-{key}",
                        "title": build_managed_playlist_title(scope, label.name, source_playlist_title),
                    },
                )()
        return result


class FakeCategoryClassifier:
    async def classify_candidates(self, candidates: list[SongCandidate], category_sets):
        assert [category.id for category in category_sets] == ["mood", "activity"]
        return {
            "v1": SongCategoryClassification(
                is_music=True,
                assignments=[
                    CategoryAssignment(
                        category_id="mood",
                        label_slugs=["happy-feel-good", "chill-relaxing"],
                        confidence=82,
                        reason="Soft vibe",
                    ),
                    CategoryAssignment(
                        category_id="activity",
                        label_slugs=["driving-road-trip"],
                        confidence=76,
                        reason="Road-trip fit",
                    ),
                ],
            ),
            "v2": SongCategoryClassification(is_music=False, assignments=[]),
            "v3": SongCategoryClassification(
                is_music=True,
                assignments=[
                    CategoryAssignment(
                        category_id="mood",
                        label_slugs=["energetic-hype"],
                        confidence=91,
                        reason="Gym energy",
                    ),
                    CategoryAssignment(
                        category_id="activity",
                        label_slugs=["workout"],
                        confidence=92,
                        reason="Workout fit",
                    ),
                ],
            ),
        }


def test_preview_all_playlists_and_non_music_exclusion(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    organizer = OrganizerService(db, FakeYouTubeService(), FakeClassifier())
    run = organizer.create_preview(RunScope.ALL_PLAYLISTS)
    assert run.summary.total_candidates == 3
    assert run.summary.default_included_count == 2
    assert [item.video_id for item in run.items] == ["v2", "v1", "v3"]
    included_item = next(item for item in run.items if item.video_id == "v1")
    assert included_item.suggested_moods == [MoodLabel.HAPPY, MoodLabel.CHILL]
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
            "v2": ["Happy / Feel-good"],
        },
    )
    assert summary["total_assignments"] == 3
    happy_playlist = "managed-Happy / Feel-good"
    chill_playlist = "managed-Chill / Relaxing"
    assert youtube.reconciled[happy_playlist] == ["v1", "v2"]
    assert youtube.reconciled[chill_playlist] == ["v1"]


def test_preview_selected_playlists_supports_multiple_sources(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    organizer = OrganizerService(db, FakeYouTubeService(), FakeClassifier())

    run = organizer.create_preview(
        RunScope.SELECTED_PLAYLISTS,
        source_playlist_ids=["source-1", "source-2"],
    )

    assert run.scope == RunScope.SELECTED_PLAYLISTS
    assert run.source_playlist_id == "source-1,source-2"
    assert run.source_playlist_title == "2 selected playlists"
    assert run.summary.total_candidates == 3


def test_preview_with_activity_category_and_apply(tmp_path: Path) -> None:
    db = build_temp_db(tmp_path)
    youtube = FakeCategoryYouTubeService()
    mood = next(category for category in built_in_category_sets() if category.id == "mood")
    activity = next(category for category in built_in_category_sets() if category.id == "activity")
    organizer = OrganizerService(db, youtube, FakeCategoryClassifier())

    run = organizer.create_preview(
        RunScope.SELECTED_PLAYLISTS,
        source_playlist_ids=["source-1", "source-2"],
        category_sets=[mood, activity],
    )
    v1 = next(item for item in run.items if item.video_id == "v1")

    assert run.category_sets == [mood, activity]
    assert v1.suggested_label_names(activity) == ["Driving/Road Trip"]

    summary = organizer.apply_run(
        run.run_id,
        {
            "activity": {
                "v1": ["driving-road-trip"],
                "v3": ["workout"],
            }
        },
    )

    assert summary["total_assignments"] == 5
    assert youtube.reconciled["managed-activity:driving-road-trip"] == ["v1"]
    assert youtube.reconciled["managed-activity:workout"] == ["v3"]
