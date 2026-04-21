from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.models import SetupSettings
from app.services.azure_openai import AzureClassificationError
from app.services.youtube import YouTubeSyncError


def test_home_renders() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "YouTube Mood Playlist Organizer" in response.text


def test_preview_classification_error_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSettingsService:
        def get_settings(self):
            return SetupSettings(
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_openai_api_key="secret",
                azure_openai_deployment="gpt-5.4",
                google_client_secrets_file="client.json",
                session_secret="secret",
            )

    class FakeYouTubeService:
        def __init__(self, settings, db):
            pass

        def has_token(self):
            return True

    class FakeOrganizerService:
        def __init__(self, db, youtube_service, classifier):
            pass

        def create_preview(self, scope, source_playlist_id=None):
            raise AzureClassificationError("mock classification failure")

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.YouTubeService", FakeYouTubeService)
    monkeypatch.setattr("app.main.OrganizerService", FakeOrganizerService)

    client = TestClient(app)
    response = client.post(
        "/runs/preview",
        data={"scope": "all_playlists", "selected_playlist_id": ""},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_apply_sync_error_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSettingsService:
        def get_settings(self):
            return SetupSettings(
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_openai_api_key="secret",
                azure_openai_deployment="gpt-5.4",
                google_client_secrets_file="client.json",
                session_secret="secret",
            )

    class FakeYouTubeService:
        def __init__(self, settings, db):
            pass

    class FakeOrganizerService:
        def __init__(self, db, youtube_service, classifier):
            pass

        def apply_run(self, run_id, overrides):
            raise YouTubeSyncError("mock sync failure")

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.YouTubeService", FakeYouTubeService)
    monkeypatch.setattr("app.main.OrganizerService", FakeOrganizerService)

    client = TestClient(app)
    response = client.post(
        "/runs/apply",
        data={"run_id": "run-123", "mood__video-1": "Happy / Feel-good"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/runs/run-123"
