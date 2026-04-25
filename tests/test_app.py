from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.models import SetupSettings
from app.services.azure_openai import AzureClassificationError
from app.services.youtube import YouTubeSyncError


GOOGLE_CLIENT_SECRETS_JSON = '{"web":{"client_id":"client-id","client_secret":"client-secret","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token"}}'


def test_home_renders() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "YouTube Mood Playlist Organizer" in response.text


def test_home_does_not_render_credential_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRETS_JSON", GOOGLE_CLIENT_SECRETS_JSON)
    monkeypatch.setenv("SESSION_SECRET", "session-secret")

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert 'name="azure_openai_api_key"' not in response.text
    assert 'name="google_client_secrets_json"' not in response.text
    assert 'action="/settings/save"' not in response.text


def test_preview_classification_error_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSettingsService:
        def get_settings(self):
            return SetupSettings(
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_openai_api_key="secret",
                azure_openai_deployment="gpt-5.4",
                google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
                session_secret="secret",
            )

    class FakeYouTubeService:
        def __init__(self, settings, db, token_payload=None):
            pass

        def has_token(self):
            return True

    class FakeOrganizerService:
        def __init__(self, db, youtube_service, classifier):
            pass

        def create_preview(self, scope, source_playlist_id=None, persist=True):
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
                google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
                session_secret="secret",
            )

    class FakeYouTubeService:
        def __init__(self, settings, db, token_payload=None):
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
    assert response.headers["location"] == "/"
