from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.models import (
    MoodLabel,
    PlaylistSummary,
    RunDetail,
    RunItemView,
    RunScope,
    RunStatus,
    RunSummary,
    SetupSettings,
)
from app.services.azure_openai import AzureClassificationError
from app.services.youtube import YouTubeAuthError, YouTubeSyncError


GOOGLE_CLIENT_SECRETS_JSON = '{"web":{"client_id":"client-id","client_secret":"client-secret","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token"}}'


def test_home_renders() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "VibeShelf" in response.text


def test_home_renders_login_only_without_status_or_preview_form(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRETS_JSON", GOOGLE_CLIENT_SECRETS_JSON)
    monkeypatch.setenv("SESSION_SECRET", "session-secret")

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "Connect YouTube" in response.text
    assert 'href="/terms"' in response.text
    assert 'href="/privacy"' in response.text
    assert 'name="policy_agreement"' in response.text
    assert "Local utility" not in response.text
    assert "Readiness" not in response.text
    assert "YouTube connection" not in response.text
    assert "Generate preview" not in response.text
    assert 'action="/runs/preview"' not in response.text
    assert "1. Setup" not in response.text
    assert "2. YouTube connection" not in response.text
    assert "3. Generate preview" not in response.text


def test_home_redirects_to_preview_when_already_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSettingsService:
        def get_settings(self):
            return SetupSettings(
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_openai_api_key="secret",
                azure_openai_deployment="gpt-5.4",
                google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
                session_secret="secret",
            )

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.google_token_payload", lambda request: {"access_token": "token"})

    client = TestClient(app)
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/preview"


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


def test_terms_page_includes_youtube_terms() -> None:
    client = TestClient(app)
    response = client.get("/terms")

    assert response.status_code == 200
    assert "Terms of Use" in response.text
    assert "https://www.youtube.com/t/terms" in response.text
    assert "bound by the" in response.text
    assert "YouTube Terms of Service" in response.text


def test_privacy_page_includes_required_youtube_api_disclosures() -> None:
    client = TestClient(app)
    response = client.get("/privacy")

    assert response.status_code == 200
    assert "Privacy Policy" in response.text
    assert "YouTube API Services" in response.text
    assert "http://www.google.com/policies/privacy" in response.text
    assert "https://security.google.com/settings/security/permissions" in response.text
    assert "session cookie" in response.text
    assert "Google token cookie" in response.text
    assert "Azure OpenAI" in response.text
    assert "ayush@scorptech.co" in response.text


def test_preview_workspace_redirects_when_not_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSettingsService:
        def get_settings(self):
            return SetupSettings(
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_openai_api_key="secret",
                azure_openai_deployment="gpt-5.4",
                google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
                session_secret="secret",
            )

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.google_token_payload", lambda request: None)

    client = TestClient(app)
    response = client.get("/preview", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_preview_workspace_renders_playlists_and_preview_form(monkeypatch: pytest.MonkeyPatch) -> None:
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

        def list_playlists(self, include_managed=False):
            return [
                PlaylistSummary(
                    playlist_id="playlist-1",
                    title="Road Trip",
                    description="desc",
                    privacy_status="private",
                    item_count=12,
                )
            ]

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.google_token_payload", lambda request: {"access_token": "token"})
    monkeypatch.setattr("app.main.YouTubeService", FakeYouTubeService)

    client = TestClient(app)
    response = client.get("/preview")

    assert response.status_code == 200
    assert "Source playlists" in response.text
    assert "Disconnect YouTube" in response.text
    assert "Road Trip" in response.text
    assert 'name="selected_playlist_ids"' in response.text
    assert 'data-select-all-playlists' in response.text
    assert "<select" not in response.text
    assert 'data-loading-title="Generating preview..."' in response.text
    assert "Fetching playlists, classifying songs, and preparing the review screen." in response.text


def test_finish_page_requires_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSettingsService:
        def get_settings(self):
            return SetupSettings(
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_openai_api_key="secret",
                azure_openai_deployment="gpt-5.4",
                google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
                session_secret="secret",
            )

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.google_token_payload", lambda request: None)

    client = TestClient(app)
    response = client.get("/finish", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_finish_page_renders_start_over_action(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSettingsService:
        def get_settings(self):
            return SetupSettings(
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_openai_api_key="secret",
                azure_openai_deployment="gpt-5.4",
                google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
                session_secret="secret",
            )

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.google_token_payload", lambda request: {"access_token": "token"})

    client = TestClient(app)
    response = client.get("/finish")

    assert response.status_code == 200
    assert "Moods applied" in response.text
    assert "Start once more" in response.text
    assert 'href="/preview"' in response.text
    assert "reloads the playlist list" in response.text


def test_google_callback_redirects_to_preview(monkeypatch: pytest.MonkeyPatch) -> None:
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

        def build_authorization_url(self, redirect_uri):
            return "https://accounts.example/auth", "state-123", "verifier-123"

        def exchange_code(self, code, state, redirect_uri, code_verifier):
            assert code == "code-123"
            assert state == "state-123"
            assert code_verifier == "verifier-123"
            return {"access_token": "token"}

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.YouTubeService", FakeYouTubeService)

    client = TestClient(app)
    connect_response = client.post(
        "/auth/google/connect",
        data={"policy_agreement": "accepted"},
        follow_redirects=False,
    )
    assert connect_response.status_code == 303

    callback_response = client.get(
        "/auth/google/callback?code=code-123&state=state-123",
        follow_redirects=False,
    )

    assert callback_response.status_code == 303
    assert callback_response.headers["location"] == "/preview"


def test_google_connect_requires_policy_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
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

        def build_authorization_url(self, redirect_uri):
            raise AssertionError("OAuth should not start without policy agreement.")

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.YouTubeService", FakeYouTubeService)

    client = TestClient(app)
    response = client.post("/auth/google/connect", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_google_disconnect_revokes_token_and_deletes_local_youtube_data(monkeypatch: pytest.MonkeyPatch) -> None:
    revoked_payloads = []

    class FakeSettingsService:
        def get_settings(self):
            return SetupSettings(
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_openai_api_key="secret",
                azure_openai_deployment="gpt-5.4",
                google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
                session_secret="secret",
            )

    class FakeDb:
        deleted = False

        def delete_authorized_youtube_data(self):
            self.deleted = True

    class FakeYouTubeService:
        def __init__(self, settings, db, token_payload=None):
            self.token_payload = token_payload

        def revoke_token(self):
            revoked_payloads.append(self.token_payload)

    fake_db = FakeDb()
    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.google_token_payload", lambda request: {"refresh_token": "refresh-token"})
    monkeypatch.setattr("app.main.db", fake_db)
    monkeypatch.setattr("app.main.YouTubeService", FakeYouTubeService)

    client = TestClient(app)
    client.cookies.set("ytmp_google_token", "encrypted-token")

    response = client.post("/auth/google/disconnect", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "ytmp_google_token" in response.headers["set-cookie"]
    assert "Max-Age=0" in response.headers["set-cookie"]
    assert revoked_payloads == [{"refresh_token": "refresh-token"}]
    assert fake_db.deleted is True


def test_google_disconnect_deletes_local_data_even_when_revoke_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSettingsService:
        def get_settings(self):
            return SetupSettings(
                azure_openai_endpoint="https://example.openai.azure.com",
                azure_openai_api_key="secret",
                azure_openai_deployment="gpt-5.4",
                google_client_secrets_json=GOOGLE_CLIENT_SECRETS_JSON,
                session_secret="secret",
            )

    class FakeDb:
        deleted = False

        def delete_authorized_youtube_data(self):
            self.deleted = True

    class FakeYouTubeService:
        def __init__(self, settings, db, token_payload=None):
            pass

        def revoke_token(self):
            raise YouTubeAuthError("mock revoke failure")

    fake_db = FakeDb()
    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.google_token_payload", lambda request: {"refresh_token": "refresh-token"})
    monkeypatch.setattr("app.main.db", fake_db)
    monkeypatch.setattr("app.main.YouTubeService", FakeYouTubeService)

    client = TestClient(app)
    response = client.post("/auth/google/disconnect", follow_redirects=False)

    assert response.status_code == 303
    assert "ytmp_google_token" in response.headers["set-cookie"]
    assert fake_db.deleted is True


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

        def create_preview(self, scope, source_playlist_id=None, source_playlist_ids=None, persist=True):
            assert scope == RunScope.SELECTED_PLAYLISTS
            assert source_playlist_ids == ["playlist-1"]
            raise AzureClassificationError("mock classification failure")

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.YouTubeService", FakeYouTubeService)
    monkeypatch.setattr("app.main.OrganizerService", FakeOrganizerService)

    client = TestClient(app)
    response = client.post(
        "/runs/preview",
        data={"scope": "selected_playlists", "selected_playlist_ids": "playlist-1"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/preview"


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


def test_apply_success_redirects_to_finish(monkeypatch: pytest.MonkeyPatch) -> None:
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
            assert run_id == "run-123"
            assert overrides["video-1"] == ["Happy / Feel-good"]
            return {"total_assignments": 1, "playlists": {}}

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
    assert response.headers["location"] == "/finish"


def test_preview_response_includes_apply_loading_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
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

        def create_preview(self, scope, source_playlist_id=None, source_playlist_ids=None, persist=True):
            assert scope == RunScope.SELECTED_PLAYLISTS
            assert source_playlist_ids == ["playlist-1", "playlist-2"]
            return RunDetail(
                run_id="run-123",
                status=RunStatus.PREVIEWED,
                scope=RunScope.SELECTED_PLAYLISTS,
                created_at="2026-04-28T00:00:00+00:00",
                summary=RunSummary(
                    total_candidates=1,
                    classified_count=1,
                    default_included_count=1,
                    excluded_count=0,
                ),
                items=[
                    RunItemView(
                        video_id="video-1",
                        title="Song One",
                        channel_title="Artist One",
                        description="desc",
                        source_playlists=["Road trip"],
                        source_positions=[0],
                        suggested_moods=[MoodLabel.HAPPY],
                        final_moods=[MoodLabel.HAPPY],
                        confidence=92,
                        reason="Upbeat metadata.",
                        is_music=True,
                        default_included=True,
                    )
                ],
            )

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.YouTubeService", FakeYouTubeService)
    monkeypatch.setattr("app.main.OrganizerService", FakeOrganizerService)

    client = TestClient(app)
    response = client.post(
        "/runs/preview",
        data={
            "scope": "selected_playlists",
            "selected_playlist_ids": ["playlist-1", "playlist-2"],
        },
    )

    assert response.status_code == 200
    assert "Preview ready" in response.text
    assert "data-progress-form" in response.text
    assert 'data-progress-start-url="/runs/apply/start"' in response.text
    assert 'data-loading-title="Syncing to YouTube..."' in response.text
    assert "Creating or updating mood playlists and applying your reviewed assignments." in response.text


def test_apply_start_returns_job_and_status_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    class ImmediateExecutor:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)
            return None

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

        def apply_run(self, run_id, overrides, progress_callback=None):
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "syncing",
                        "message": "Syncing Happy / Feel-good",
                        "current": 1,
                        "total": 1,
                        "percent": 75,
                    }
                )
            return {"total_assignments": 1, "playlists": {}}

    monkeypatch.setattr("app.main.settings_service", FakeSettingsService())
    monkeypatch.setattr("app.main.google_token_payload", lambda request: {"access_token": "token"})
    monkeypatch.setattr("app.main.YouTubeService", FakeYouTubeService)
    monkeypatch.setattr("app.main.OrganizerService", FakeOrganizerService)
    monkeypatch.setattr("app.main.apply_executor", ImmediateExecutor())

    client = TestClient(app)
    response = client.post(
        "/runs/apply/start",
        data={"run_id": "run-123", "mood__video-1": "Happy / Feel-good"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"]
    assert payload["status_url"].startswith("/runs/apply/status/")

    status_response = client.get(payload["status_url"])
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["status"] == "complete"
    assert status_payload["percent"] == 100
    assert status_payload["finish_url"] == "/finish"
