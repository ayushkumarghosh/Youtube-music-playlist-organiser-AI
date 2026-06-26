"""YouTube OAuth and playlist operations."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.constants import (
    APP_MANAGED_MARKER,
    APP_NAME,
    APP_PLAYLIST_PREFIX,
    LEGACY_APP_MANAGED_MARKERS,
    PLAYLIST_ITEMS_PAGE_SIZE,
    YOUTUBE_API_RETRY_ATTEMPTS,
)
from app.db import Database
from app.models import (
    CategorySetDefinition,
    PlaylistItemRecord,
    PlaylistSummary,
    RunScope,
    SetupSettings,
    default_mood_category_set,
    slugify_label,
)


GOOGLE_SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
GOOGLE_PROVIDER = "google"
GOOGLE_TOKEN_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
logger = logging.getLogger(__name__)


class YouTubeAuthError(RuntimeError):
    """Raised when YouTube OAuth state is invalid."""


class YouTubeSyncError(RuntimeError):
    """Raised when YouTube playlist sync fails."""


def is_managed_playlist(title: str, description: str = "") -> bool:
    managed_markers = [APP_MANAGED_MARKER, *LEGACY_APP_MANAGED_MARKERS]
    return title.startswith(f"{APP_PLAYLIST_PREFIX} [") or any(marker in description for marker in managed_markers)


def build_managed_playlist_title(scope: RunScope, mood: str, source_playlist_title: str | None = None) -> str:
    return mood


def build_managed_playlist_description(
    scope: RunScope,
    mood: str,
    source_playlist_title: str | None = None,
    category_set: CategorySetDefinition | None = None,
    label_slug: str | None = None,
) -> str:
    if scope == RunScope.ALL_PLAYLISTS:
        source_label = "All playlists"
    elif scope == RunScope.SELECTED_PLAYLISTS:
        source_label = source_playlist_title or "Selected playlists"
    else:
        source_label = source_playlist_title or "Selected playlist"
    category_set = category_set or default_mood_category_set()
    label_slug = label_slug or slugify_label(mood)
    legacy_mood = f" Mood: {mood}." if category_set.id == "mood" else ""
    return (
        f"{APP_MANAGED_MARKER} Managed by {APP_NAME}. "
        f"Category: {category_set.name}. Category ID: {category_set.id}. "
        f"Label: {mood}. Label slug: {label_slug}. "
        f"Scope: {source_label}.{legacy_mood}"
    )


def extract_managed_playlist_mood(title: str, description: str = "") -> str | None:
    if any(marker in description for marker in [APP_MANAGED_MARKER, *LEGACY_APP_MANAGED_MARKERS]):
        marker = "Mood: "
        start = description.find(marker)
        if start != -1:
            mood_text = description[start + len(marker) :].strip()
            return mood_text[:-1] if mood_text.endswith(".") else mood_text
    if title.startswith(f"{APP_PLAYLIST_PREFIX} [") and " - " in title:
        return title.rsplit(" - ", 1)[-1].strip() or None
    return None


def _extract_description_value(description: str, marker: str) -> str | None:
    start = description.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = description.find(".", start)
    value = description[start:end if end != -1 else None].strip()
    return value or None


def extract_managed_playlist_category_key(title: str, description: str = "") -> str | None:
    if any(marker in description for marker in [APP_MANAGED_MARKER, *LEGACY_APP_MANAGED_MARKERS]):
        category_id = _extract_description_value(description, "Category ID: ")
        label_slug = _extract_description_value(description, "Label slug: ")
        if category_id and label_slug:
            return f"{slugify_label(category_id)}:{slugify_label(label_slug)}"
        legacy_mood = extract_managed_playlist_mood(title, description)
        if legacy_mood:
            mood_label = default_mood_category_set().label_for_slug(legacy_mood)
            return f"mood:{mood_label.slug if mood_label else slugify_label(legacy_mood)}"
    legacy_mood = extract_managed_playlist_mood(title, description)
    if legacy_mood:
        mood_label = default_mood_category_set().label_for_slug(legacy_mood)
        return f"mood:{mood_label.slug if mood_label else slugify_label(legacy_mood)}"
    return None


class YouTubeService:
    def __init__(
        self,
        settings: SetupSettings,
        db: Database,
        token_payload: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.token_payload = token_payload

    def _client_config(self) -> dict[str, Any]:
        try:
            config = json.loads(self.settings.google_client_secrets_json)
        except json.JSONDecodeError as exc:
            raise YouTubeAuthError("GOOGLE_CLIENT_SECRETS_JSON must be valid JSON.") from exc
        if not isinstance(config, dict) or not any(key in config for key in ("web", "installed")):
            raise YouTubeAuthError("GOOGLE_CLIENT_SECRETS_JSON must be a Google OAuth client JSON object.")
        return config

    def has_token(self) -> bool:
        return self.token_payload is not None

    def build_authorization_url(self, redirect_uri: str) -> tuple[str, str, str | None]:
        flow = Flow.from_client_config(
            self._client_config(),
            scopes=GOOGLE_SCOPES,
            autogenerate_code_verifier=True,
        )
        flow.redirect_uri = redirect_uri
        auth_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return auth_url, state, flow.code_verifier

    def exchange_code(
        self,
        code: str,
        state: str,
        redirect_uri: str,
        code_verifier: str | None = None,
    ) -> dict[str, Any]:
        flow = Flow.from_client_config(
            self._client_config(),
            scopes=GOOGLE_SCOPES,
            state=state,
            autogenerate_code_verifier=False,
        )
        flow.redirect_uri = redirect_uri
        flow.code_verifier = code_verifier
        flow.fetch_token(code=code)
        creds = flow.credentials
        payload = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
        }
        self.token_payload = payload
        return payload

    def _credentials(self) -> Credentials:
        if not self.token_payload:
            raise YouTubeAuthError("YouTube is not connected.")
        return Credentials.from_authorized_user_info(self.token_payload, GOOGLE_SCOPES)

    def _client(self):
        return build("youtube", "v3", credentials=self._credentials(), cache_discovery=False)

    def revoke_token(self) -> None:
        if not self.token_payload:
            return
        token = self.token_payload.get("refresh_token") or self.token_payload.get("token")
        if not token:
            return
        body = urlencode({"token": token}).encode("utf-8")
        request = Request(
            GOOGLE_TOKEN_REVOKE_URL,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10):
                return
        except HTTPError as exc:
            if exc.code == 400:
                return
            raise YouTubeAuthError(f"Could not revoke Google authorization: HTTP {exc.code}.") from exc
        except URLError as exc:
            raise YouTubeAuthError(f"Could not revoke Google authorization: {exc.reason}.") from exc

    def list_playlists(self, include_managed: bool = False) -> list[PlaylistSummary]:
        youtube = self._client()
        playlists: list[PlaylistSummary] = []
        page_token: str | None = None
        while True:
            response = self._execute_request(
                lambda: youtube.playlists().list(
                    part="snippet,contentDetails,status",
                    mine=True,
                    maxResults=50,
                    pageToken=page_token,
                ),
                "listing YouTube playlists",
            )
            for item in response.get("items", []):
                title = item["snippet"]["title"]
                description = item["snippet"].get("description", "")
                if not include_managed and is_managed_playlist(title, description):
                    continue
                playlists.append(
                    PlaylistSummary(
                        playlist_id=item["id"],
                        title=title,
                        description=description,
                        privacy_status=item["status"].get("privacyStatus", ""),
                        item_count=item["contentDetails"].get("itemCount", 0),
                    )
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return playlists

    def list_playlist_items(self, playlist_id: str, playlist_title: str) -> list[PlaylistItemRecord]:
        youtube = self._client()
        items: list[PlaylistItemRecord] = []
        page_token: str | None = None
        while True:
            response = self._execute_request(
                lambda: youtube.playlistItems().list(
                    part="snippet,contentDetails,status",
                    playlistId=playlist_id,
                    maxResults=PLAYLIST_ITEMS_PAGE_SIZE,
                    pageToken=page_token,
                ),
                f"listing items for playlist {playlist_id}",
            )
            for item in response.get("items", []):
                snippet = item["snippet"]
                resource_id = snippet.get("resourceId", {})
                video_id = resource_id.get("videoId")
                if not video_id:
                    continue
                items.append(
                    PlaylistItemRecord(
                        playlist_item_id=item["id"],
                        playlist_id=playlist_id,
                        playlist_title=playlist_title,
                        video_id=video_id,
                        title=snippet.get("title", ""),
                        description=snippet.get("description", ""),
                        channel_title=snippet.get("videoOwnerChannelTitle")
                        or snippet.get("channelTitle", ""),
                        position=snippet.get("position", 0),
                    )
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return items

    def get_source_playlists(
        self,
        scope: RunScope,
        selected_playlist_id: str | None = None,
        selected_playlist_ids: list[str] | None = None,
    ) -> list[PlaylistSummary]:
        playlists = self.list_playlists(include_managed=False)
        if scope == RunScope.ALL_PLAYLISTS:
            return playlists
        if scope == RunScope.SELECTED_PLAYLISTS:
            selected_ids = set(selected_playlist_ids or [])
            return [playlist for playlist in playlists if playlist.playlist_id in selected_ids]
        return [playlist for playlist in playlists if playlist.playlist_id == selected_playlist_id]

    def ensure_managed_playlists(
        self,
        scope: RunScope,
        source_playlist_title: str | None,
        category_sets: list[CategorySetDefinition] | None = None,
    ) -> dict[str, PlaylistSummary]:
        category_sets = category_sets or [default_mood_category_set()]
        youtube = self._client()
        existing_by_key: dict[str, PlaylistSummary] = {}
        for playlist in self.list_playlists(include_managed=True):
            if not is_managed_playlist(playlist.title, playlist.description):
                continue
            key = extract_managed_playlist_category_key(playlist.title, playlist.description)
            if key and key not in existing_by_key:
                existing_by_key[key] = playlist
        result: dict[str, PlaylistSummary] = {}
        for category_set in category_sets:
            for label in category_set.labels:
                key = f"{category_set.id}:{label.slug}"
                title = build_managed_playlist_title(scope, label.name, source_playlist_title)
                description = build_managed_playlist_description(
                    scope,
                    label.name,
                    source_playlist_title,
                    category_set,
                    label.slug,
                )
                playlist = existing_by_key.get(key)
                if playlist is None:
                    response = self._execute_request(
                        lambda: youtube.playlists().insert(
                            part="snippet,status",
                            body={
                                "snippet": {
                                    "title": title,
                                    "description": description,
                                },
                                "status": {"privacyStatus": "private"},
                            },
                        ),
                        f"creating managed playlist '{title}'",
                    )
                    playlist = PlaylistSummary(
                        playlist_id=response["id"],
                        title=response["snippet"]["title"],
                        description=response["snippet"].get("description", ""),
                        privacy_status=response["status"].get("privacyStatus", ""),
                        item_count=response.get("contentDetails", {}).get("itemCount", 0),
                    )
                elif (
                    playlist.title != title
                    or playlist.description != description
                    or playlist.privacy_status != "private"
                ):
                    response = self._execute_request(
                        lambda: youtube.playlists().update(
                            part="snippet,status",
                            body={
                                "id": playlist.playlist_id,
                                "snippet": {
                                    "title": title,
                                    "description": description,
                                },
                                "status": {"privacyStatus": "private"},
                            },
                        ),
                        f"updating managed playlist '{playlist.playlist_id}'",
                    )
                    playlist = PlaylistSummary(
                        playlist_id=response["id"],
                        title=response["snippet"]["title"],
                        description=response["snippet"].get("description", ""),
                        privacy_status=response["status"].get("privacyStatus", ""),
                        item_count=response.get("contentDetails", {}).get("itemCount", playlist.item_count),
                    )
                result[key] = playlist
        return result

    def reconcile_playlist(
        self,
        playlist_id: str,
        desired_video_ids: list[str],
    ) -> dict[str, int]:
        youtube = self._client()
        existing_records = self._fetch_playlist_records(youtube, playlist_id)
        existing_video_ids = {record["video_id"] for record in existing_records}
        desired_unique_video_ids = list(dict.fromkeys(desired_video_ids))

        inserts = 0
        for video_id in desired_unique_video_ids:
            if video_id in existing_video_ids:
                continue
            self._execute_request(
                lambda: youtube.playlistItems().insert(
                    part="snippet",
                    body={
                        "snippet": {
                            "playlistId": playlist_id,
                            "resourceId": {"kind": "youtube#video", "videoId": video_id},
                        }
                    },
                ),
                f"appending video {video_id} to playlist {playlist_id}",
            )
            inserts += 1

        return {"deletes": 0, "inserts": inserts, "updates": 0}

    def _fetch_playlist_records(self, youtube: Any, playlist_id: str) -> list[dict[str, Any]]:
        items = []
        page_token: str | None = None
        while True:
            response = self._execute_request(
                lambda: youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=PLAYLIST_ITEMS_PAGE_SIZE,
                    pageToken=page_token,
                ),
                f"fetching playlist records for {playlist_id}",
            )
            for item in response.get("items", []):
                snippet = item["snippet"]
                video_id = snippet.get("resourceId", {}).get("videoId")
                if not video_id:
                    continue
                items.append(
                    {
                        "playlist_item_id": item["id"],
                        "video_id": video_id,
                        "position": snippet.get("position", 0),
                    }
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return items

    def _execute_request(self, request_factory, operation: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, YOUTUBE_API_RETRY_ATTEMPTS + 1):
            try:
                return request_factory().execute()
            except HttpError as exc:
                last_error = exc
                if not self._is_retryable_http_error(exc) or attempt == YOUTUBE_API_RETRY_ATTEMPTS:
                    raise YouTubeSyncError(self._format_http_error(operation, exc)) from exc
                delay_seconds = 2 ** (attempt - 1)
                logger.warning(
                    "Retrying YouTube API request after transient failure",
                    extra={
                        "operation": operation,
                        "attempt": attempt,
                        "max_attempts": YOUTUBE_API_RETRY_ATTEMPTS,
                        "status": getattr(exc.resp, "status", None),
                    },
                )
                time.sleep(delay_seconds)
            except Exception as exc:
                last_error = exc
                raise YouTubeSyncError(f"YouTube request failed while {operation}: {exc}") from exc
        raise YouTubeSyncError(f"YouTube request failed while {operation}: {last_error}")

    def _is_retryable_http_error(self, exc: HttpError) -> bool:
        status = getattr(exc.resp, "status", None)
        reasons, _ = self._extract_http_error_details(exc)
        retryable_reasons = {
            "SERVICE_UNAVAILABLE",
            "backendError",
            "internalError",
            "rateLimitExceeded",
            "userRateLimitExceeded",
        }
        return bool(
            status in {409, 500, 502, 503, 504}
            or any(reason in retryable_reasons for reason in reasons)
        )

    def _format_http_error(self, operation: str, exc: HttpError) -> str:
        status = getattr(exc.resp, "status", "unknown")
        reasons, message = self._extract_http_error_details(exc)
        reason_text = f" ({', '.join(reasons)})" if reasons else ""
        return f"YouTube API error while {operation}: HTTP {status}{reason_text}. {message}"

    def _extract_http_error_details(self, exc: HttpError) -> tuple[list[str], str]:
        default_message = str(exc)
        try:
            content = exc.content.decode("utf-8") if isinstance(exc.content, bytes) else str(exc.content)
            payload = json.loads(content)
        except Exception:
            return [], default_message

        error_block = payload.get("error", {})
        details = error_block.get("errors", [])
        reasons = [detail.get("reason", "") for detail in details if detail.get("reason")]
        message = error_block.get("message") or default_message
        return reasons, message
