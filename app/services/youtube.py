"""YouTube OAuth and playlist operations."""

from __future__ import annotations

from collections import Counter
import json
import logging
import time
from typing import Any

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.constants import (
    APP_MANAGED_MARKER,
    APP_PLAYLIST_PREFIX,
    MOOD_LABELS,
    PLAYLIST_ITEMS_PAGE_SIZE,
    YOUTUBE_API_RETRY_ATTEMPTS,
)
from app.db import Database
from app.models import PlaylistItemRecord, PlaylistSummary, RunScope, SetupSettings


GOOGLE_SCOPES = ["https://www.googleapis.com/auth/youtube"]
GOOGLE_PROVIDER = "google"
logger = logging.getLogger(__name__)


class YouTubeAuthError(RuntimeError):
    """Raised when YouTube OAuth state is invalid."""


class YouTubeSyncError(RuntimeError):
    """Raised when YouTube playlist sync fails."""


def is_managed_playlist(title: str, description: str = "") -> bool:
    return title.startswith(f"{APP_PLAYLIST_PREFIX} [") or APP_MANAGED_MARKER in description


def build_managed_playlist_title(scope: RunScope, mood: str, source_playlist_title: str | None = None) -> str:
    source_label = "All" if scope == RunScope.ALL_PLAYLISTS else (source_playlist_title or "Selected")
    return f"{APP_PLAYLIST_PREFIX} [{source_label}] - {mood}"


def build_managed_playlist_description(scope: RunScope, mood: str, source_playlist_title: str | None = None) -> str:
    source_label = "All playlists" if scope == RunScope.ALL_PLAYLISTS else (source_playlist_title or "Selected playlist")
    return (
        f"{APP_MANAGED_MARKER} Managed by YouTube Mood Playlist Organizer. "
        f"Scope: {source_label}. Mood: {mood}."
    )


class YouTubeService:
    def __init__(self, settings: SetupSettings, db: Database) -> None:
        self.settings = settings
        self.db = db

    def has_token(self) -> bool:
        return self.db.load_token_payload(GOOGLE_PROVIDER) is not None

    def build_authorization_url(self, redirect_uri: str) -> tuple[str, str, str | None]:
        flow = Flow.from_client_secrets_file(
            self.settings.google_client_secrets_file,
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
    ) -> None:
        flow = Flow.from_client_secrets_file(
            self.settings.google_client_secrets_file,
            scopes=GOOGLE_SCOPES,
            state=state,
            autogenerate_code_verifier=False,
        )
        flow.redirect_uri = redirect_uri
        flow.code_verifier = code_verifier
        flow.fetch_token(code=code)
        creds = flow.credentials
        self.db.save_token_payload(
            GOOGLE_PROVIDER,
            {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            },
        )

    def _credentials(self) -> Credentials:
        payload = self.db.load_token_payload(GOOGLE_PROVIDER)
        if not payload:
            raise YouTubeAuthError("YouTube is not connected.")
        return Credentials.from_authorized_user_info(payload, GOOGLE_SCOPES)

    def _client(self):
        return build("youtube", "v3", credentials=self._credentials(), cache_discovery=False)

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
    ) -> list[PlaylistSummary]:
        playlists = self.list_playlists(include_managed=False)
        if scope == RunScope.ALL_PLAYLISTS:
            return playlists
        return [playlist for playlist in playlists if playlist.playlist_id == selected_playlist_id]

    def ensure_managed_playlists(
        self,
        scope: RunScope,
        source_playlist_title: str | None,
    ) -> dict[str, PlaylistSummary]:
        youtube = self._client()
        existing = {
            playlist.title: playlist for playlist in self.list_playlists(include_managed=True)
            if is_managed_playlist(playlist.title, playlist.description)
        }
        result: dict[str, PlaylistSummary] = {}
        for mood in MOOD_LABELS:
            title = build_managed_playlist_title(scope, mood, source_playlist_title)
            playlist = existing.get(title)
            if playlist is None:
                response = self._execute_request(
                    lambda: youtube.playlists().insert(
                        part="snippet,status",
                        body={
                            "snippet": {
                                "title": title,
                                "description": build_managed_playlist_description(
                                    scope,
                                    mood,
                                    source_playlist_title,
                                ),
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
            result[mood] = playlist
        return result

    def reconcile_playlist(
        self,
        playlist_id: str,
        desired_video_ids: list[str],
    ) -> dict[str, int]:
        youtube = self._client()
        existing_records = self._fetch_playlist_records(youtube, playlist_id)

        deletes = 0
        inserts = 0
        updates = 0

        desired_counter = Counter(desired_video_ids)
        seen_counter: Counter[str] = Counter()
        for record in existing_records:
            seen_counter[record["video_id"]] += 1
            if desired_counter[record["video_id"]] < seen_counter[record["video_id"]]:
                self._execute_request(
                    lambda: youtube.playlistItems().delete(id=record["playlist_item_id"]),
                    f"removing video {record['video_id']} from playlist {playlist_id}",
                )
                deletes += 1

        refreshed_records = self._fetch_playlist_records(youtube, playlist_id)
        existing_map = {record["video_id"]: record for record in refreshed_records}
        for video_id in desired_video_ids:
            if video_id in existing_map:
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
                f"adding video {video_id} to playlist {playlist_id}",
            )
            inserts += 1

        refreshed_records = self._fetch_playlist_records(youtube, playlist_id)
        for position, video_id in enumerate(desired_video_ids):
            record = next((row for row in refreshed_records if row["video_id"] == video_id), None)
            if record is None or record["position"] == position:
                continue
            self._execute_request(
                lambda: youtube.playlistItems().update(
                    part="snippet",
                    body={
                        "id": record["playlist_item_id"],
                        "snippet": {
                            "playlistId": playlist_id,
                            "resourceId": {"kind": "youtube#video", "videoId": video_id},
                            "position": position,
                        },
                    },
                ),
                f"reordering video {video_id} in playlist {playlist_id}",
            )
            updates += 1

        return {"deletes": deletes, "inserts": inserts, "updates": updates}

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
