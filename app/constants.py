"""Static application constants."""

from __future__ import annotations

APP_MANAGED_MARKER = "[yt-mood-organizer-managed]"
APP_PLAYLIST_PREFIX = "Mood"
APP_NAME = "YouTube Mood Playlist Organizer"
APP_DATA_DIR = "data"
DB_FILENAME = "app.db"
TOKENS_FILENAME = "google_token.json"
SESSION_SECRET_DEFAULT = "local-dev-session-secret-change-me"
PROMPT_VERSION = "2026-04-22.v3-multi-mood"
CLASSIFICATION_RETRY_ATTEMPTS = 3
PLAYLIST_ITEMS_PAGE_SIZE = 50
YOUTUBE_API_RETRY_ATTEMPTS = 4
SYSTEM_REASONING_EFFORT = "low"
CLASSIFICATION_MAX_OUTPUT_TOKENS = 400
CLASSIFICATION_BATCH_MIN_SONGS = 1000
CLASSIFICATION_BATCH_MAX_SONGS = 2000
CLASSIFICATION_INPUT_SOFT_TOKEN_BUDGET = 120000
CLASSIFICATION_OUTPUT_TOKEN_RESERVE_PER_SONG = 48
CLASSIFICATION_DESCRIPTION_CHAR_LIMIT = 400

MOOD_LABELS = [
    "Happy / Feel-good",
    "Sad / Emotional",
    "Romantic / Love",
    "Chill / Relaxing",
    "Energetic / Hype",
    "Dark / Intense",
]
