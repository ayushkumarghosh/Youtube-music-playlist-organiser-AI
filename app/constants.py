"""Static application constants."""

from __future__ import annotations

APP_MANAGED_MARKER = "[vibeshelf-managed]"
LEGACY_APP_MANAGED_MARKERS = ["[yt-mood-organizer-managed]"]
APP_PLAYLIST_PREFIX = "Mood"
APP_NAME = "VibeShelf"
APP_DATA_DIR = "data"
DB_FILENAME = "app.db"
TOKENS_FILENAME = "google_token.json"
SESSION_SECRET_DEFAULT = "local-dev-session-secret-change-me"
PROMPT_VERSION = "2026-06-26.v4-categories"
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
CATEGORY_SET_MOOD_ID = "mood"
CUSTOM_CATEGORY_ID_PREFIX = "custom-"

BUILT_IN_CATEGORY_SETS = [
    {
        "id": CATEGORY_SET_MOOD_ID,
        "name": "Mood",
        "description": "Emotional tone and listener feeling.",
        "labels": [
            {
                "slug": "happy-feel-good",
                "name": "Happy / Feel-good",
                "description": "Bright, positive, uplifting, playful, or optimistic songs.",
            },
            {
                "slug": "sad-emotional",
                "name": "Sad / Emotional",
                "description": "Melancholy, vulnerable, reflective, grieving, or tearful songs.",
            },
            {
                "slug": "romantic-love",
                "name": "Romantic / Love",
                "description": "Love, affection, longing, intimacy, heartbreak, or relationship songs.",
            },
            {
                "slug": "chill-relaxing",
                "name": "Chill / Relaxing",
                "description": "Calm, mellow, soothing, laid-back, ambient, or easygoing songs.",
            },
            {
                "slug": "energetic-hype",
                "name": "Energetic / Hype",
                "description": "Fast, motivating, anthemic, workout-ready, or high-intensity songs.",
            },
            {
                "slug": "dark-intense",
                "name": "Dark / Intense",
                "description": "Heavy, ominous, aggressive, brooding, dramatic, or suspenseful songs.",
            },
        ],
    },
    {
        "id": "activity",
        "name": "Activity",
        "description": "Common listening contexts and use cases.",
        "labels": [
            {"slug": "workout", "name": "Workout", "description": "Training, gym, running, or high-motion listening."},
            {"slug": "focus-study", "name": "Focus/Study", "description": "Concentration, study, work, or low-distraction listening."},
            {"slug": "driving-road-trip", "name": "Driving/Road Trip", "description": "Travel, cruising, road trips, or car-friendly listening."},
            {"slug": "party-social", "name": "Party/Social", "description": "Group listening, dancing, celebration, or social energy."},
            {"slug": "sleep", "name": "Sleep", "description": "Gentle, quiet, restful, or winding-down listening."},
            {"slug": "cooking-housework", "name": "Cooking/Housework", "description": "Casual background listening for chores or home routines."},
        ],
    },
    {
        "id": "genre",
        "name": "Genre",
        "description": "Broad musical style.",
        "labels": [
            {"slug": "pop", "name": "Pop", "description": "Mainstream pop, radio pop, or pop-adjacent songs."},
            {"slug": "rock", "name": "Rock", "description": "Rock, metal, punk, or guitar-forward songs."},
            {"slug": "hip-hop-rap", "name": "Hip-Hop/Rap", "description": "Rap, hip-hop, trap, drill, or MC-led songs."},
            {"slug": "electronic-dance", "name": "Electronic/Dance", "description": "EDM, house, techno, synth, or electronic production-led songs."},
            {"slug": "r-b-soul", "name": "R&B/Soul", "description": "R&B, soul, funk, gospel-influenced, or groove-led songs."},
            {"slug": "indie-alternative", "name": "Indie/Alternative", "description": "Indie, alternative, experimental pop, or non-mainstream rock/pop."},
            {"slug": "classical-instrumental", "name": "Classical/Instrumental", "description": "Classical, score, instrumental, orchestral, or no-vocal pieces."},
            {"slug": "jazz-blues", "name": "Jazz/Blues", "description": "Jazz, blues, swing, big band, or improvisational songs."},
            {"slug": "country-folk", "name": "Country/Folk", "description": "Country, folk, acoustic singer-songwriter, or roots music."},
        ],
    },
    {
        "id": "era",
        "name": "Era",
        "description": "Likely release decade or period when metadata supports it.",
        "labels": [
            {"slug": "1970s-or-earlier", "name": "1970s or earlier", "description": "Songs from the 1970s or any earlier period."},
            {"slug": "1980s", "name": "1980s", "description": "Songs from the 1980s."},
            {"slug": "1990s", "name": "1990s", "description": "Songs from the 1990s."},
            {"slug": "2000s", "name": "2000s", "description": "Songs from the 2000s."},
            {"slug": "2010s", "name": "2010s", "description": "Songs from the 2010s."},
            {"slug": "2020s", "name": "2020s", "description": "Songs from the 2020s."},
        ],
    },
    {
        "id": "energy",
        "name": "Energy",
        "description": "Overall intensity and movement level.",
        "labels": [
            {"slug": "low-energy", "name": "Low Energy", "description": "Slow, sparse, gentle, quiet, or minimal songs."},
            {"slug": "medium-energy", "name": "Medium Energy", "description": "Moderate tempo or balanced intensity songs."},
            {"slug": "high-energy", "name": "High Energy", "description": "Fast, loud, driving, intense, or highly active songs."},
        ],
    },
]

MOOD_LABELS = [label["name"] for label in BUILT_IN_CATEGORY_SETS[0]["labels"]]
