# Project Structure

This repository contains VibeShelf, a FastAPI app that connects to YouTube, classifies songs with Azure OpenAI, and syncs mood-based playlists back to the user's account.

## Source Tree

```text
.
|-- api/
|   `-- index.py                  # Vercel serverless entrypoint
|-- app/
|   |-- __init__.py               # Python package marker
|   |-- config.py                 # Runtime paths and environment-backed settings loader
|   |-- constants.py              # App constants, mood labels, API tuning values
|   |-- db.py                     # SQLite persistence layer
|   |-- main.py                   # FastAPI app, routes, OAuth flow, preview/apply workflow
|   |-- models.py                 # Pydantic domain models and serializers
|   |-- security.py               # Encrypted JSON state helpers
|   `-- services/
|       |-- __init__.py
|       |-- azure_openai.py       # Azure OpenAI song classification and cache integration
|       |-- organizer.py          # Preview creation and playlist sync orchestration
|       |-- settings.py           # Settings validation and retrieval service
|       `-- youtube.py            # YouTube OAuth, playlist reads, playlist writes, revocation
|-- data/
|   `-- app.db                    # Local SQLite runtime database
|-- secrets/
|   |-- client_secret_*.json      # Local Google OAuth client secrets; do not expose
|-- static/
|   |-- app.js                    # Browser-side UI behavior
|   `-- styles.css                # App styling
|-- templates/
|   |-- base.html                 # Shared page layout
|   |-- finish.html               # Completion page
|   |-- index.html                # Login/connect page
|   |-- preview.html              # Playlist selection and preview workspace
|   |-- privacy.html              # YouTube API privacy policy page
|   |-- run_detail.html           # Classification review/apply page
|   `-- terms.html                # Terms page
|-- tests/
|   |-- test_app.py               # FastAPI route and workflow tests
|   `-- test_core.py              # Core service/model/database behavior tests
|-- .gitignore
|-- pyproject.toml                # Package metadata and pytest configuration
|-- README.md                     # Setup, local run, deploy, and policy notes
|-- vercel.json                   # Vercel routing/build configuration
`-- YOUTUBE_API_SERVICES_IMPLEMENTATION.md
```

## Generated Or Local-Only Paths

These paths are present locally but should normally be ignored when reasoning about source structure:

```text
.git/
.pytest_cache/
.venv/
.vercel/
__pycache__/
yt_mood_playlist_organizer.egg-info/
```

## Common Commands

```powershell
python -m pip install -e .[dev]
uvicorn app.main:app --reload
python -m pytest
```

## UI Development Workflow

When changing the UI:

1. Create a mockup for the intended UI change if it is a major change.
2. Test the change by running the app and taking screenshots of the changed screens.
3. Fix the implementation if the UI does not match the mockup. Generated text does not need to match exactly.
4. Repeat steps 2 and 3 until the UI matches the mockup.
