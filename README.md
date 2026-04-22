# YouTube Mood Playlist Organizer

Local FastAPI app that signs into YouTube, classifies songs with Azure OpenAI, and syncs mood-based playlists back to your account. Songs can be assigned to multiple mood playlists when they fit more than one category.

## Requirements

- Python 3.11+
- A Google OAuth client secrets JSON file with YouTube Data API enabled
- An Azure OpenAI deployment configured for GPT-5.4

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`, enter your Azure settings plus the Google OAuth client secrets JSON path, connect your Google account in the browser, then generate a preview. The created playlists should also appear in YouTube Music for the same account.
