# YouTube Mood Playlist Organizer

FastAPI app that signs into YouTube, classifies songs with Azure OpenAI, and syncs mood-based playlists back to your account. Songs can be assigned to multiple mood playlists when they fit more than one category.

## Requirements

- Python 3.11+
- Google OAuth client JSON with YouTube Data API enabled
- Azure OpenAI deployment configured for GPT-5.4

## Environment

Credentials are loaded only from environment variables. The setup page shows whether each value is configured, but never asks for secrets in the browser.

```powershell
$env:AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
$env:AZURE_OPENAI_API_KEY="..."
$env:AZURE_OPENAI_DEPLOYMENT="gpt-5.4"
$env:GOOGLE_CLIENT_SECRETS_JSON=(Get-Content .\secrets\client_secret.json -Raw)
$env:SESSION_SECRET="replace-with-a-long-random-secret"
$env:APP_BASE_URL="http://127.0.0.1:8000"
```

`APP_BASE_URL` is optional locally, but useful on Vercel so Google OAuth redirects use your production domain.

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`, connect your Google account, then generate a preview.

## Deploy To Vercel

Set these project environment variables in Vercel:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_DEPLOYMENT`
- `GOOGLE_CLIENT_SECRETS_JSON`
- `SESSION_SECRET`
- `APP_BASE_URL`, for example `https://your-project.vercel.app`

In Google Cloud Console, add this authorized redirect URI:

```text
https://<your-vercel-domain>/auth/google/callback
```

Vercel Functions have a read-only deployment filesystem, so this deployment does not keep durable run history or a durable classification cache. The Google login token is kept in an encrypted HttpOnly browser cookie, and the active preview/apply flow is carried in encrypted browser-submitted state.
