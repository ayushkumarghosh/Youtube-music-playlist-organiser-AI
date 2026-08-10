# VibeShelf

VibeShelf is a FastAPI web app that turns songs from your YouTube playlists into private, AI-organized playlists. Choose one or more source playlists, select the category labels you want, review Azure OpenAI's suggestions, and apply the approved assignments back to YouTube.

Songs can belong to more than one label, and the app never writes to YouTube until you confirm the preview.

## Features

- Connects to a user's YouTube account with Google OAuth and PKCE.
- Reads one or more source playlists and deduplicates songs that appear in several of them.
- Classifies songs in batches with the Azure OpenAI Responses API and structured output.
- Includes built-in category sets for mood, activity, genre, era, and energy.
- Lets you select individual labels instead of generating every playlist in a category set.
- Generates reusable custom category sets from a natural-language prompt.
- Provides a review screen where every suggested label can be added or removed before syncing.
- Creates or reuses private VibeShelf-managed playlists and appends missing songs without deleting existing items.
- Caches classifications in SQLite using the song metadata, category definitions, model, and prompt version.
- Includes public Terms and Privacy pages plus a disconnect flow that revokes Google access and removes locally stored YouTube-derived data.

## How it works

1. Connect a Google account that has access to the source YouTube playlists.
2. Select the source playlists and the category labels to use.
3. Optionally ask Azure OpenAI to propose a custom category containing 2–12 labels, then edit and save it.
4. Generate a preview. VibeShelf fetches playlist items, deduplicates them by video ID, and classifies their metadata.
5. Review the assignments and adjust any labels.
6. Apply the result. VibeShelf creates or updates private managed playlists and adds missing videos.

Managed playlists are identified by a VibeShelf marker in their descriptions. Sync is currently append-only: applying a later run adds missing videos but does not remove videos already in a managed playlist.

## Built-in categories

| Category | Labels |
| --- | --- |
| Mood | Happy / Feel-good, Sad / Emotional, Romantic / Love, Chill / Relaxing, Energetic / Hype, Dark / Intense |
| Activity | Workout, Focus/Study, Driving/Road Trip, Party/Social, Sleep, Cooking/Housework |
| Genre | Pop, Rock, Hip-Hop/Rap, Electronic/Dance, R&B/Soul, Indie/Alternative, Classical/Instrumental, Jazz/Blues, Country/Folk |
| Era | 1970s or earlier, 1980s, 1990s, 2000s, 2010s, 2020s |
| Energy | Low Energy, Medium Energy, High Energy |

## Requirements

- Python 3.11 or newer
- A Google Cloud project with the YouTube Data API v3 enabled
- A Google OAuth client configuration
- An Azure OpenAI resource and a deployment that supports the Responses API, reasoning, and structured outputs

## Google Cloud setup

1. Enable **YouTube Data API v3** in your Google Cloud project.
2. Configure the OAuth consent screen and add any required test users while the app is in testing mode.
3. Create an OAuth client, download its JSON configuration, and keep it outside version control.
4. Add the callback URL as an authorized redirect URI:

   ```text
   http://127.0.0.1:8000/auth/google/callback
   ```

   For a deployed app, also add:

   ```text
   https://<your-domain>/auth/google/callback
   ```

5. For a public deployment, add `https://<your-domain>/privacy` to the consent screen's privacy-policy field.

VibeShelf requests the `https://www.googleapis.com/auth/youtube.force-ssl` scope so it can read playlists and create or update private managed playlists after confirmation.

## Configuration

All credentials are read from environment variables. They are never collected by the browser UI.

| Variable | Required | Description |
| --- | --- | --- |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI resource root, such as `https://your-resource.openai.azure.com`. Do not include `/openai/v1`; the app adds it. |
| `AZURE_OPENAI_API_KEY` | Yes | API key for the Azure OpenAI resource. |
| `AZURE_OPENAI_DEPLOYMENT` | Yes | Azure deployment name, for example `gpt-5.4`. |
| `GOOGLE_CLIENT_SECRETS_JSON` | Yes | Full contents of the downloaded Google OAuth client JSON file. |
| `SESSION_SECRET` | Production | Long random value used to sign sessions and encrypt OAuth/run state. Local development falls back to an insecure built-in value, so set this for any shared or deployed instance. |
| `APP_BASE_URL` | No | Public origin used to construct the OAuth callback, such as `https://vibeshelf.example.com`. If omitted, the request origin is used. |

PowerShell example:

```powershell
$env:AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
$env:AZURE_OPENAI_API_KEY="replace-with-your-api-key"
$env:AZURE_OPENAI_DEPLOYMENT="gpt-5.4"
$env:GOOGLE_CLIENT_SECRETS_JSON=(Get-Content .\secrets\client_secret.json -Raw)
$env:SESSION_SECRET=(-join ((1..64) | ForEach-Object { '{0:x}' -f (Get-Random -Maximum 16) }))
$env:APP_BASE_URL="http://127.0.0.1:8000"
```

Do not commit client secrets, API keys, session secrets, the local `data/` directory, or `.env` files. These paths are already covered by `.gitignore`.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), accept the Terms and Privacy Policy, connect YouTube, and generate a preview.

## Tests

Install the development dependencies and run:

```powershell
python -m pytest
```

The tests mock YouTube and Azure OpenAI interactions; they do not require live cloud credentials.

## Deploy to Vercel

The repository includes `vercel.json` and `api/index.py`, which route requests to the FastAPI app.

1. Import the repository into Vercel.
2. Add all required environment variables from the configuration table.
3. Set `APP_BASE_URL` to the production origin, without a trailing path.
4. Add `https://<your-vercel-domain>/auth/google/callback` to the Google OAuth client's authorized redirect URIs.
5. Add the deployed `/privacy` URL to the Google OAuth consent screen.

Vercel Functions do not provide durable local storage. On Vercel, SQLite is placed in the function's temporary directory, so the classification cache and saved custom categories can disappear between instances or deployments. Preview/apply state is carried as encrypted browser-submitted state, and the encrypted Google token is stored in an HttpOnly cookie.

For durable cache or custom-category storage in production, replace the local SQLite layer with a persistent database.

## Data and security notes

- The Google OAuth token payload is encrypted and stored in an HttpOnly, `SameSite=Lax` cookie. The cookie is also marked `Secure` on HTTPS requests.
- Preview state is encrypted before being returned to the browser and is validated when it is submitted for apply.
- Song metadata is sent to the configured Azure OpenAI resource for classification. API requests use `store=False`.
- Locally, classification cache entries and custom category definitions are stored in `data/app.db`.
- Disconnecting attempts to revoke the Google token, clears the token cookie, and deletes locally stored runs and classification cache entries derived from YouTube.
- Managed playlists are always created or updated as private playlists.
- The app does not upload videos, publish public playlists, or share one user's playlists with another user.

See the running app's `/privacy` and `/terms` pages for the user-facing policies.

## Project structure

```text
api/index.py                 Vercel serverless entrypoint
app/main.py                  FastAPI routes, OAuth, preview, and apply workflow
app/config.py                Environment-backed settings and runtime paths
app/constants.py             Category definitions and tuning values
app/db.py                    SQLite persistence and classification cache
app/models.py                Domain and structured-output models
app/security.py              Encrypted JSON state helpers
app/services/azure_openai.py Azure OpenAI classification and category generation
app/services/organizer.py    Preview construction and playlist sync orchestration
app/services/youtube.py      YouTube OAuth and playlist operations
static/                      Browser JavaScript and styles
templates/                   Jinja pages
tests/                       Route and service tests
```

## License

No license file is currently included. Unless a license is added, the repository's code remains under its default copyright protections.
