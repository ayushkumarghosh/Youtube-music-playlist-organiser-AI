# YouTube API Services Implementation, Access, Integration, and Use

## Application Overview

Application name: VibeShelf

VibeShelf is a FastAPI web application that helps a signed-in user organize their own YouTube playlists into private mood-based playlists. The application reads the user's YouTube playlists and playlist items, classifies music items into mood categories using Azure OpenAI, shows the user a preview for review, and then creates or updates private managed playlists in the user's YouTube account after the user confirms the changes.

The application is intended for use by the account owner who signs in through Google OAuth. It is not designed to publish YouTube content publicly or expose YouTube data to other users.

## API Client Access

The API Client is a web application. It can be run locally or deployed as a Vercel application.

Local development URL:

```text
http://127.0.0.1:8000
```

Production URL:

```text
Set through APP_BASE_URL, for example https://your-project.vercel.app
```

The application requires the following server-side environment variables:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_DEPLOYMENT
GOOGLE_CLIENT_SECRETS_JSON
SESSION_SECRET
APP_BASE_URL
```

Secrets are loaded from environment variables only. The browser setup page displays whether each value is configured, but it does not ask users to type secrets into the web UI.

## YouTube API Services Used

The application uses the YouTube Data API v3 through the official Google API Python client.

Requested OAuth scope:

```text
https://www.googleapis.com/auth/youtube.force-ssl
```

This scope is used because the application must read the user's playlists and playlist items, and must create or update playlist resources in the user's own YouTube account. The application does not request broader YouTube scopes.

The application uses these YouTube API operations:

| Feature | YouTube API operation | Purpose |
| --- | --- | --- |
| Google OAuth connection | OAuth authorization code flow | Allows the user to grant access to their own YouTube account. |
| List playlists | `playlists.list` with `mine=true` | Displays the user's source playlists and allows the user to choose all playlists or one playlist. |
| List playlist items | `playlistItems.list` | Reads video IDs, titles, descriptions, channel names, and positions from selected source playlists. |
| Create managed playlists | `playlists.insert` | Creates private mood playlists when they do not already exist. |
| Update managed playlists | `playlists.update` | Keeps app-managed playlist titles, descriptions, and privacy status consistent. |
| Add videos to managed playlists | `playlistItems.insert` | Adds selected videos to the appropriate private mood playlists after user confirmation. |

The application does not upload videos, modify videos, delete videos, post comments, read comments, manage subscriptions, or access analytics.

## OAuth and Authentication Flow

1. The user opens the application home page.
2. The user clicks `Connect YouTube`.
3. The application starts the Google OAuth authorization code flow using the configured Google OAuth client JSON.
4. The user signs in on Google's OAuth screen and grants YouTube access.
5. Google redirects the user back to `/auth/google/callback`.
6. The application exchanges the authorization code for OAuth credentials.
7. The OAuth token payload is stored in an encrypted, HttpOnly browser cookie.

The application validates OAuth state during the callback to reduce cross-site request forgery risk. On HTTPS deployments, the token cookie is marked secure.

When the user clicks `Disconnect YouTube`, the application calls Google's token revocation endpoint, clears the local Google token cookie, clears OAuth session state, deletes locally stored run history and classification cache data derived from YouTube API data, and clears in-memory apply job results.

## User Workflow

1. Setup

   The home page shows whether the required Azure OpenAI and Google OAuth settings are configured.

2. Connect YouTube

   The user connects their YouTube account through Google OAuth. After connection, the application lists source playlists owned by the user.

3. Select organization scope

   The user can choose:

   - All playlists
   - One selected playlist

   App-managed playlists are excluded from the source list so the application does not repeatedly process playlists it created.

4. Generate preview

   The application reads playlist items from YouTube, deduplicates videos that appear in multiple playlists, and classifies music items into these mood categories:

   - Happy / Feel-good
   - Sad / Emotional
   - Romantic / Love
   - Chill / Relaxing
   - Energetic / Hype
   - Dark / Intense

5. Review assignments

   The preview page displays each item with:

   - Song title
   - Channel title
   - Source playlist name and position
   - Suggested mood or moods
   - Classification confidence
   - Classification reason
   - Editable final mood checkboxes

   The user can change mood assignments or uncheck all moods to skip a video.

6. Apply to YouTube

   After the user clicks `Apply to YouTube`, the application creates or updates private mood playlists and adds selected videos to those playlists.

## YouTube Data Displayed in the API Client

The application displays the following YouTube data to the signed-in user:

| Data | Source | Display location |
| --- | --- | --- |
| Playlist title | YouTube playlist snippet | Home page playlist selector and available source playlists section. |
| Playlist item count | YouTube playlist content details | Home page available source playlists section. |
| Playlist privacy status | YouTube playlist status | Home page available source playlists section. |
| Video title | YouTube playlist item snippet | Preview review table. |
| Video owner channel title | YouTube playlist item snippet | Preview review table. |
| Video description | YouTube playlist item snippet | Used for classification and included in preview state. |
| Source playlist and source position | YouTube playlist item snippet | Preview review table. |
| YouTube video ID | YouTube playlist item resource ID | Used internally for deduplication and playlist sync. |

The application displays YouTube data only to the signed-in user during their session.

## YouTube Data Stored or Processed

Depending on runtime mode, the application may use SQLite locally for run history and classification cache. On Vercel, the deployment filesystem is read-only and durable run history is not retained by the deployment.

The application processes:

- Playlist IDs
- Playlist titles
- Playlist descriptions
- Playlist privacy status
- Playlist item IDs
- Video IDs
- Video titles
- Video descriptions
- Channel titles
- Source playlist positions
- User-approved final mood assignments

OAuth tokens are encrypted before being stored in the browser cookie. The application does not store Google OAuth secrets in the browser. Google OAuth client configuration is read from the `GOOGLE_CLIENT_SECRETS_JSON` environment variable.

Users can revoke access inside the app with `Disconnect YouTube` or from Google's security settings. The in-app disconnect action programmatically revokes the token immediately and deletes locally stored YouTube-derived data. If revocation has already happened outside the app, the next failed token use requires the user to reconnect and local browser cookies can still be cleared with `Disconnect YouTube`.

## Playlist Creation and Management

The application creates private managed playlists for mood categories. New managed playlists are identified by this internal marker in the playlist description:

```text
[vibeshelf-managed]
```

Existing managed playlists with this legacy marker are also recognized:

```text
[yt-mood-organizer-managed]
```

Managed playlist descriptions include the source scope and mood. The application uses this marker to avoid treating its own generated playlists as source playlists. Existing playlists with the legacy marker continue to be recognized.

Created playlists are private by default.

The current sync behavior adds missing videos to managed playlists. It does not delete videos from playlists.

## Data Sharing and External Services

The application sends selected video metadata to Azure OpenAI for mood classification. The classification input can include title, channel title, description, source playlist names, and source positions.

YouTube data is not sold, shared with advertising networks, or made available to unrelated third parties by this application. Data is used only to provide the playlist organization functionality requested by the signed-in user.

## Security Controls

- Google OAuth is performed through Google's authorization flow.
- The application requests the `youtube.force-ssl` scope and does not request broader YouTube scopes.
- OAuth state is validated during callback.
- OAuth token payload is encrypted before storage in an HttpOnly cookie.
- The Disconnect YouTube action programmatically revokes the Google token and deletes local YouTube-derived run/cache data.
- Secrets are read from environment variables.
- The UI masks secret configuration values.
- Managed playlists are created as private playlists.
- The application excludes managed playlists from source playlist processing.
- YouTube API calls include retry handling for transient API failures.

## Screencast Requirements for Non-Public API Client

If the API Client is not publicly accessible, provide a screencast showing each YouTube API Service functionality and the YouTube data displayed or integrated into the application.

Suggested screencast outline:

1. Show the home page setup status.
2. Click `Connect YouTube`.
3. Show the Google OAuth consent screen and requested YouTube access.
4. Return to the application after successful connection.
5. Show the loaded source playlist list, including playlist title, item count, and privacy status.
6. Select `All playlists` or `One playlist`.
7. Generate a preview.
8. Show the preview table with video titles, channel titles, source playlists, source positions, suggested moods, confidence, and reasons.
9. Change at least one final mood assignment in the review table.
10. Click `Apply to YouTube`.
11. Open YouTube or YouTube Music in the same signed-in account.
12. Show the private managed mood playlists created by the application.
13. Open a managed playlist and show that selected videos were added.

The screencast should avoid exposing secrets, OAuth client secrets, API keys, refresh tokens, or private credentials.

## Implementation Files

Primary implementation files:

- `app/main.py`: Web routes, OAuth connection, preview, and apply flow.
- `app/services/youtube.py`: YouTube OAuth and YouTube Data API integration.
- `app/services/organizer.py`: Preview creation, deduplication, user-reviewed assignment application, and playlist sync orchestration.
- `app/services/azure_openai.py`: Mood classification integration.
- `app/security.py`: Encryption and decryption helpers for browser-submitted state and token cookie payloads.
- `templates/index.html`: Setup, YouTube connection, playlist selection, and source playlist display.
- `templates/run_detail.html`: Preview review UI and final mood assignment controls.

## Summary

VibeShelf uses YouTube API Services only to let a signed-in user read their own playlists, review mood assignments, and create or update private mood playlists in their own account. The application displays YouTube playlist and playlist item metadata to the signed-in user, requires user confirmation before writing playlists, and stores credentials and tokens using server-side environment variables and encrypted browser cookies.
