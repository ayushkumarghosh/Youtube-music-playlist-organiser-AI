"""FastAPI application entrypoint."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.config import RuntimePaths, ensure_data_dir
from app.constants import APP_NAME, MOOD_LABELS, SESSION_SECRET_DEFAULT
from app.db import Database
from app.models import RunDetail, RunScope
from app.security import EncryptedStateError, decrypt_json, encrypt_json
from app.services.azure_openai import AzureClassificationError, AzureOpenAIClassifier
from app.services.organizer import OrganizerService
from app.services.settings import SettingsService
from app.services.youtube import YouTubeAuthError, YouTubeService, YouTubeSyncError


paths = RuntimePaths()
ensure_data_dir()
db = Database(paths.db_path)
db.initialize()
settings_service = SettingsService(db)

app = FastAPI(title=APP_NAME)
templates = Jinja2Templates(directory=str(paths.templates_dir))
app.mount("/static", StaticFiles(directory=str(paths.static_dir)), name="static")
GOOGLE_TOKEN_COOKIE = "ytmp_google_token"
RUN_STATE_FIELD = "run_state"


def session_secret() -> str:
    settings = settings_service.get_settings()
    return settings.session_secret or SESSION_SECRET_DEFAULT


app.add_middleware(SessionMiddleware, secret_key=session_secret())


def set_flash(request: Request, message: str, level: str = "info") -> None:
    request.session["flash"] = {"message": message, "level": level}


def pop_flash(request: Request) -> dict[str, str] | None:
    return request.session.pop("flash", None)


def secure_cookie(request: Request) -> bool:
    return request.url.scheme == "https"


def google_token_payload(request: Request) -> dict[str, Any] | None:
    token = request.cookies.get(GOOGLE_TOKEN_COOKIE)
    if not token:
        return None
    try:
        return decrypt_json(token, session_secret())
    except EncryptedStateError:
        return None


def set_google_token_cookie(response: RedirectResponse, request: Request, payload: dict[str, Any]) -> None:
    response.set_cookie(
        GOOGLE_TOKEN_COOKIE,
        encrypt_json(payload, session_secret()),
        httponly=True,
        samesite="lax",
        secure=secure_cookie(request),
        max_age=60 * 60 * 24 * 30,
    )


def encrypted_run_state(run: RunDetail) -> str:
    return encrypt_json(run.model_dump(mode="json"), session_secret())


def render_run_detail(request: Request, run: RunDetail, status_code: int = 200):
    settings = settings_service.get_settings()
    return templates.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={
            "request": request,
            "app_name": APP_NAME,
            "flash": pop_flash(request),
            "settings_complete": settings.is_complete(),
            "youtube_connected": google_token_payload(request) is not None,
            "run": run,
            "run_state": encrypted_run_state(run),
            "mood_labels": MOOD_LABELS,
        },
        status_code=status_code,
    )


def get_base_context(request: Request) -> dict[str, Any]:
    settings = settings_service.get_settings()
    masked_settings = settings.masked()
    setup_errors = settings_service.validate(settings)
    youtube_connected = google_token_payload(request) is not None
    playlists = []
    errors: list[str] = list(setup_errors)
    if settings.is_complete() and youtube_connected:
        try:
            youtube_service = YouTubeService(settings, db, google_token_payload(request))
            playlists = youtube_service.list_playlists(include_managed=False)
        except Exception as exc:
            youtube_connected = False
            errors.append(str(exc))
    return {
        "request": request,
        "app_name": APP_NAME,
        "flash": pop_flash(request),
        "errors": errors,
        "settings": masked_settings,
        "settings_complete": settings.is_complete(),
        "youtube_connected": youtube_connected,
        "playlists": playlists,
        "run_scopes": list(RunScope),
        "mood_labels": MOOD_LABELS,
    }


def redirect_uri_for(request: Request) -> str:
    settings = settings_service.get_settings()
    if settings.app_base_url.strip():
        return settings.app_base_url.rstrip("/") + request.url_for("google_callback").path
    return str(request.url_for("google_callback"))


@app.get("/")
def home(request: Request):
    context = get_base_context(request)
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.post("/auth/google/connect")
def google_connect(request: Request):
    settings = settings_service.get_settings()
    if not settings.is_complete():
        set_flash(request, "Set all required environment variables before connecting YouTube.", "error")
        return RedirectResponse(url="/", status_code=303)
    youtube_service = YouTubeService(settings, db)
    auth_url, state, code_verifier = youtube_service.build_authorization_url(redirect_uri_for(request))
    request.session["google_oauth_state"] = state
    request.session["google_code_verifier"] = code_verifier
    return RedirectResponse(url=auth_url, status_code=303)


@app.get("/auth/google/callback", name="google_callback")
def google_callback(request: Request, code: str | None = None, state: str | None = None):
    expected_state = request.session.get("google_oauth_state")
    code_verifier = request.session.get("google_code_verifier")
    if not code or not state or not expected_state or state != expected_state:
        raise HTTPException(status_code=400, detail="Invalid Google OAuth response.")
    settings = settings_service.get_settings()
    youtube_service = YouTubeService(settings, db)
    try:
        token_payload = youtube_service.exchange_code(code, state, redirect_uri_for(request), code_verifier)
    except Exception as exc:
        request.session.pop("google_oauth_state", None)
        request.session.pop("google_code_verifier", None)
        set_flash(request, f"Google OAuth failed: {exc}", "error")
        return RedirectResponse(url="/", status_code=303)
    request.session.pop("google_oauth_state", None)
    request.session.pop("google_code_verifier", None)
    set_flash(request, "YouTube connected successfully.", "success")
    response = RedirectResponse(url="/", status_code=303)
    set_google_token_cookie(response, request, token_payload)
    return response


@app.post("/runs/preview")
def preview_run(
    request: Request,
    scope: str = Form(...),
    selected_playlist_id: str = Form(""),
):
    settings = settings_service.get_settings()
    if not settings.is_complete():
        set_flash(request, "Set all required environment variables before generating a preview.", "error")
        return RedirectResponse(url="/", status_code=303)
    token_payload = google_token_payload(request)
    youtube_service = YouTubeService(settings, db, token_payload)
    if not youtube_service.has_token():
        set_flash(request, "Connect YouTube before generating a preview.", "error")
        return RedirectResponse(url="/", status_code=303)

    try:
        run_scope = RunScope(scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid scope.") from exc
    if run_scope == RunScope.SINGLE_PLAYLIST and not selected_playlist_id:
        set_flash(request, "Choose a playlist for single-playlist preview mode.", "error")
        return RedirectResponse(url="/", status_code=303)
    classifier = AzureOpenAIClassifier(settings, db)
    organizer = OrganizerService(db, youtube_service, classifier)
    try:
        run = organizer.create_preview(run_scope, selected_playlist_id or None, persist=False)
    except AzureClassificationError as exc:
        set_flash(request, f"Preview failed during song classification: {exc}", "error")
        return RedirectResponse(url="/", status_code=303)
    set_flash(request, "Preview generated.", "success")
    return render_run_detail(request, run)


@app.get("/runs/{run_id}")
def run_detail(request: Request, run_id: str):
    settings = settings_service.get_settings()
    token_payload = google_token_payload(request)
    youtube_connected = token_payload is not None
    organizer = OrganizerService(
        db,
        YouTubeService(settings, db, token_payload),
        AzureOpenAIClassifier(settings, db),
    )
    run = organizer.load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return templates.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={
            "request": request,
            "app_name": APP_NAME,
            "flash": pop_flash(request),
            "settings_complete": settings.is_complete(),
            "youtube_connected": youtube_connected,
            "run": run,
            "run_state": encrypted_run_state(run),
            "mood_labels": MOOD_LABELS,
        },
    )


@app.post("/runs/apply")
async def apply_run(request: Request):
    form = await request.form()
    run_id = str(form.get("run_id", "")).strip()
    run_state = str(form.get(RUN_STATE_FIELD, "")).strip()
    if not run_id and not run_state:
        raise HTTPException(status_code=400, detail="run_id or run_state is required.")
    overrides: dict[str, list[str]] = {}
    for key, value in form.multi_items():
        if not key.startswith("mood__"):
            continue
        video_id = key.replace("mood__", "", 1)
        overrides.setdefault(video_id, []).append(str(value).strip())
    settings = settings_service.get_settings()
    token_payload = google_token_payload(request)
    youtube_service = YouTubeService(settings, db, token_payload)
    classifier = AzureOpenAIClassifier(settings, db)
    organizer = OrganizerService(db, youtube_service, classifier)
    try:
        if run_state:
            payload = decrypt_json(run_state, session_secret())
            run = RunDetail.model_validate(payload)
            organizer.apply_run_detail(run, overrides)
            run_id = run.run_id
        else:
            organizer.apply_run(run_id, overrides)
    except EncryptedStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except YouTubeAuthError as exc:
        set_flash(request, str(exc), "error")
        return RedirectResponse(url="/", status_code=303)
    except YouTubeSyncError as exc:
        set_flash(request, f"Applying playlists failed: {exc}", "error")
        return RedirectResponse(url="/", status_code=303)
    set_flash(request, "Mood playlists synced to YouTube.", "success")
    return RedirectResponse(url="/", status_code=303)
