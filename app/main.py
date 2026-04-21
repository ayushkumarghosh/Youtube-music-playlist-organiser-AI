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
from app.models import RunScope, SetupSettings
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


def session_secret() -> str:
    settings = settings_service.get_settings()
    return settings.session_secret or SESSION_SECRET_DEFAULT


app.add_middleware(SessionMiddleware, secret_key=session_secret())


def set_flash(request: Request, message: str, level: str = "info") -> None:
    request.session["flash"] = {"message": message, "level": level}


def pop_flash(request: Request) -> dict[str, str] | None:
    return request.session.pop("flash", None)


def get_base_context(request: Request) -> dict[str, Any]:
    settings = settings_service.get_settings()
    masked_settings = settings.masked()
    youtube_connected = False
    playlists = []
    errors: list[str] = []
    if settings.is_complete():
        try:
            youtube_service = YouTubeService(settings, db)
            youtube_connected = youtube_service.has_token()
            if youtube_connected:
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
    return str(request.url_for("google_callback"))


@app.get("/")
def home(request: Request):
    context = get_base_context(request)
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.post("/settings/save")
def save_settings(
    request: Request,
    azure_openai_endpoint: str = Form(...),
    azure_openai_api_key: str = Form(...),
    azure_openai_deployment: str = Form(...),
    google_client_secrets_file: str = Form(...),
    session_secret_value: str = Form(""),
):
    existing = settings_service.get_settings()
    settings = SetupSettings(
        azure_openai_endpoint=azure_openai_endpoint.strip(),
        azure_openai_api_key=azure_openai_api_key.strip() or existing.azure_openai_api_key,
        azure_openai_deployment=azure_openai_deployment.strip(),
        google_client_secrets_file=google_client_secrets_file.strip(),
        session_secret=session_secret_value.strip() or existing.session_secret or SESSION_SECRET_DEFAULT,
    )
    errors = settings_service.validate_before_save(settings)
    if errors:
        context = get_base_context(request)
        context["errors"] = errors
        context["settings"] = settings.masked()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context,
            status_code=400,
        )

    classifier = AzureOpenAIClassifier(settings, db)
    try:
        classifier.probe()
    except AzureClassificationError as exc:
        context = get_base_context(request)
        context["errors"] = [str(exc)]
        context["settings"] = settings.masked()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context,
            status_code=400,
        )

    settings_service.save_settings(settings)
    set_flash(request, "Setup saved and Azure connectivity verified. You can now connect YouTube.", "success")
    return RedirectResponse(url="/", status_code=303)


@app.post("/auth/google/connect")
def google_connect(request: Request):
    settings = settings_service.get_settings()
    if not settings.is_complete():
        set_flash(request, "Complete setup before connecting YouTube.", "error")
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
        youtube_service.exchange_code(code, state, redirect_uri_for(request), code_verifier)
    except Exception as exc:
        request.session.pop("google_oauth_state", None)
        request.session.pop("google_code_verifier", None)
        set_flash(request, f"Google OAuth failed: {exc}", "error")
        return RedirectResponse(url="/", status_code=303)
    request.session.pop("google_oauth_state", None)
    request.session.pop("google_code_verifier", None)
    set_flash(request, "YouTube connected successfully.", "success")
    return RedirectResponse(url="/", status_code=303)


@app.post("/runs/preview")
def preview_run(
    request: Request,
    scope: str = Form(...),
    selected_playlist_id: str = Form(""),
):
    settings = settings_service.get_settings()
    if not settings.is_complete():
        set_flash(request, "Finish setup before generating a preview.", "error")
        return RedirectResponse(url="/", status_code=303)
    youtube_service = YouTubeService(settings, db)
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
        run = organizer.create_preview(run_scope, selected_playlist_id or None)
    except AzureClassificationError as exc:
        set_flash(request, f"Preview failed during song classification: {exc}", "error")
        return RedirectResponse(url="/", status_code=303)
    set_flash(request, "Preview generated.", "success")
    return RedirectResponse(url=f"/runs/{run.run_id}", status_code=303)


@app.get("/runs/{run_id}")
def run_detail(request: Request, run_id: str):
    settings = settings_service.get_settings()
    youtube_connected = False
    if settings.is_complete():
        youtube_connected = YouTubeService(settings, db).has_token()
    organizer = OrganizerService(
        db,
        YouTubeService(settings, db),
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
            "mood_labels": MOOD_LABELS,
        },
    )


@app.post("/runs/apply")
async def apply_run(request: Request):
    form = await request.form()
    run_id = str(form.get("run_id", "")).strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required.")
    overrides = {
        key.replace("mood__", "", 1): str(value)
        for key, value in form.items()
        if key.startswith("mood__")
    }
    settings = settings_service.get_settings()
    youtube_service = YouTubeService(settings, db)
    classifier = AzureOpenAIClassifier(settings, db)
    organizer = OrganizerService(db, youtube_service, classifier)
    try:
        organizer.apply_run(run_id, overrides)
    except YouTubeAuthError as exc:
        set_flash(request, str(exc), "error")
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)
    except YouTubeSyncError as exc:
        set_flash(request, f"Applying playlists failed: {exc}", "error")
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)
    set_flash(request, "Mood playlists synced to YouTube.", "success")
    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)
