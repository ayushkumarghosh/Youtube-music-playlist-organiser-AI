"""FastAPI application entrypoint."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any
import uuid

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.config import RuntimePaths, ensure_data_dir
from app.constants import APP_NAME, CUSTOM_CATEGORY_ID_PREFIX, MOOD_LABELS, SESSION_SECRET_DEFAULT
from app.db import Database
from app.models import (
    CategoryLabelDefinition,
    CategorySetDefinition,
    RunDetail,
    RunScope,
    built_in_category_sets,
    default_mood_category_set,
    slugify_label,
)
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
APPLY_JOBS_MAX = 50
CONTACT_EMAIL = "ayush@scorptech.co"
GOOGLE_PRIVACY_POLICY_URL = "http://www.google.com/policies/privacy"
GOOGLE_SECURITY_SETTINGS_URL = "https://security.google.com/settings/security/permissions"
YOUTUBE_TERMS_URL = "https://www.youtube.com/t/terms"


@dataclass
class ApplyJob:
    job_id: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Queued"
    current: int = 0
    total: int = 1
    percent: int = 0
    error: str = ""
    result: dict[str, object] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "current": self.current,
            "total": self.total,
            "percent": self.percent,
            "error": self.error,
            "result": self.result,
            "finish_url": "/finish" if self.status == "complete" else "",
        }


apply_jobs: dict[str, ApplyJob] = {}
apply_jobs_lock = Lock()
apply_executor = ThreadPoolExecutor(max_workers=2)


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


def clear_google_token_cookie(response: RedirectResponse) -> None:
    response.delete_cookie(GOOGLE_TOKEN_COOKIE)


def legal_context(request: Request) -> dict[str, Any]:
    return {
        "request": request,
        "app_name": APP_NAME,
        "flash": pop_flash(request),
        "settings_complete": settings_service.get_settings().is_complete(),
        "youtube_connected": google_token_payload(request) is not None,
        "contact_email": CONTACT_EMAIL,
        "google_privacy_policy_url": GOOGLE_PRIVACY_POLICY_URL,
        "google_security_settings_url": GOOGLE_SECURITY_SETTINGS_URL,
        "youtube_terms_url": YOUTUBE_TERMS_URL,
    }


def remember_apply_job(job: ApplyJob) -> None:
    with apply_jobs_lock:
        apply_jobs[job.job_id] = job
        while len(apply_jobs) > APPLY_JOBS_MAX:
            oldest_id = min(apply_jobs, key=lambda key: apply_jobs[key].created_at)
            apply_jobs.pop(oldest_id, None)


def get_apply_job(job_id: str) -> ApplyJob | None:
    with apply_jobs_lock:
        return apply_jobs.get(job_id)


def update_apply_job(job_id: str, **updates: object) -> None:
    with apply_jobs_lock:
        job = apply_jobs.get(job_id)
        if job is None:
            return
        for key, value in updates.items():
            setattr(job, key, value)


def clear_apply_jobs() -> None:
    with apply_jobs_lock:
        apply_jobs.clear()


def parse_assignment_overrides(form) -> dict[str, dict[str, list[str]]]:
    overrides: dict[str, dict[str, list[str]]] = {}
    for key, value in form.multi_items():
        value = str(value).strip()
        if key.startswith("assignment__"):
            parts = key.split("__", 2)
            if len(parts) != 3 or not value:
                continue
            _, category_id, video_id = parts
            overrides.setdefault(slugify_label(category_id), {}).setdefault(video_id, []).append(slugify_label(value))
        elif key.startswith("mood__"):
            video_id = key.replace("mood__", "", 1)
            if value:
                overrides.setdefault("mood", {}).setdefault(video_id, []).append(slugify_label(value))
    return overrides


def parse_mood_overrides(form) -> dict[str, dict[str, list[str]]]:
    return parse_assignment_overrides(form)


def available_category_sets() -> list[CategorySetDefinition]:
    return [*built_in_category_sets(), *db.list_custom_category_sets(include_archived=False)]


def resolve_category_sets(category_ids: list[str]) -> list[CategorySetDefinition]:
    normalized_ids = [slugify_label(category_id) for category_id in category_ids if category_id.strip()]
    if not normalized_ids:
        normalized_ids = [default_mood_category_set().id]
    available = {category.id: category for category in available_category_sets()}
    category_sets = []
    for category_id in dict.fromkeys(normalized_ids):
        category = available.get(category_id)
        if category is None:
            raise HTTPException(status_code=400, detail=f"Unknown category set: {category_id}")
        category_sets.append(category)
    if not category_sets:
        raise HTTPException(status_code=400, detail="Choose at least one category set.")
    return category_sets


def custom_category_id(name: str) -> str:
    return f"{CUSTOM_CATEGORY_ID_PREFIX}{slugify_label(name)}-{uuid.uuid4().hex[:8]}"


def parse_target_count(raw: str | int) -> int:
    try:
        target_count = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Target playlist count must be a number from 2 to 12.") from exc
    if target_count < 2 or target_count > 12:
        raise ValueError("Target playlist count must be between 2 and 12.")
    return target_count


def build_custom_category(
    *,
    category_id: str,
    name: str,
    prompt: str,
    label_names: list[str],
    label_descriptions: list[str],
) -> CategorySetDefinition:
    labels: list[CategoryLabelDefinition] = []
    seen_slugs: set[str] = set()
    for index, label_name in enumerate(label_names):
        label_name = label_name.strip()
        if not label_name:
            continue
        slug = slugify_label(label_name)
        if slug in seen_slugs:
            raise ValueError("Custom playlist labels must be unique.")
        seen_slugs.add(slug)
        description = label_descriptions[index].strip() if index < len(label_descriptions) else ""
        labels.append(CategoryLabelDefinition(slug=slug, name=label_name, description=description))
    if len(labels) < 2:
        raise ValueError("Save at least two custom playlist labels.")
    return CategorySetDefinition(
        id=category_id or custom_category_id(name),
        name=name,
        description="Custom playlist category.",
        labels=labels,
        source="custom",
        prompt=prompt,
    )


def run_apply_job(
    job_id: str,
    *,
    settings,
    token_payload: dict[str, Any] | None,
    run_id: str,
    run_state: str,
    overrides: dict[str, dict[str, list[str]]],
) -> None:
    def report(progress: dict[str, object]) -> None:
        update_apply_job(
            job_id,
            status="running",
            stage=str(progress.get("stage", "running")),
            message=str(progress.get("message", "Syncing to YouTube")),
            current=int(progress.get("current", 0)),
            total=max(1, int(progress.get("total", 1))),
            percent=max(0, min(100, int(progress.get("percent", 0)))),
        )

    update_apply_job(job_id, status="running", stage="starting", message="Starting YouTube sync", percent=2)
    youtube_service = YouTubeService(settings, db, token_payload)
    classifier = AzureOpenAIClassifier(settings, db)
    organizer = OrganizerService(db, youtube_service, classifier)
    try:
        if run_state:
            payload = decrypt_json(run_state, session_secret())
            run = RunDetail.model_validate(payload)
            result = organizer.apply_run_detail(run, overrides, progress_callback=report)
        else:
            result = organizer.apply_run(run_id, overrides, progress_callback=report)
        update_apply_job(
            job_id,
            status="complete",
            stage="complete",
            message="Category playlists synced",
            current=1,
            total=1,
            percent=100,
            result=result,
        )
    except Exception as exc:
        update_apply_job(
            job_id,
            status="failed",
            stage="failed",
            message="Applying playlists failed",
            percent=100,
            error=str(exc),
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
            "category_sets": run.category_sets,
        },
        status_code=status_code,
    )


def get_login_context(request: Request) -> dict[str, Any]:
    settings = settings_service.get_settings()
    setup_errors = settings_service.validate(settings)
    return {
        "request": request,
        "app_name": APP_NAME,
        "flash": pop_flash(request),
        "errors": list(setup_errors),
        "settings_complete": settings.is_complete(),
        "youtube_connected": google_token_payload(request) is not None,
    }


def get_preview_context(request: Request, custom_proposal: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings_service.get_settings()
    token_payload = google_token_payload(request)
    playlists = []
    errors: list[str] = []
    if settings.is_complete() and token_payload is not None:
        try:
            youtube_service = YouTubeService(settings, db, token_payload)
            playlists = youtube_service.list_playlists(include_managed=False)
        except Exception as exc:
            errors.append(str(exc))
    return {
        "request": request,
        "app_name": APP_NAME,
        "flash": pop_flash(request),
        "errors": errors,
        "settings_complete": settings.is_complete(),
        "youtube_connected": token_payload is not None,
        "playlists": playlists,
        "run_scopes": list(RunScope),
        "mood_labels": MOOD_LABELS,
        "built_in_category_sets": built_in_category_sets(),
        "custom_category_sets": db.list_custom_category_sets(include_archived=False),
        "custom_proposal": custom_proposal,
    }


def redirect_uri_for(request: Request) -> str:
    settings = settings_service.get_settings()
    if settings.app_base_url.strip():
        return settings.app_base_url.rstrip("/") + request.url_for("google_callback").path
    return str(request.url_for("google_callback"))


@app.get("/")
def home(request: Request):
    if settings_service.get_settings().is_complete() and google_token_payload(request) is not None:
        return RedirectResponse(url="/preview", status_code=303)
    context = get_login_context(request)
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/terms")
def terms(request: Request):
    return templates.TemplateResponse(request=request, name="terms.html", context=legal_context(request))


@app.get("/privacy")
def privacy(request: Request):
    return templates.TemplateResponse(request=request, name="privacy.html", context=legal_context(request))


@app.get("/preview")
def preview_workspace(request: Request):
    settings = settings_service.get_settings()
    if not settings.is_complete():
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=get_login_context(request),
        )
    if google_token_payload(request) is None:
        set_flash(request, "Connect YouTube before generating a preview.", "error")
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="preview.html",
        context=get_preview_context(request),
    )


@app.get("/finish")
def finish(request: Request):
    settings = settings_service.get_settings()
    if not settings.is_complete() or google_token_payload(request) is None:
        set_flash(request, "Connect YouTube before starting another preview.", "error")
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="finish.html",
        context={
            "request": request,
            "app_name": APP_NAME,
            "flash": pop_flash(request),
            "settings_complete": settings.is_complete(),
            "youtube_connected": True,
        },
    )


@app.post("/auth/google/connect")
def google_connect(request: Request, policy_agreement: str | None = Form(None)):
    settings = settings_service.get_settings()
    if not settings.is_complete():
        set_flash(request, "Set all required environment variables before connecting YouTube.", "error")
        return RedirectResponse(url="/", status_code=303)
    if policy_agreement != "accepted":
        set_flash(request, "Accept the Terms and Privacy Policy before connecting YouTube.", "error")
        return RedirectResponse(url="/", status_code=303)
    youtube_service = YouTubeService(settings, db)
    auth_url, state, code_verifier = youtube_service.build_authorization_url(redirect_uri_for(request))
    request.session["google_oauth_state"] = state
    request.session["google_code_verifier"] = code_verifier
    return RedirectResponse(url=auth_url, status_code=303)


@app.post("/auth/google/disconnect")
def google_disconnect(request: Request):
    token_payload = google_token_payload(request)
    request.session.pop("google_oauth_state", None)
    request.session.pop("google_code_verifier", None)
    revoke_error = ""
    if token_payload is not None:
        try:
            YouTubeService(settings_service.get_settings(), db, token_payload).revoke_token()
        except YouTubeAuthError as exc:
            revoke_error = str(exc)
    db.delete_authorized_youtube_data()
    clear_apply_jobs()
    if revoke_error:
        set_flash(
            request,
            f"YouTube disconnected locally and stored YouTube data was deleted. Google token revocation failed: {revoke_error}",
            "error",
        )
    else:
        set_flash(request, "YouTube access revoked, local token cleared, and stored YouTube data deleted.", "success")
    response = RedirectResponse(url="/", status_code=303)
    clear_google_token_cookie(response)
    return response


@app.post("/categories/custom/propose")
def propose_custom_category(
    request: Request,
    category_name: str = Form(""),
    category_prompt: str = Form(""),
    target_count: str = Form("6"),
):
    settings = settings_service.get_settings()
    if not settings.is_complete():
        set_flash(request, "Set all required environment variables before creating custom categories.", "error")
        return RedirectResponse(url="/", status_code=303)
    name = category_name.strip()
    prompt = category_prompt.strip()
    if not name or not prompt:
        set_flash(request, "Custom categories need a name and prompt.", "error")
        return RedirectResponse(url="/preview", status_code=303)
    try:
        parsed_target_count = parse_target_count(target_count)
        proposal = AzureOpenAIClassifier(settings, db).propose_custom_category_labels(
            name,
            prompt,
            parsed_target_count,
        )
    except (ValueError, AzureClassificationError) as exc:
        set_flash(request, str(exc), "error")
        return RedirectResponse(url="/preview", status_code=303)

    return templates.TemplateResponse(
        request=request,
        name="preview.html",
        context=get_preview_context(
            request,
            custom_proposal={
                "name": name,
                "prompt": prompt,
                "target_count": parsed_target_count,
                "labels": proposal.labels,
            },
        ),
    )


@app.post("/categories/custom/save")
async def save_custom_category(request: Request):
    form = await request.form()
    name = str(form.get("category_name", "")).strip()
    prompt = str(form.get("category_prompt", "")).strip()
    category_id = str(form.get("category_id", "")).strip()
    label_names = [str(value).strip() for value in form.getlist("label_name")]
    label_descriptions = [str(value).strip() for value in form.getlist("label_description")]
    if not name or not prompt:
        set_flash(request, "Custom categories need a name and prompt.", "error")
        return RedirectResponse(url="/preview", status_code=303)
    try:
        category = build_custom_category(
            category_id=category_id,
            name=name,
            prompt=prompt,
            label_names=label_names,
            label_descriptions=label_descriptions,
        )
        db.save_custom_category_set(category)
    except ValueError as exc:
        set_flash(request, str(exc), "error")
        return RedirectResponse(url="/preview", status_code=303)
    set_flash(request, f"Custom category '{category.name}' saved.", "success")
    return RedirectResponse(url="/preview", status_code=303)


@app.post("/categories/custom/archive")
def archive_custom_category(category_id: str = Form("")):
    if category_id:
        db.archive_custom_category_set(slugify_label(category_id))
    return RedirectResponse(url="/preview", status_code=303)


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
    response = RedirectResponse(url="/preview", status_code=303)
    set_google_token_cookie(response, request, token_payload)
    return response


@app.post("/runs/preview")
def preview_run(
    request: Request,
    scope: str = Form("selected_playlists"),
    selected_playlist_id: str = Form(""),
    selected_playlist_ids: list[str] = Form(default=[]),
    category_ids: list[str] = Form(default=[]),
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
    if selected_playlist_id and selected_playlist_id not in selected_playlist_ids:
        selected_playlist_ids = [selected_playlist_id, *selected_playlist_ids]
    if run_scope == RunScope.SELECTED_PLAYLISTS and not selected_playlist_ids:
        set_flash(request, "Choose at least one playlist before generating a preview.", "error")
        return RedirectResponse(url="/preview", status_code=303)
    if run_scope == RunScope.SINGLE_PLAYLIST and not selected_playlist_id:
        set_flash(request, "Choose a playlist before generating a preview.", "error")
        return RedirectResponse(url="/preview", status_code=303)
    category_sets = resolve_category_sets(category_ids)
    classifier = AzureOpenAIClassifier(settings, db)
    organizer = OrganizerService(db, youtube_service, classifier)
    try:
        try:
            run = organizer.create_preview(
                run_scope,
                selected_playlist_id or None,
                source_playlist_ids=selected_playlist_ids,
                persist=False,
                category_sets=category_sets,
            )
        except TypeError:
            run = organizer.create_preview(
                run_scope,
                selected_playlist_id or None,
                source_playlist_ids=selected_playlist_ids,
                persist=False,
            )
    except AzureClassificationError as exc:
        set_flash(request, f"Preview failed during song classification: {exc}", "error")
        return RedirectResponse(url="/preview", status_code=303)
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
            "category_sets": run.category_sets,
        },
    )


@app.post("/runs/apply")
async def apply_run(request: Request):
    form = await request.form()
    run_id = str(form.get("run_id", "")).strip()
    run_state = str(form.get(RUN_STATE_FIELD, "")).strip()
    if not run_id and not run_state:
        raise HTTPException(status_code=400, detail="run_id or run_state is required.")
    overrides = parse_assignment_overrides(form)
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
    return RedirectResponse(url="/finish", status_code=303)


@app.post("/runs/apply/start")
async def start_apply_run(request: Request):
    form = await request.form()
    run_id = str(form.get("run_id", "")).strip()
    run_state = str(form.get(RUN_STATE_FIELD, "")).strip()
    if not run_id and not run_state:
        return JSONResponse({"error": "run_id or run_state is required."}, status_code=400)
    settings = settings_service.get_settings()
    token_payload = google_token_payload(request)
    if not settings.is_complete():
        return JSONResponse({"error": "Set all required environment variables before applying."}, status_code=400)
    if token_payload is None:
        return JSONResponse({"error": "Connect YouTube before applying playlists."}, status_code=401)

    job = ApplyJob(job_id=str(uuid.uuid4()))
    remember_apply_job(job)
    apply_executor.submit(
        run_apply_job,
        job.job_id,
        settings=settings,
        token_payload=token_payload,
        run_id=run_id,
        run_state=run_state,
        overrides=parse_assignment_overrides(form),
    )
    return JSONResponse({"job_id": job.job_id, "status_url": f"/runs/apply/status/{job.job_id}"})


@app.get("/runs/apply/status/{job_id}")
def apply_run_status(job_id: str):
    job = get_apply_job(job_id)
    if job is None:
        return JSONResponse({"error": "Apply job not found."}, status_code=404)
    return JSONResponse(job.as_dict())
