"""Settings management helpers."""

from __future__ import annotations

import json

from app.config import settings_from_env
from app.db import Database
from app.models import SetupSettings


class SettingsService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_settings(self) -> SetupSettings:
        return settings_from_env()

    def validate(self, settings: SetupSettings | None = None) -> list[str]:
        settings = settings or self.get_settings()
        errors: list[str] = []
        if not settings.azure_openai_endpoint.strip():
            errors.append("AZURE_OPENAI_ENDPOINT is required.")
        if not settings.azure_openai_api_key.strip():
            errors.append("AZURE_OPENAI_API_KEY is required.")
        if not settings.azure_openai_deployment.strip():
            errors.append("AZURE_OPENAI_DEPLOYMENT is required.")
        if not settings.google_client_secrets_json.strip():
            errors.append("GOOGLE_CLIENT_SECRETS_JSON is required.")
        else:
            try:
                json.loads(settings.google_client_secrets_json)
            except json.JSONDecodeError:
                errors.append("GOOGLE_CLIENT_SECRETS_JSON must be valid JSON.")
        if not settings.session_secret.strip():
            errors.append("SESSION_SECRET is required.")
        return errors
