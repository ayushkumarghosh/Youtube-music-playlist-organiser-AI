"""Settings management helpers."""

from __future__ import annotations

from pathlib import Path

from app.config import settings_from_env
from app.db import Database
from app.models import SetupSettings


class SettingsService:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get_settings(self) -> SetupSettings:
        stored = self.db.load_settings()
        env = settings_from_env()
        return SetupSettings(
            azure_openai_endpoint=stored.azure_openai_endpoint or env.azure_openai_endpoint,
            azure_openai_api_key=stored.azure_openai_api_key or env.azure_openai_api_key,
            azure_openai_deployment=stored.azure_openai_deployment or env.azure_openai_deployment,
            google_client_secrets_file=stored.google_client_secrets_file or env.google_client_secrets_file,
            session_secret=stored.session_secret or env.session_secret,
        )

    def validate_before_save(self, settings: SetupSettings) -> list[str]:
        errors: list[str] = []
        if not settings.azure_openai_endpoint.strip():
            errors.append("Azure OpenAI endpoint is required.")
        if not settings.azure_openai_api_key.strip():
            errors.append("Azure OpenAI API key is required.")
        if not settings.azure_openai_deployment.strip():
            errors.append("Azure OpenAI deployment is required.")
        secrets_path = settings.google_client_secrets_file.strip()
        if not secrets_path:
            errors.append("Google client secrets file path is required.")
        elif not Path(secrets_path).expanduser().exists():
            errors.append("Google client secrets file was not found.")
        return errors

    def save_settings(self, settings: SetupSettings) -> None:
        self.db.save_settings(settings)
