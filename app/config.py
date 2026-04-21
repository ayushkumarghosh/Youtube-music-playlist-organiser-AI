"""Configuration loading and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from app.constants import APP_DATA_DIR, DB_FILENAME, SESSION_SECRET_DEFAULT, TOKENS_FILENAME
from app.models import SetupSettings


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / APP_DATA_DIR
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"
DB_PATH = DATA_DIR / DB_FILENAME
TOKENS_PATH = DATA_DIR / TOKENS_FILENAME


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class RuntimePaths:
    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    templates_dir: Path = TEMPLATES_DIR
    static_dir: Path = STATIC_DIR
    db_path: Path = DB_PATH
    tokens_path: Path = TOKENS_PATH


def settings_from_env() -> SetupSettings:
    return SetupSettings(
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        google_client_secrets_file=os.getenv("GOOGLE_CLIENT_SECRETS_FILE", ""),
        session_secret=os.getenv("SESSION_SECRET", SESSION_SECRET_DEFAULT),
    )
