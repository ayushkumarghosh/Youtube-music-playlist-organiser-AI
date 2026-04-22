"""SQLite persistence helpers."""

from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from app.models import (
    RunDetail,
    RunItemView,
    RunScope,
    RunStatus,
    RunSummary,
    SetupSettings,
    deserialize_mood_labels,
    serialize_mood_labels,
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tokens (
    provider TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classification_cache (
    cache_key TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    metadata_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    scope TEXT NOT NULL,
    source_playlist_id TEXT,
    source_playlist_title TEXT,
    created_at TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    apply_summary_json TEXT
);

CREATE TABLE IF NOT EXISTS run_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    title TEXT NOT NULL,
    channel_title TEXT NOT NULL,
    description TEXT NOT NULL,
    source_playlists_json TEXT NOT NULL,
    source_positions_json TEXT NOT NULL,
    suggested_mood TEXT,
    final_mood TEXT,
    confidence INTEGER NOT NULL,
    reason TEXT NOT NULL,
    is_music INTEGER NOT NULL,
    default_included INTEGER NOT NULL,
    override_applied INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_run_items_run_video
    ON run_items(run_id, video_id);
"""


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def load_settings(self) -> SetupSettings:
        with self.connect() as conn:
            rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        data = {row["key"]: row["value"] for row in rows}
        return SetupSettings(**data)

    def save_settings(self, settings: SetupSettings) -> None:
        payload = settings.model_dump()
        with self.connect() as conn:
            conn.execute("DELETE FROM app_settings")
            conn.executemany(
                "INSERT INTO app_settings (key, value) VALUES (?, ?)",
                [(key, value) for key, value in payload.items()],
            )

    def load_token_payload(self, provider: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM tokens WHERE provider = ?",
                (provider,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["payload"])

    def save_token_payload(self, provider: str, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tokens (provider, payload) VALUES (?, ?)
                ON CONFLICT(provider) DO UPDATE SET payload = excluded.payload
                """,
                (provider, json.dumps(payload)),
            )

    def load_cached_classification(self, cache_key: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM classification_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row["payload"])

    def save_cached_classification(
        self,
        cache_key: str,
        video_id: str,
        metadata_hash: str,
        prompt_version: str,
        payload: dict[str, Any],
        updated_at: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO classification_cache
                    (cache_key, video_id, metadata_hash, prompt_version, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    video_id,
                    metadata_hash,
                    prompt_version,
                    json.dumps(payload),
                    updated_at,
                ),
            )

    def save_run(
        self,
        run_id: str,
        status: RunStatus,
        scope: RunScope,
        source_playlist_id: str | None,
        source_playlist_title: str | None,
        created_at: str,
        summary_json: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs
                    (id, status, scope, source_playlist_id, source_playlist_title, created_at, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    status.value,
                    scope.value,
                    source_playlist_id,
                    source_playlist_title,
                    created_at,
                    json.dumps(summary_json),
                ),
            )
            conn.executemany(
                """
                INSERT INTO run_items
                    (
                        run_id, video_id, title, channel_title, description,
                        source_playlists_json, source_positions_json,
                        suggested_mood, final_mood, confidence, reason,
                        is_music, default_included, override_applied
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item["video_id"],
                        item["title"],
                        item["channel_title"],
                        item["description"],
                        json.dumps(item["source_playlists"]),
                        json.dumps(item["source_positions"]),
                        serialize_mood_labels(item.get("suggested_moods", item.get("suggested_mood"))),
                        serialize_mood_labels(item.get("final_moods", item.get("final_mood"))),
                        item["confidence"],
                        item["reason"],
                        int(item["is_music"]),
                        int(item["default_included"]),
                        int(item.get("override_applied", False)),
                    )
                    for item in items
                ],
            )

    def get_run(self, run_id: str) -> RunDetail | None:
        with self.connect() as conn:
            run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not run_row:
                return None
            item_rows = conn.execute(
                "SELECT * FROM run_items WHERE run_id = ? ORDER BY confidence ASC, LOWER(title), video_id",
                (run_id,),
            ).fetchall()
        items = [
            RunItemView(
                video_id=row["video_id"],
                title=row["title"],
                channel_title=row["channel_title"],
                description=row["description"],
                source_playlists=json.loads(row["source_playlists_json"]),
                source_positions=json.loads(row["source_positions_json"]),
                suggested_moods=deserialize_mood_labels(row["suggested_mood"]),
                final_moods=deserialize_mood_labels(row["final_mood"]),
                confidence=row["confidence"],
                reason=row["reason"],
                is_music=bool(row["is_music"]),
                default_included=bool(row["default_included"]),
                override_applied=bool(row["override_applied"]),
            )
            for row in item_rows
        ]
        summary = RunSummary.model_validate(json.loads(run_row["summary_json"]))
        return RunDetail(
            run_id=run_row["id"],
            status=RunStatus(run_row["status"]),
            scope=RunScope(run_row["scope"]),
            source_playlist_id=run_row["source_playlist_id"],
            source_playlist_title=run_row["source_playlist_title"],
            created_at=run_row["created_at"],
            summary=summary,
            items=items,
        )

    def update_run_items(
        self,
        run_id: str,
        final_moods: dict[str, list[str]],
        overrides: set[str],
    ) -> None:
        with self.connect() as conn:
            for video_id, final_mood in final_moods.items():
                conn.execute(
                    """
                    UPDATE run_items
                    SET final_mood = ?, override_applied = ?
                    WHERE run_id = ? AND video_id = ?
                    """,
                    (serialize_mood_labels(final_mood), int(video_id in overrides), run_id, video_id),
                )

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        apply_summary: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, apply_summary_json = ? WHERE id = ?",
                (
                    status.value,
                    json.dumps(apply_summary) if apply_summary is not None else None,
                    run_id,
                ),
            )
