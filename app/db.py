"""SQLite persistence helpers."""

from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

from app.models import (
    CategoryAssignment,
    CategorySetDefinition,
    RunDetail,
    RunItemView,
    RunScope,
    RunStatus,
    RunSummary,
    SetupSettings,
    default_mood_category_set,
    deserialize_mood_labels,
    mood_values_from_assignment,
    serialize_mood_labels,
    utc_now,
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
    category_sets_json TEXT,
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
    category_assignments_json TEXT,
    override_applied INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_run_items_run_video
    ON run_items(run_id, video_id);

CREATE TABLE IF NOT EXISTS custom_category_sets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
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
            self._ensure_column(conn, "runs", "category_sets_json", "TEXT")
            self._ensure_column(conn, "run_items", "category_assignments_json", "TEXT")

    def _ensure_column(self, conn: sqlite3.Connection, table_name: str, column_name: str, column_type: str) -> None:
        columns = [row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

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

    def list_custom_category_sets(self, include_archived: bool = False) -> list[CategorySetDefinition]:
        query = "SELECT * FROM custom_category_sets"
        params: tuple[Any, ...] = ()
        if not include_archived:
            query += " WHERE archived = 0"
        query += " ORDER BY LOWER(name), id"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            CategorySetDefinition(
                id=row["id"],
                name=row["name"],
                description="Custom playlist category.",
                labels=json.loads(row["labels_json"]),
                source="custom",
                prompt=row["prompt"],
                archived=bool(row["archived"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_custom_category_set(self, category_id: str) -> CategorySetDefinition | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM custom_category_sets WHERE id = ?", (category_id,)).fetchone()
        if row is None:
            return None
        return CategorySetDefinition(
            id=row["id"],
            name=row["name"],
            description="Custom playlist category.",
            labels=json.loads(row["labels_json"]),
            source="custom",
            prompt=row["prompt"],
            archived=bool(row["archived"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save_custom_category_set(self, category_set: CategorySetDefinition) -> CategorySetDefinition:
        now = utc_now()
        existing = self.get_custom_category_set(category_set.id)
        created_at = category_set.created_at or (existing.created_at if existing is not None else now)
        saved = category_set.model_copy(
            update={
                "source": "custom",
                "created_at": created_at,
                "updated_at": now,
            }
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO custom_category_sets
                    (id, name, prompt, labels_json, archived, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    prompt = excluded.prompt,
                    labels_json = excluded.labels_json,
                    archived = excluded.archived,
                    updated_at = excluded.updated_at
                """,
                (
                    saved.id,
                    saved.name,
                    saved.prompt,
                    json.dumps([label.model_dump(mode="json") for label in saved.labels]),
                    int(saved.archived),
                    saved.created_at,
                    saved.updated_at,
                ),
            )
        return saved

    def archive_custom_category_set(self, category_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE custom_category_sets SET archived = 1, updated_at = ? WHERE id = ?",
                (utc_now(), category_id),
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
        category_sets: list[CategorySetDefinition] | None = None,
    ) -> None:
        category_sets = category_sets or [default_mood_category_set()]
        run_items = [RunItemView.model_validate(item) for item in items]
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs
                    (
                        id, status, scope, source_playlist_id, source_playlist_title,
                        created_at, summary_json, category_sets_json
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    status.value,
                    scope.value,
                    source_playlist_id,
                    source_playlist_title,
                    created_at,
                    json.dumps(summary_json),
                    json.dumps([category.model_dump(mode="json") for category in category_sets]),
                ),
            )
            conn.executemany(
                """
                INSERT INTO run_items
                    (
                        run_id, video_id, title, channel_title, description,
                        source_playlists_json, source_positions_json,
                        suggested_mood, final_mood, confidence, reason,
                        is_music, default_included, category_assignments_json, override_applied
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item.video_id,
                        item.title,
                        item.channel_title,
                        item.description,
                        json.dumps(item.source_playlists),
                        json.dumps(item.source_positions),
                        serialize_mood_labels(item.suggested_moods),
                        serialize_mood_labels(item.final_moods),
                        item.confidence,
                        item.reason,
                        int(item.is_music),
                        int(item.default_included),
                        json.dumps(
                            {
                                "suggested": [
                                    assignment.model_dump(mode="json")
                                    for assignment in item.suggested_assignments
                                ],
                                "final": [
                                    assignment.model_dump(mode="json")
                                    for assignment in item.final_assignments
                                ],
                            }
                        ),
                        int(item.override_applied),
                    )
                    for item in run_items
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
        items = []
        for row in item_rows:
            assignments_payload = {}
            if row["category_assignments_json"]:
                try:
                    assignments_payload = json.loads(row["category_assignments_json"])
                except json.JSONDecodeError:
                    assignments_payload = {}
            item_data = {
                "video_id": row["video_id"],
                "title": row["title"],
                "channel_title": row["channel_title"],
                "description": row["description"],
                "source_playlists": json.loads(row["source_playlists_json"]),
                "source_positions": json.loads(row["source_positions_json"]),
                "confidence": row["confidence"],
                "reason": row["reason"],
                "is_music": bool(row["is_music"]),
                "default_included": bool(row["default_included"]),
                "override_applied": bool(row["override_applied"]),
            }
            if assignments_payload:
                item_data["suggested_assignments"] = assignments_payload.get("suggested", [])
                item_data["final_assignments"] = assignments_payload.get("final", [])
            else:
                item_data["suggested_moods"] = deserialize_mood_labels(row["suggested_mood"])
                item_data["final_moods"] = deserialize_mood_labels(row["final_mood"])
            items.append(RunItemView.model_validate(item_data))
        summary = RunSummary.model_validate(json.loads(run_row["summary_json"]))
        category_sets = [default_mood_category_set()]
        if run_row["category_sets_json"]:
            try:
                category_sets = [
                    CategorySetDefinition.model_validate(category)
                    for category in json.loads(run_row["category_sets_json"])
                ]
            except (json.JSONDecodeError, ValueError):
                category_sets = [default_mood_category_set()]
        return RunDetail(
            run_id=run_row["id"],
            status=RunStatus(run_row["status"]),
            scope=RunScope(run_row["scope"]),
            source_playlist_id=run_row["source_playlist_id"],
            source_playlist_title=run_row["source_playlist_title"],
            created_at=run_row["created_at"],
            summary=summary,
            items=items,
            category_sets=category_sets,
        )

    def update_run_items(
        self,
        run_id: str,
        final_assignments: dict[str, list[CategoryAssignment] | list[str]],
        overrides: set[str],
    ) -> None:
        run = self.get_run(run_id)
        if run is None:
            return
        existing_items = {item.video_id: item for item in run.items}
        with self.connect() as conn:
            for video_id, values in final_assignments.items():
                item = existing_items.get(video_id)
                if item is None:
                    continue
                if values and all(isinstance(value, CategoryAssignment) for value in values):
                    assignments = list(values)  # type: ignore[arg-type]
                elif values and all(isinstance(value, dict) for value in values):
                    assignments = [CategoryAssignment.model_validate(value) for value in values]  # type: ignore[arg-type]
                else:
                    assignments = [
                        CategoryAssignment(
                            category_id="mood",
                            label_slugs=[
                                default_mood_category_set().label_for_slug(value).slug
                                if default_mood_category_set().label_for_slug(value)
                                else value
                                for value in values  # type: ignore[union-attr]
                            ],
                        )
                    ]
                mood_assignment = next(
                    (assignment for assignment in assignments if assignment.category_id == "mood"),
                    None,
                )
                payload = {
                    "suggested": [
                        assignment.model_dump(mode="json")
                        for assignment in item.suggested_assignments
                    ],
                    "final": [assignment.model_dump(mode="json") for assignment in assignments],
                }
                conn.execute(
                    """
                    UPDATE run_items
                    SET final_mood = ?, category_assignments_json = ?, override_applied = ?
                    WHERE run_id = ? AND video_id = ?
                    """,
                    (
                        serialize_mood_labels(mood_values_from_assignment(mood_assignment)),
                        json.dumps(payload),
                        int(video_id in overrides),
                        run_id,
                        video_id,
                    ),
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

    def delete_authorized_youtube_data(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM run_items")
            conn.execute("DELETE FROM runs")
            conn.execute("DELETE FROM classification_cache")
            conn.execute("DELETE FROM tokens")
