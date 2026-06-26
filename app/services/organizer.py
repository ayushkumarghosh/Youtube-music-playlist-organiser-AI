"""Preview and apply orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from collections import defaultdict
import uuid
from typing import Any

from app.constants import CATEGORY_SET_MOOD_ID
from app.db import Database
from app.models import (
    CategoryAssignment,
    CategorySetDefinition,
    RunDetail,
    RunItemView,
    RunScope,
    RunStatus,
    RunSummary,
    SongCandidate,
    default_mood_category_set,
    find_category_set,
    mood_assignment_from_values,
    slugify_label,
    utc_now,
)
from app.services.azure_openai import AzureOpenAIClassifier
from app.services.youtube import YouTubeService


ProgressCallback = Callable[[dict[str, object]], None]
AssignmentOverrides = dict[str, dict[str, list[str]]]


def dedupe_candidates(items: list) -> list[SongCandidate]:
    candidates: dict[str, SongCandidate] = {}
    for item in items:
        candidate = candidates.get(item.video_id)
        if candidate is None:
            candidate = SongCandidate(
                video_id=item.video_id,
                title=item.title,
                channel_title=item.channel_title,
                description=item.description,
                source_playlists=[item.playlist_title],
                source_playlist_ids=[item.playlist_id],
                source_positions=[item.position],
            )
            candidates[item.video_id] = candidate
            continue
        candidate.source_playlists.append(item.playlist_title)
        candidate.source_playlist_ids.append(item.playlist_id)
        candidate.source_positions.append(item.position)
    return list(candidates.values())


def playlist_key(category_id: str, label_slug: str) -> str:
    return f"{slugify_label(category_id)}:{slugify_label(label_slug)}"


def normalize_override_map(overrides: AssignmentOverrides | dict[str, list[str]]) -> AssignmentOverrides:
    if not overrides:
        return {}
    if all(isinstance(value, dict) for value in overrides.values()):
        return {
            slugify_label(category_id): {
                str(video_id): [slugify_label(value) for value in values if str(value).strip()]
                for video_id, values in video_overrides.items()
            }
            for category_id, video_overrides in overrides.items()  # type: ignore[union-attr]
        }
    return {
        CATEGORY_SET_MOOD_ID: {
            str(video_id): [slugify_label(value) for value in values if str(value).strip()]
            for video_id, values in overrides.items()  # type: ignore[union-attr]
        }
    }


def normalize_assignment_for_category(
    category: CategorySetDefinition,
    raw_label_slugs: list[str],
    current_assignment: CategoryAssignment | None = None,
) -> CategoryAssignment:
    allowed = {label.slug for label in category.labels}
    label_slugs = [slug for slug in dict.fromkeys(raw_label_slugs) if slug in allowed]
    return CategoryAssignment(
        category_id=category.id,
        label_slugs=label_slugs,
        confidence=current_assignment.confidence if current_assignment else 0,
        reason=current_assignment.reason if current_assignment else "",
    )


def coerce_classification_assignments(
    classification: Any,
    category_sets: list[CategorySetDefinition],
) -> list[CategoryAssignment]:
    if hasattr(classification, "assignments"):
        assignments = [
            assignment if isinstance(assignment, CategoryAssignment) else CategoryAssignment.model_validate(assignment)
            for assignment in classification.assignments
        ]
    else:
        assignment = mood_assignment_from_values(
            getattr(classification, "moods", []),
            confidence=int(getattr(classification, "confidence", 0) or 0),
            reason=str(getattr(classification, "reason", "")),
        )
        assignments = [assignment]

    assignments_by_category = {assignment.category_id: assignment for assignment in assignments}
    if getattr(classification, "is_music", False):
        for category in category_sets:
            current = assignments_by_category.get(category.id)
            raw_labels = current.label_slugs if current is not None else []
            assignments_by_category[category.id] = normalize_assignment_for_category(category, raw_labels, current)
    else:
        assignments_by_category = {
            category.id: CategoryAssignment(category_id=category.id, label_slugs=[])
            for category in category_sets
        }
    return [assignments_by_category[category.id] for category in category_sets]


def aggregate_assignment_confidence(assignments: list[CategoryAssignment], fallback: int = 0) -> int:
    confident = [assignment.confidence for assignment in assignments if assignment.label_slugs]
    return max(confident) if confident else fallback


def aggregate_assignment_reason(assignments: list[CategoryAssignment], fallback: str = "") -> str:
    reasons = [assignment.reason for assignment in assignments if assignment.label_slugs and assignment.reason]
    return reasons[0] if reasons else fallback or "No category matched."


class OrganizerService:
    def __init__(
        self,
        db: Database,
        youtube_service: YouTubeService,
        classifier: AzureOpenAIClassifier,
    ) -> None:
        self.db = db
        self.youtube_service = youtube_service
        self.classifier = classifier

    def create_preview(
        self,
        scope: RunScope,
        source_playlist_id: str | None = None,
        source_playlist_ids: list[str] | None = None,
        persist: bool = True,
        category_sets: list[CategorySetDefinition] | None = None,
    ) -> RunDetail:
        category_sets = category_sets or [default_mood_category_set()]
        source_playlists = self.youtube_service.get_source_playlists(
            scope,
            source_playlist_id,
            source_playlist_ids,
        )
        playlist_items = []
        for playlist in source_playlists:
            playlist_items.extend(
                self.youtube_service.list_playlist_items(playlist.playlist_id, playlist.title)
            )
        candidates = dedupe_candidates(playlist_items)
        try:
            classifications = asyncio.run(self.classifier.classify_candidates(candidates, category_sets))
        except TypeError:
            classifications = asyncio.run(self.classifier.classify_candidates(candidates))

        items: list[RunItemView] = []
        for candidate in candidates:
            classification = classifications[candidate.video_id]
            assignments = coerce_classification_assignments(classification, category_sets)
            default_included = any(assignment.label_slugs for assignment in assignments)
            items.append(
                RunItemView(
                    video_id=candidate.video_id,
                    title=candidate.title,
                    channel_title=candidate.channel_title,
                    description=candidate.description,
                    source_playlists=candidate.source_playlists,
                    source_positions=candidate.source_positions,
                    suggested_assignments=assignments,
                    final_assignments=assignments,
                    confidence=aggregate_assignment_confidence(
                        assignments,
                        int(getattr(classification, "confidence", 0) or 0),
                    ),
                    reason=aggregate_assignment_reason(assignments, str(getattr(classification, "reason", ""))),
                    is_music=bool(getattr(classification, "is_music", False)),
                    default_included=default_included,
                    override_applied=False,
                )
            )

        items.sort(
            key=lambda item: (
                item.confidence,
                item.title.lower(),
                item.video_id,
            )
        )
        summary = RunSummary(
            total_candidates=len(candidates),
            classified_count=len(candidates),
            default_included_count=sum(1 for item in items if item.default_included),
            excluded_count=sum(1 for item in items if not item.default_included),
        )
        run_id = str(uuid.uuid4())
        source_title = None
        if scope == RunScope.SINGLE_PLAYLIST and source_playlists:
            source_title = source_playlists[0].title
        elif scope == RunScope.SELECTED_PLAYLISTS:
            source_title = f"{len(source_playlists)} selected playlists"
        if not persist:
            return RunDetail(
                run_id=run_id,
                status=RunStatus.PREVIEWED,
                scope=scope,
                source_playlist_id=source_playlist_id or ",".join(source_playlist_ids or []),
                source_playlist_title=source_title,
                created_at=utc_now(),
                summary=summary,
                items=items,
                category_sets=category_sets,
            )
        self.db.save_run(
            run_id=run_id,
            status=RunStatus.PREVIEWED,
            scope=scope,
            source_playlist_id=source_playlist_id or ",".join(source_playlist_ids or []),
            source_playlist_title=source_title,
            created_at=utc_now(),
            summary_json=summary.model_dump(mode="json"),
            items=[item.model_dump(mode="json") for item in items],
            category_sets=category_sets,
        )
        run = self.db.get_run(run_id)
        if run is None:
            raise RuntimeError("Run was not persisted.")
        return run

    def load_run(self, run_id: str) -> RunDetail | None:
        return self.db.get_run(run_id)

    def apply_run(
        self,
        run_id: str,
        overrides: AssignmentOverrides | dict[str, list[str]],
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, object]:
        run = self.db.get_run(run_id)
        if run is None:
            raise ValueError("Run not found.")

        override_map = normalize_override_map(overrides)
        final_assignments_by_video: dict[str, list[CategoryAssignment]] = {}
        override_ids: set[str] = set()
        for item in run.items:
            final_assignments = self._apply_overrides_to_item(run.category_sets, item, override_map)
            if self._assignment_signature(final_assignments) != self._assignment_signature(item.final_assignments):
                override_ids.add(item.video_id)
            final_assignments_by_video[item.video_id] = final_assignments
        self.db.update_run_items(run_id, final_assignments_by_video, override_ids)

        updated_run = self.db.get_run(run_id)
        if updated_run is None:
            raise RuntimeError("Updated run not found.")
        sync_summary = self.sync_run_detail(updated_run, progress_callback)
        self.db.update_run_status(run_id, RunStatus.APPLIED, sync_summary)
        return sync_summary

    def apply_run_detail(
        self,
        run: RunDetail,
        overrides: AssignmentOverrides | dict[str, list[str]],
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, object]:
        override_map = normalize_override_map(overrides)
        updated_items: list[RunItemView] = []
        for item in run.items:
            final_assignments = self._apply_overrides_to_item(run.category_sets, item, override_map)
            updated_item = item.model_copy(
                update={
                    "final_assignments": final_assignments,
                    "override_applied": self._assignment_signature(final_assignments)
                    != self._assignment_signature(item.final_assignments),
                }
            )
            updated_items.append(updated_item)

        updated_run = run.model_copy(update={"items": updated_items})
        return self.sync_run_detail(updated_run, progress_callback)

    def sync_run_detail(
        self,
        run: RunDetail,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, object]:
        def report(payload: dict[str, object]) -> None:
            if progress_callback is not None:
                progress_callback(payload)

        grouped_video_ids: dict[str, list[tuple[list[str], list[int], str]]] = defaultdict(list)
        for item in run.items:
            for assignment in item.final_assignments:
                for label_slug in assignment.label_slugs:
                    grouped_video_ids[playlist_key(assignment.category_id, label_slug)].append(
                        (item.source_playlists, item.source_positions, item.video_id)
                    )

        label_count = sum(len(category.labels) for category in run.category_sets)
        report(
            {
                "stage": "preparing",
                "message": "Preparing category assignments",
                "current": 1,
                "total": label_count + 2,
                "percent": 8,
            }
        )
        try:
            managed_playlists = self.youtube_service.ensure_managed_playlists(
                run.scope,
                run.source_playlist_title,
                run.category_sets,
            )
        except TypeError:
            managed_playlists = self.youtube_service.ensure_managed_playlists(
                run.scope,
                run.source_playlist_title,
            )
        normalized_managed_playlists = {}
        for key, playlist in managed_playlists.items():
            if ":" in key:
                normalized_managed_playlists[key] = playlist
                continue
            normalized_managed_playlists[playlist_key(CATEGORY_SET_MOOD_ID, key)] = playlist
        managed_playlists = normalized_managed_playlists

        sync_summary: dict[str, object] = {"playlists": {}, "total_assignments": 0}
        total_steps = len(managed_playlists) + 2
        for index, (key, playlist) in enumerate(managed_playlists.items(), start=2):
            category_id, label_slug = key.split(":", 1)
            category = find_category_set(run.category_sets, category_id)
            label = category.label_for_slug(label_slug) if category is not None else None
            label_name = label.name if label is not None else label_slug
            category_name = category.name if category is not None else category_id
            ordered_video_ids = [
                video_id
                for _, _, video_id in sorted(
                    grouped_video_ids.get(key, []),
                    key=lambda row: (
                        [name.lower() for name in row[0]],
                        row[1],
                        row[2],
                    ),
                )
            ]
            report(
                {
                    "stage": "syncing",
                    "message": f"Syncing {label_name}",
                    "playlist": label_name,
                    "category": category_name,
                    "video_count": len(ordered_video_ids),
                    "current": index,
                    "total": total_steps,
                    "percent": round((index - 1) / total_steps * 100),
                }
            )
            sync_counts = self.youtube_service.reconcile_playlist(
                playlist.playlist_id,
                ordered_video_ids,
            )
            summary_key = f"{category_name}: {label_name}"
            sync_summary["playlists"][summary_key] = {
                "playlist_id": playlist.playlist_id,
                "title": playlist.title,
                "category": category_name,
                "label": label_name,
                "video_count": len(ordered_video_ids),
                "sync_counts": sync_counts,
            }
            sync_summary["total_assignments"] = int(sync_summary["total_assignments"]) + len(ordered_video_ids)

        report(
            {
                "stage": "complete",
                "message": "Category playlists synced",
                "current": total_steps,
                "total": total_steps,
                "percent": 100,
            }
        )
        return sync_summary

    def _apply_overrides_to_item(
        self,
        category_sets: list[CategorySetDefinition],
        item: RunItemView,
        overrides: AssignmentOverrides,
    ) -> list[CategoryAssignment]:
        assignments: list[CategoryAssignment] = []
        for category in category_sets:
            current = item.final_assignment_for(category.id) or CategoryAssignment(category_id=category.id)
            selected = overrides.get(category.id, {}).get(item.video_id, current.label_slugs)
            assignments.append(normalize_assignment_for_category(category, selected, current))
        return assignments

    def _assignment_signature(self, assignments: list[CategoryAssignment]) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            sorted((assignment.category_id, tuple(assignment.label_slugs)) for assignment in assignments)
        )
