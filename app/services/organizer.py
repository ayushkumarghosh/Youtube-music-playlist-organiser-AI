"""Preview and apply orchestration."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import uuid

from app.db import Database
from app.models import (
    ApprovedAssignment,
    RunDetail,
    RunItemView,
    RunScope,
    RunStatus,
    RunSummary,
    SongCandidate,
    utc_now,
)
from app.services.azure_openai import AzureOpenAIClassifier
from app.services.youtube import YouTubeService


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
    ) -> RunDetail:
        source_playlists = self.youtube_service.get_source_playlists(scope, source_playlist_id)
        playlist_items = []
        for playlist in source_playlists:
            playlist_items.extend(
                self.youtube_service.list_playlist_items(playlist.playlist_id, playlist.title)
            )
        candidates = dedupe_candidates(playlist_items)
        classifications = asyncio.run(self.classifier.classify_candidates(candidates))

        items: list[RunItemView] = []
        for candidate in candidates:
            classification = classifications[candidate.video_id]
            default_included = bool(classification.is_music and classification.mood)
            items.append(
                RunItemView(
                    video_id=candidate.video_id,
                    title=candidate.title,
                    channel_title=candidate.channel_title,
                    description=candidate.description,
                    source_playlists=candidate.source_playlists,
                    source_positions=candidate.source_positions,
                    suggested_mood=classification.mood,
                    final_mood=classification.mood,
                    confidence=classification.confidence,
                    reason=classification.reason,
                    is_music=classification.is_music,
                    default_included=default_included,
                    override_applied=False,
                )
            )

        items.sort(
            key=lambda item: (
                [playlist.lower() for playlist in item.source_playlists],
                item.source_positions,
                item.title.lower(),
            )
        )
        summary = RunSummary(
            total_candidates=len(candidates),
            classified_count=len(candidates),
            default_included_count=sum(1 for item in items if item.default_included),
            excluded_count=sum(1 for item in items if not item.default_included),
        )
        run_id = str(uuid.uuid4())
        source_title = source_playlists[0].title if scope == RunScope.SINGLE_PLAYLIST and source_playlists else None
        self.db.save_run(
            run_id=run_id,
            status=RunStatus.PREVIEWED,
            scope=scope,
            source_playlist_id=source_playlist_id,
            source_playlist_title=source_title,
            created_at=utc_now(),
            summary_json=summary.model_dump(mode="json"),
            items=[item.model_dump(mode="json") for item in items],
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
        overrides: dict[str, str],
    ) -> dict[str, object]:
        run = self.db.get_run(run_id)
        if run is None:
            raise ValueError("Run not found.")

        final_moods: dict[str, str | None] = {}
        override_ids: set[str] = set()
        assignments: list[ApprovedAssignment] = []
        for item in run.items:
            override_supplied = item.video_id in overrides
            selected_value = overrides.get(item.video_id, item.final_mood or "")
            final_mood = selected_value or None
            if final_mood != item.final_mood:
                override_ids.add(item.video_id)
            final_moods[item.video_id] = final_mood or None
            assignments.append(
                ApprovedAssignment(
                    video_id=item.video_id,
                    final_mood=final_mood or None,
                    source_scope=run.scope,
                    override_applied=(item.video_id in override_ids) if override_supplied else False,
                )
            )
        self.db.update_run_items(run_id, final_moods, override_ids)

        grouped_video_ids: dict[str, list[tuple[list[str], list[int], str]]] = defaultdict(list)
        updated_run = self.db.get_run(run_id)
        if updated_run is None:
            raise RuntimeError("Updated run not found.")
        for item in updated_run.items:
            if not item.final_mood:
                continue
            grouped_video_ids[str(item.final_mood)].append(
                (item.source_playlists, item.source_positions, item.video_id)
            )

        managed_playlists = self.youtube_service.ensure_managed_playlists(
            updated_run.scope,
            updated_run.source_playlist_title,
        )

        sync_summary: dict[str, object] = {"playlists": {}, "total_assignments": 0}
        for mood, playlist in managed_playlists.items():
            ordered_video_ids = [
                video_id
                for _, _, video_id in sorted(
                    grouped_video_ids.get(mood, []),
                    key=lambda row: (
                        [name.lower() for name in row[0]],
                        row[1],
                        row[2],
                    ),
                )
            ]
            sync_counts = self.youtube_service.reconcile_playlist(
                playlist.playlist_id,
                ordered_video_ids,
            )
            sync_summary["playlists"][mood] = {
                "playlist_id": playlist.playlist_id,
                "title": playlist.title,
                "video_count": len(ordered_video_ids),
                "sync_counts": sync_counts,
            }
            sync_summary["total_assignments"] = int(sync_summary["total_assignments"]) + len(ordered_video_ids)

        self.db.update_run_status(run_id, RunStatus.APPLIED, sync_summary)
        return sync_summary
