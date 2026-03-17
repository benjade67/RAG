from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rag_pdf.config import get_project_root


@dataclass(frozen=True)
class IndexStatus:
    state: str = "not_indexed"
    last_indexed_at: str | None = None
    chunk_count: int = 0
    embedding_backend: str | None = None
    embedding_model: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class PlanRecord:
    plan_id: str
    logical_plan_id: str
    version_id: str
    filename: str
    file_path: str
    checksum: str
    added_at: str
    revision_label: str | None = None
    duplicate_of_version_id: str | None = None
    is_current: bool = True
    is_archived: bool = False
    index_status: IndexStatus = field(default_factory=IndexStatus)


@dataclass(frozen=True)
class ManagedPlanRecord:
    logical_plan_id: str
    display_name: str
    canonical_name: str
    project_code: str | None = None
    discipline: str | None = None
    notes: str | None = None
    is_archived: bool = False
    active_version_id: str | None = None
    versions: tuple[PlanRecord, ...] = ()

    @property
    def duplicate_versions(self) -> tuple[PlanRecord, ...]:
        return tuple(version for version in self.versions if version.duplicate_of_version_id)

    @property
    def current_version(self) -> PlanRecord | None:
        return next((version for version in self.versions if version.version_id == self.active_version_id), None)


class PlanCatalog:
    SCHEMA_VERSION = 2

    def __init__(self, catalog_path: Path | None = None) -> None:
        self._catalog_path = catalog_path or get_project_root() / "data" / "plans" / "catalog.json"
        self._catalog_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._catalog_path.exists():
            self._save_payload({"schema_version": self.SCHEMA_VERSION, "plans": []})

    def list_plans(self) -> list[PlanRecord]:
        """Return the current active document versions used by the assistant."""
        plans = []
        for managed_plan in self.list_managed_plans():
            if managed_plan.is_archived:
                continue
            current_version = managed_plan.current_version
            if current_version is not None and not current_version.is_archived:
                plans.append(current_version)
        return sorted(plans, key=lambda item: item.filename.lower())

    def list_managed_plans(self) -> list[ManagedPlanRecord]:
        payload = self._load_payload()
        return [
            self._deserialize_managed_plan(item)
            for item in sorted(payload["plans"], key=lambda item: item["display_name"].lower())
        ]

    def get_plan(self, plan_id: str) -> PlanRecord | None:
        return next((plan for plan in self.list_all_versions() if plan.plan_id == plan_id), None)

    def get_managed_plan(self, logical_plan_id: str) -> ManagedPlanRecord | None:
        return next(
            (plan for plan in self.list_managed_plans() if plan.logical_plan_id == logical_plan_id),
            None,
        )

    def list_all_versions(self) -> list[PlanRecord]:
        versions: list[PlanRecord] = []
        for managed_plan in self.list_managed_plans():
            versions.extend(managed_plan.versions)
        return sorted(versions, key=lambda item: (item.logical_plan_id, item.filename.lower()))

    def register_plan(self, file_path: str | Path) -> PlanRecord:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Plan introuvable: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Le fichier n'est pas un PDF: {path}")

        checksum = self._compute_checksum(path)
        revision_label = self._extract_revision_label(path.stem)
        canonical_name = self._canonicalize_name(path.stem)
        display_name = self._humanize_name(canonical_name)
        logical_plan_id = self._build_logical_plan_id(canonical_name)
        version_id = self._build_version_id(path)

        payload = self._load_payload()
        managed_plans = [self._deserialize_managed_plan(item) for item in payload["plans"]]

        existing_by_path = {
            version.file_path: (managed_plan.logical_plan_id, version)
            for managed_plan in managed_plans
            for version in managed_plan.versions
        }
        duplicate_of = self._find_duplicate_version_id(managed_plans, checksum, exclude_path=str(path))

        if str(path) in existing_by_path:
            previous_logical_plan_id, existing_version = existing_by_path[str(path)]
            logical_plan_id = previous_logical_plan_id
            version_id = existing_version.version_id

        new_record = PlanRecord(
            plan_id=version_id,
            logical_plan_id=logical_plan_id,
            version_id=version_id,
            filename=path.name,
            file_path=str(path),
            checksum=checksum,
            added_at=self._existing_added_at(existing_by_path.get(str(path)), default=datetime.now(UTC).isoformat()),
            revision_label=revision_label,
            duplicate_of_version_id=duplicate_of,
        )

        target_plan = next((plan for plan in managed_plans if plan.logical_plan_id == logical_plan_id), None)
        if target_plan is None:
            target_plan = ManagedPlanRecord(
                logical_plan_id=logical_plan_id,
                display_name=display_name,
                canonical_name=canonical_name,
                active_version_id=version_id,
                versions=(new_record,),
            )
            managed_plans.append(target_plan)
        else:
            if existing_by_path.get(str(path)) and existing_by_path[str(path)][1].checksum == checksum:
                new_record = self._reuse_existing_index_status(existing_by_path[str(path)][1], new_record)

            versions = [
                new_record if version.file_path == str(path) else version
                for version in target_plan.versions
            ]
            if not any(version.file_path == str(path) for version in target_plan.versions):
                versions.append(new_record)

            active_version_id = target_plan.active_version_id or version_id
            if len(versions) == 1:
                active_version_id = versions[0].version_id
            elif target_plan.active_version_id is None or not any(
                version.version_id == target_plan.active_version_id and not version.is_archived
                for version in versions
            ):
                active_version_id = self._choose_active_version(versions)

            target_plan = ManagedPlanRecord(
                logical_plan_id=target_plan.logical_plan_id,
                display_name=target_plan.display_name,
                canonical_name=target_plan.canonical_name,
                project_code=target_plan.project_code,
                discipline=target_plan.discipline,
                notes=target_plan.notes,
                is_archived=target_plan.is_archived,
                active_version_id=active_version_id,
                versions=tuple(sorted(versions, key=self._version_sort_key)),
            )
            managed_plans = [
                target_plan if managed_plan.logical_plan_id == logical_plan_id else managed_plan
                for managed_plan in managed_plans
            ]

        managed_plans = self._refresh_duplicate_flags(managed_plans)
        self._save_managed_plans(managed_plans)
        return self.get_plan(version_id) or new_record

    def register_directory(self, directory_path: str | Path, recursive: bool = True) -> list[PlanRecord]:
        path = Path(directory_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dossier introuvable: {path}")
        if not path.is_dir():
            raise ValueError(f"Le chemin n'est pas un dossier: {path}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        added: list[PlanRecord] = []
        for pdf_path in sorted(path.glob(pattern)):
            added.append(self.register_plan(pdf_path))
        return added

    def mark_indexed(
        self,
        plan_id: str,
        chunk_count: int,
        embedding_backend: str,
        embedding_model: str,
    ) -> PlanRecord:
        return self._update_version(
            plan_id,
            lambda version: PlanRecord(
                plan_id=version.plan_id,
                logical_plan_id=version.logical_plan_id,
                version_id=version.version_id,
                filename=version.filename,
                file_path=version.file_path,
                checksum=version.checksum,
                added_at=version.added_at,
                revision_label=version.revision_label,
                duplicate_of_version_id=version.duplicate_of_version_id,
                is_current=version.is_current,
                is_archived=version.is_archived,
                index_status=IndexStatus(
                    state="indexed",
                    last_indexed_at=datetime.now(UTC).isoformat(),
                    chunk_count=chunk_count,
                    embedding_backend=embedding_backend,
                    embedding_model=embedding_model,
                    error_message=None,
                ),
            ),
        )

    def mark_index_error(
        self,
        plan_id: str,
        embedding_backend: str,
        embedding_model: str,
        error_message: str,
    ) -> PlanRecord:
        return self._update_version(
            plan_id,
            lambda version: PlanRecord(
                plan_id=version.plan_id,
                logical_plan_id=version.logical_plan_id,
                version_id=version.version_id,
                filename=version.filename,
                file_path=version.file_path,
                checksum=version.checksum,
                added_at=version.added_at,
                revision_label=version.revision_label,
                duplicate_of_version_id=version.duplicate_of_version_id,
                is_current=version.is_current,
                is_archived=version.is_archived,
                index_status=IndexStatus(
                    state="error",
                    last_indexed_at=datetime.now(UTC).isoformat(),
                    chunk_count=0,
                    embedding_backend=embedding_backend,
                    embedding_model=embedding_model,
                    error_message=error_message,
                ),
            ),
        )

    def clear_index_status(self, plan_id: str) -> PlanRecord:
        return self._update_version(
            plan_id,
            lambda version: PlanRecord(
                plan_id=version.plan_id,
                logical_plan_id=version.logical_plan_id,
                version_id=version.version_id,
                filename=version.filename,
                file_path=version.file_path,
                checksum=version.checksum,
                added_at=version.added_at,
                revision_label=version.revision_label,
                duplicate_of_version_id=version.duplicate_of_version_id,
                is_current=version.is_current,
                is_archived=version.is_archived,
                index_status=IndexStatus(),
            ),
        )

    def update_plan_metadata(
        self,
        logical_plan_id: str,
        display_name: str | None = None,
        project_code: str | None = None,
        discipline: str | None = None,
        notes: str | None = None,
    ) -> ManagedPlanRecord:
        return self._update_managed_plan(
            logical_plan_id,
            lambda plan: ManagedPlanRecord(
                logical_plan_id=plan.logical_plan_id,
                display_name=(display_name or plan.display_name).strip() or plan.display_name,
                canonical_name=plan.canonical_name,
                project_code=self._clean_optional(project_code, plan.project_code),
                discipline=self._clean_optional(discipline, plan.discipline),
                notes=self._clean_optional(notes, plan.notes),
                is_archived=plan.is_archived,
                active_version_id=plan.active_version_id,
                versions=plan.versions,
            ),
        )

    def set_active_version(self, logical_plan_id: str, version_id: str) -> ManagedPlanRecord:
        return self._update_managed_plan(
            logical_plan_id,
            lambda plan: self._validate_active_version(plan, version_id),
        )

    def set_plan_archived(self, logical_plan_id: str, archived: bool) -> ManagedPlanRecord:
        return self._update_managed_plan(
            logical_plan_id,
            lambda plan: ManagedPlanRecord(
                logical_plan_id=plan.logical_plan_id,
                display_name=plan.display_name,
                canonical_name=plan.canonical_name,
                project_code=plan.project_code,
                discipline=plan.discipline,
                notes=plan.notes,
                is_archived=archived,
                active_version_id=plan.active_version_id,
                versions=plan.versions,
            ),
        )

    def remove_version(self, version_id: str) -> PlanRecord | None:
        managed_plans = self.list_managed_plans()
        removed_version: PlanRecord | None = None
        updated_plans: list[ManagedPlanRecord] = []

        for managed_plan in managed_plans:
            remaining_versions = [version for version in managed_plan.versions if version.version_id != version_id]
            if len(remaining_versions) != len(managed_plan.versions):
                removed_version = next(version for version in managed_plan.versions if version.version_id == version_id)
                if not remaining_versions:
                    continue
                active_version_id = managed_plan.active_version_id
                if active_version_id == version_id:
                    active_version_id = self._choose_active_version(remaining_versions)
                updated_plans.append(
                    ManagedPlanRecord(
                        logical_plan_id=managed_plan.logical_plan_id,
                        display_name=managed_plan.display_name,
                        canonical_name=managed_plan.canonical_name,
                        project_code=managed_plan.project_code,
                        discipline=managed_plan.discipline,
                        notes=managed_plan.notes,
                        is_archived=managed_plan.is_archived,
                        active_version_id=active_version_id,
                        versions=tuple(sorted(remaining_versions, key=self._version_sort_key)),
                    )
                )
            else:
                updated_plans.append(managed_plan)

        if removed_version is None:
            return None

        updated_plans = self._refresh_duplicate_flags(updated_plans)
        self._save_managed_plans(updated_plans)
        return removed_version

    def remove_plan(self, logical_plan_id: str) -> ManagedPlanRecord | None:
        managed_plans = self.list_managed_plans()
        removed_plan = next((plan for plan in managed_plans if plan.logical_plan_id == logical_plan_id), None)
        if removed_plan is None:
            return None

        updated_plans = [plan for plan in managed_plans if plan.logical_plan_id != logical_plan_id]
        updated_plans = self._refresh_duplicate_flags(updated_plans)
        self._save_managed_plans(updated_plans)
        return removed_plan

    def _update_version(self, plan_id: str, updater) -> PlanRecord:
        managed_plans = self.list_managed_plans()
        updated_version: PlanRecord | None = None
        refreshed_plans: list[ManagedPlanRecord] = []

        for managed_plan in managed_plans:
            updated_versions = []
            changed = False
            for version in managed_plan.versions:
                if version.plan_id == plan_id:
                    updated_version = updater(version)
                    updated_versions.append(updated_version)
                    changed = True
                else:
                    updated_versions.append(version)

            if changed:
                refreshed_plans.append(
                    ManagedPlanRecord(
                        logical_plan_id=managed_plan.logical_plan_id,
                        display_name=managed_plan.display_name,
                        canonical_name=managed_plan.canonical_name,
                        project_code=managed_plan.project_code,
                        discipline=managed_plan.discipline,
                        notes=managed_plan.notes,
                        is_archived=managed_plan.is_archived,
                        active_version_id=managed_plan.active_version_id,
                        versions=tuple(sorted(updated_versions, key=self._version_sort_key)),
                    )
                )
            else:
                refreshed_plans.append(managed_plan)

        if updated_version is None:
            raise ValueError(f"Plan inconnu: {plan_id}")

        self._save_managed_plans(refreshed_plans)
        return updated_version

    def _update_managed_plan(self, logical_plan_id: str, updater) -> ManagedPlanRecord:
        managed_plans = self.list_managed_plans()
        updated_plan: ManagedPlanRecord | None = None
        refreshed_plans: list[ManagedPlanRecord] = []

        for managed_plan in managed_plans:
            if managed_plan.logical_plan_id == logical_plan_id:
                updated_plan = updater(managed_plan)
                refreshed_plans.append(updated_plan)
            else:
                refreshed_plans.append(managed_plan)

        if updated_plan is None:
            raise ValueError(f"Plan metier inconnu: {logical_plan_id}")

        refreshed_plans = self._refresh_duplicate_flags(refreshed_plans)
        self._save_managed_plans(refreshed_plans)
        return next(plan for plan in self.list_managed_plans() if plan.logical_plan_id == logical_plan_id)

    def _load_payload(self) -> dict:
        raw = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            payload = self._migrate_legacy_payload(raw)
            self._save_payload(payload)
            return payload

        if "schema_version" not in raw:
            raw["schema_version"] = self.SCHEMA_VERSION
        raw.setdefault("plans", [])
        return raw

    def _save_managed_plans(self, plans: list[ManagedPlanRecord]) -> None:
        self._save_payload(
            {
                "schema_version": self.SCHEMA_VERSION,
                "plans": [self._serialize_managed_plan(plan) for plan in plans],
            }
        )

    def _save_payload(self, payload: dict) -> None:
        self._catalog_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _migrate_legacy_payload(self, legacy_payload: list[dict]) -> dict:
        managed_plans: dict[str, ManagedPlanRecord] = {}

        for item in legacy_payload:
            filename = item["filename"]
            stem = Path(filename).stem
            canonical_name = self._canonicalize_name(stem)
            logical_plan_id = self._build_logical_plan_id(canonical_name)
            plan_id = item["plan_id"]
            revision_label = self._extract_revision_label(stem)

            version = PlanRecord(
                plan_id=plan_id,
                logical_plan_id=logical_plan_id,
                version_id=plan_id,
                filename=filename,
                file_path=item["file_path"],
                checksum=item["checksum"],
                added_at=item["added_at"],
                revision_label=revision_label,
                index_status=IndexStatus(**(item.get("index_status") or {})),
            )
            managed_plan = managed_plans.get(logical_plan_id)
            if managed_plan is None:
                managed_plans[logical_plan_id] = ManagedPlanRecord(
                    logical_plan_id=logical_plan_id,
                    display_name=self._humanize_name(canonical_name),
                    canonical_name=canonical_name,
                    active_version_id=version.version_id,
                    versions=(version,),
                )
            else:
                versions = tuple(sorted((*managed_plan.versions, version), key=self._version_sort_key))
                managed_plans[logical_plan_id] = ManagedPlanRecord(
                    logical_plan_id=managed_plan.logical_plan_id,
                    display_name=managed_plan.display_name,
                    canonical_name=managed_plan.canonical_name,
                    project_code=managed_plan.project_code,
                    discipline=managed_plan.discipline,
                    notes=managed_plan.notes,
                    is_archived=managed_plan.is_archived,
                    active_version_id=managed_plan.active_version_id,
                    versions=versions,
                )

        refreshed = self._refresh_duplicate_flags(list(managed_plans.values()))
        return {
            "schema_version": self.SCHEMA_VERSION,
            "plans": [self._serialize_managed_plan(plan) for plan in refreshed],
        }

    def _serialize_managed_plan(self, plan: ManagedPlanRecord) -> dict:
        payload = asdict(plan)
        payload["versions"] = [self._serialize_version(version) for version in plan.versions]
        return payload

    def _serialize_version(self, version: PlanRecord) -> dict:
        payload = asdict(version)
        payload["index_status"] = asdict(version.index_status)
        return payload

    def _deserialize_managed_plan(self, payload: dict) -> ManagedPlanRecord:
        versions = tuple(
            self._deserialize_version(version_payload)
            for version_payload in payload.get("versions", [])
        )
        return ManagedPlanRecord(
            logical_plan_id=payload["logical_plan_id"],
            display_name=payload["display_name"],
            canonical_name=payload["canonical_name"],
            project_code=payload.get("project_code"),
            discipline=payload.get("discipline"),
            notes=payload.get("notes"),
            is_archived=payload.get("is_archived", False),
            active_version_id=payload.get("active_version_id"),
            versions=versions,
        )

    def _deserialize_version(self, payload: dict) -> PlanRecord:
        index_payload = payload.get("index_status") or {}
        return PlanRecord(
            plan_id=payload["plan_id"],
            logical_plan_id=payload.get("logical_plan_id", payload["plan_id"]),
            version_id=payload.get("version_id", payload["plan_id"]),
            filename=payload["filename"],
            file_path=payload["file_path"],
            checksum=payload["checksum"],
            added_at=payload["added_at"],
            revision_label=payload.get("revision_label"),
            duplicate_of_version_id=payload.get("duplicate_of_version_id"),
            is_current=payload.get("is_current", True),
            is_archived=payload.get("is_archived", False),
            index_status=IndexStatus(**index_payload),
        )

    def _refresh_duplicate_flags(self, plans: list[ManagedPlanRecord]) -> list[ManagedPlanRecord]:
        checksum_anchor: dict[str, str] = {}
        refreshed: list[ManagedPlanRecord] = []

        for plan in sorted(plans, key=lambda item: item.display_name.lower()):
            refreshed_versions = []
            active_version_id = plan.active_version_id
            if active_version_id is None and plan.versions:
                active_version_id = self._choose_active_version(list(plan.versions))

            for version in sorted(plan.versions, key=self._version_sort_key):
                duplicate_of = checksum_anchor.get(version.checksum)
                if duplicate_of is None:
                    checksum_anchor[version.checksum] = version.version_id
                refreshed_versions.append(
                    PlanRecord(
                        plan_id=version.plan_id,
                        logical_plan_id=plan.logical_plan_id,
                        version_id=version.version_id,
                        filename=version.filename,
                        file_path=version.file_path,
                        checksum=version.checksum,
                        added_at=version.added_at,
                        revision_label=version.revision_label,
                        duplicate_of_version_id=duplicate_of,
                        is_current=version.version_id == active_version_id,
                        is_archived=version.is_archived,
                        index_status=version.index_status,
                    )
                )

            refreshed.append(
                ManagedPlanRecord(
                    logical_plan_id=plan.logical_plan_id,
                    display_name=plan.display_name,
                    canonical_name=plan.canonical_name,
                    project_code=plan.project_code,
                    discipline=plan.discipline,
                    notes=plan.notes,
                    is_archived=plan.is_archived,
                    active_version_id=active_version_id,
                    versions=tuple(refreshed_versions),
                )
            )

        return refreshed

    def _validate_active_version(self, plan: ManagedPlanRecord, version_id: str) -> ManagedPlanRecord:
        if not any(version.version_id == version_id for version in plan.versions):
            raise ValueError(f"Version inconnue pour {plan.logical_plan_id}: {version_id}")

        return ManagedPlanRecord(
            logical_plan_id=plan.logical_plan_id,
            display_name=plan.display_name,
            canonical_name=plan.canonical_name,
            project_code=plan.project_code,
            discipline=plan.discipline,
            notes=plan.notes,
            is_archived=plan.is_archived,
            active_version_id=version_id,
            versions=plan.versions,
        )

    def _find_duplicate_version_id(
        self,
        managed_plans: list[ManagedPlanRecord],
        checksum: str,
        exclude_path: str,
    ) -> str | None:
        for managed_plan in managed_plans:
            for version in managed_plan.versions:
                if version.file_path != exclude_path and version.checksum == checksum:
                    return version.version_id
        return None

    def _reuse_existing_index_status(self, existing: PlanRecord, replacement: PlanRecord) -> PlanRecord:
        return PlanRecord(
            plan_id=replacement.plan_id,
            logical_plan_id=replacement.logical_plan_id,
            version_id=replacement.version_id,
            filename=replacement.filename,
            file_path=replacement.file_path,
            checksum=replacement.checksum,
            added_at=existing.added_at,
            revision_label=replacement.revision_label,
            duplicate_of_version_id=replacement.duplicate_of_version_id,
            is_current=replacement.is_current,
            is_archived=replacement.is_archived,
            index_status=existing.index_status,
        )

    def _existing_added_at(self, existing_item, default: str) -> str:
        if existing_item is None:
            return default
        _, version = existing_item
        return version.added_at

    def _build_logical_plan_id(self, canonical_name: str) -> str:
        return canonical_name.upper().replace(" ", "_")

    def _build_version_id(self, path: Path) -> str:
        path_hash = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
        stem = re.sub(r"[^A-Za-z0-9]+", "_", path.stem.upper()).strip("_") or "PLAN"
        return f"{stem}_{path_hash}"

    def _compute_checksum(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _extract_revision_label(self, stem: str) -> str | None:
        match = re.search(
            r"(?:^|[_\-\s])(rev(?:ision)?|v)(?:[_\-\s])*([A-Za-z0-9]+)$",
            stem,
            re.IGNORECASE,
        )
        if match:
            return match.group(2).upper()
        return None

    def _canonicalize_name(self, stem: str) -> str:
        simplified = re.sub(
            r"(?:^|[_\-\s])(rev(?:ision)?|v)(?:[_\-\s])*[A-Za-z0-9]+$",
            "",
            stem,
            flags=re.IGNORECASE,
        )
        simplified = re.sub(r"[_\-]+", " ", simplified)
        simplified = re.sub(r"\s+", " ", simplified).strip()
        return simplified or stem.strip() or "Plan"

    def _humanize_name(self, canonical_name: str) -> str:
        return canonical_name.replace("_", " ").strip().title()

    def _choose_active_version(self, versions: list[PlanRecord] | tuple[PlanRecord, ...]) -> str:
        ordered = sorted(
            versions,
            key=lambda version: (
                version.is_archived,
                version.duplicate_of_version_id is not None,
                version.added_at,
                version.filename.lower(),
            ),
            reverse=False,
        )
        return ordered[-1].version_id if ordered else ""

    def _version_sort_key(self, version: PlanRecord) -> tuple[str, str]:
        revision = version.revision_label or ""
        return (revision, version.added_at, version.filename.lower())

    def _clean_optional(self, new_value: str | None, previous: str | None) -> str | None:
        if new_value is None:
            return previous
        cleaned = new_value.strip()
        return cleaned or None
