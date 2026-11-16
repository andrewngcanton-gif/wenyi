"""Build one sanitized preparation snapshot and seed per-variant state from it.

The snapshot is constructed through public storage APIs, never by copying a live
state directory. It carries prepared chapters, analysis, glossary and cached
preparation artifacts only: no body targets, review results, usage ledger, event
log, locks or SQLite WAL sidecars.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from .common import (
    EvalError,
    PathGuard,
    UnsafeSnapshotError,
    hash_tree,
    read_json,
    sha256_file,
    sha256_text,
    utc_now,
    write_json_atomic,
)
from .manifest import variant_config

_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")
_ARTIFACT_EXACT = {"parsed_document.json", "language-policies/checkpoint.json"}
_ARTIFACT_PREFIXES = ("preparation/digests/", "preparation/analysis/", "language-policies/")
_RUNTIME_SIDECARS = re.compile(r"(?i)(?:\.lock$|-wal$|-shm$)")
_FORBIDDEN_FILES = re.compile(
    r"(?i)(?:\.lock$|-wal$|-shm$|(?:^|[\\/])(?:usage\.json|events\.jsonl)$|(?:^|[\\/])reviews(?:[\\/]|$))"
)


def write_prepare_config(
    root: Path,
    *,
    group: str = "A_prime",
    base_config: dict[str, Any] | None = None,
    reflect_model: str | None = None,
) -> Path:
    guard = PathGuard(root)
    config = variant_config(
        group,
        guard.child("_prepared", "staging"),
        base=base_config,
        reflect_model=reflect_model,
    )
    path = guard.child("config", "prepare.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def write_group_prepare_config(
    root: Path,
    group: str,
    *,
    base_config: dict[str, Any] | None = None,
    reflect_model: str | None = None,
) -> Path:
    """Write the isolated preparation configuration for one incompatible group."""
    guard = PathGuard(root)
    config = variant_config(
        group,
        guard.child(group, "_prepared", "staging"),
        base=base_config,
        reflect_model=reflect_model,
    )
    path = guard.child(group, "_prepared", "prepare.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def find_run_dir(state_root: Path) -> Path:
    matches = sorted(path.parent for path in state_root.rglob("manifest.json") if path.is_file())
    if len(matches) != 1:
        raise EvalError(
            f"Expected exactly one prepared run under {state_root}, found {len(matches)}"
        )
    return matches[0]


def sanitize_value(value: Any) -> Any:
    """Replace absolute paths in state metadata without touching ordinary prose."""
    if isinstance(value, dict):
        return {str(key): sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str) and _ABSOLUTE.match(value.strip()):
        return "<stripped-absolute-path>"
    return value


def absolute_path_strings(value: Any, location: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(absolute_path_strings(item, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(absolute_path_strings(item, f"{location}[{index}]"))
    elif isinstance(value, str) and _ABSOLUTE.match(value.strip()):
        found.append(location)
    return found


def _copy_glossary(source_path: Path, destination_path: Path) -> int:
    if not source_path.is_file():
        return 0
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{source_path.as_posix()}?mode=ro", uri=True)
    destination = sqlite3.connect(str(destination_path))
    try:
        source.backup(destination)
        destination.commit()
        row = destination.execute("SELECT COUNT(*) FROM glossary").fetchone()
        return int(row[0]) if row else 0
    finally:
        destination.close()
        source.close()


def create_snapshot(
    *,
    root: Path,
    run_dir: Path,
    expected_source_sha256: str,
    expected_target_lang: str,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Write a fresh prepared tree under ``_prepared/state`` and return its receipt."""
    from wenyi_core.models.document import Chapter
    from wenyi_core.storage.file import FileStorage

    guard = PathGuard(root)
    snapshot_root = guard.ensure(snapshot_dir) if snapshot_dir else guard.child("_prepared")
    snapshot_state = snapshot_root / "state"
    metadata_path = snapshot_root / "snapshot.json"
    if snapshot_state.exists() or metadata_path.exists():
        raise EvalError(f"Refusing to overwrite an existing preparation snapshot: {snapshot_root}")

    source = FileStorage(str(run_dir), create=False)
    try:
        manifest = source.load_manifest()
        if manifest.get("source_sha256") != expected_source_sha256:
            raise UnsafeSnapshotError("Prepared manifest source hash does not match the fixture")
        if manifest.get("target_lang") != expected_target_lang:
            raise UnsafeSnapshotError("Prepared manifest target language is incompatible")
        if not manifest.get("initialized"):
            raise UnsafeSnapshotError("Prepared manifest is not committed as initialized")
        chapters: list[Chapter] = []
        for entry in manifest.get("chapters", []):
            if entry.get("status") != "pending":
                raise UnsafeSnapshotError(
                    f"Prepared chapter {entry.get('index')} is not pending: {entry.get('status')}"
                )
            chapter = Chapter.model_validate(source.load_chapter(entry["index"]).model_dump())
            for segment in chapter.segments:
                if segment.target is not None or segment.target_before_polish is not None:
                    raise UnsafeSnapshotError(
                        f"Prepared chapter {chapter.index} segment {segment.index} has a body target"
                    )
            chapters.append(chapter)
        analysis = source.load_analysis()
        contexts = source.load_annotation_contexts()
        artifacts = _allowed_artifacts(source, run_dir)
        artifact_values = {key: sanitize_value(source.read_artifact(key)) for key in artifacts}
        term_count = len(source.all_terms())
        glossary_source = Path(source.glossary_path)
    finally:
        source.close()

    manifest = sanitize_value(manifest)
    analysis = sanitize_value(analysis)
    contexts = sanitize_value(contexts)
    for chapter in chapters:
        _sanitize_chapter(chapter)

    destination = FileStorage(str(snapshot_state), create=True)
    try:
        for chapter in chapters:
            destination.save_chapter(chapter)
        if isinstance(analysis, dict) and analysis:
            destination.save_analysis(analysis)
        if isinstance(contexts, dict) and contexts:
            destination.save_annotation_contexts(contexts)
        for key, value in artifact_values.items():
            destination.write_artifact(key, value)
        copied_terms = _copy_glossary(glossary_source, Path(destination.glossary_path))
        # Manifest last: it is the initialization marker.
        destination.save_manifest(manifest)
    finally:
        destination.close()

    _prune_runtime_files(snapshot_state)
    _validate_snapshot_tree(snapshot_state, manifest, chapters)
    digest = hash_tree(snapshot_state)
    receipt = {
        "created_at": utc_now(),
        "source_sha256": expected_source_sha256,
        "target_lang": expected_target_lang,
        "tree_sha256": digest,
        "chapters": len(chapters),
        "segments": sum(len(chapter.segments) for chapter in chapters),
        "terms": copied_terms if term_count else 0,
        "artifacts": sorted(artifact_values),
        "glossary_consistent_backup": True,
        "body_targets": 0,
    }
    write_json_atomic(metadata_path, receipt)
    return receipt


def _allowed_artifacts(source, run_dir: Path) -> list[str]:
    allowed: list[str] = []
    for key in source.list_artifacts(""):
        if key in _ARTIFACT_EXACT or key.startswith(_ARTIFACT_PREFIXES):
            if _FORBIDDEN_FILES.search(key.replace("/", os.sep)):
                raise UnsafeSnapshotError(f"Preparation artifact key looks unsafe: {key}")
            path = (run_dir / key).resolve()
            if not path.is_relative_to(run_dir.resolve()):
                raise UnsafeSnapshotError(f"Preparation artifact escapes the run: {key}")
            allowed.append(key)
    return sorted(allowed)


def _sanitize_chapter(chapter) -> None:
    """Drop any prepared-only path fields while preserving stable segment identity."""
    chapter.meta = sanitize_value(chapter.meta)
    chapter.href = sanitize_value(chapter.href)
    for segment in chapter.segments:
        segment.meta = sanitize_value(segment.meta)
        segment.resource_href = sanitize_value(segment.resource_href)
        segment.anchor = sanitize_value(segment.anchor)


def _prune_runtime_files(state: Path) -> None:
    """Drop lock/WAL sidecars created by the fresh storage used to build the snapshot."""
    for path in sorted(state.rglob("*")):
        if path.is_file() and _RUNTIME_SIDECARS.search(path.relative_to(state).as_posix()):
            path.unlink()


def _validate_snapshot_tree(snapshot_state: Path, manifest: dict, chapters: list) -> None:
    files = [path for path in snapshot_state.rglob("*") if path.is_file()]
    for path in files:
        relative = path.relative_to(snapshot_state).as_posix()
        if _FORBIDDEN_FILES.search(relative):
            raise UnsafeSnapshotError(f"Snapshot contains forbidden state: {relative}")
        if path.is_symlink():
            raise UnsafeSnapshotError(f"Snapshot contains a symlink: {relative}")
    if not (snapshot_state / "manifest.json").is_file():
        raise UnsafeSnapshotError("Snapshot manifest was not committed")
    leaks = absolute_path_strings(manifest, "$.manifest")
    for chapter in chapters:
        leaks.extend(
            absolute_path_strings(chapter.model_dump(mode="json"), f"$.chapters[{chapter.index}]")
        )
    if leaks:
        raise UnsafeSnapshotError("Snapshot retains absolute paths: " + ", ".join(leaks[:5]))
    for chapter in chapters:
        for segment in chapter.segments:
            if segment.target is not None or segment.target_before_polish is not None:
                raise UnsafeSnapshotError("Snapshot chapter contains a body target")
    for path in snapshot_state.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UnsafeSnapshotError(f"Snapshot JSON is unreadable: {path}") from error
        leaks = absolute_path_strings(data, path.name)
        if leaks:
            raise UnsafeSnapshotError(
                f"Snapshot JSON {path.name} retains absolute paths: {', '.join(leaks[:5])}"
            )


def seed_state(snapshot_state: Path, destination: Path, *, expected_digest: str) -> None:
    """Copy a validated snapshot into one variant's empty state directory."""
    import shutil

    if destination.exists() and any(destination.iterdir()):
        raise EvalError(f"Refusing to seed a non-empty variant state: {destination}")
    if hash_tree(snapshot_state) != expected_digest:
        raise EvalError("Preparation snapshot changed before seeding")
    if destination.exists():
        destination.rmdir()
    shutil.copytree(snapshot_state, destination, symlinks=False)
    if hash_tree(destination) != expected_digest:
        raise EvalError(f"Seeded state does not match the snapshot digest: {destination}")


def read_snapshot_receipt(root: Path) -> dict[str, Any]:
    guard = PathGuard(root)
    return read_json(guard.child("_prepared", "snapshot.json"))


def read_snapshot_receipt_at(snapshot_dir: Path) -> dict[str, Any]:
    return read_json(Path(snapshot_dir) / "snapshot.json")


def snapshot_source_digest(state: Path) -> str:
    return sha256_file(state / "manifest.json")


class PreparationIncompatibilityError(UnsafeSnapshotError):
    """A prepared snapshot cannot be reused by a group's binding without reparation."""


def preparation_identity(state: Path) -> dict[str, Any]:
    """Read the persisted ingestion, preparation and prescan identity of a snapshot.

    Every field comes from a product-owned persisted contract: the parsed
    document's ``ingest_config``, the analysis ``style_policy``, the language
    policy checkpoint and each chapter's digest policy metadata.
    """
    from wenyi_core.storage.file import FileStorage

    before = hash_tree(state)
    store = FileStorage(str(state), create=False)
    try:
        manifest = store.load_manifest()
        parsed = store.read_artifact("parsed_document.json")
        analysis = store.load_analysis() or {}
        checkpoint = store.read_artifact("language-policies/checkpoint.json")
        digest_policies = []
        for entry in manifest.get("chapters", []):
            chapter = store.load_chapter(entry["index"])
            policy = chapter.meta.get("source_digest_policy")
            if isinstance(policy, str) and policy:
                digest_policies.append(policy)
    finally:
        store.close()
    after = hash_tree(state)
    if before != after:
        raise UnsafeSnapshotError("Reading the preparation identity changed the snapshot")
    identity = {
        "source_sha256": manifest.get("source_sha256"),
        "source_lang": manifest.get("source_lang"),
        "target_lang": manifest.get("target_lang"),
        "ingest_config": parsed.get("ingest_config") if isinstance(parsed, dict) else None,
        "analysis_style_policy": analysis.get("style_policy"),
        "language_policies": checkpoint if isinstance(checkpoint, dict) else None,
        "chapter_digest_policies": sorted(set(digest_policies)),
        "chapters": len(manifest.get("chapters", [])),
    }
    identity["fingerprint"] = sha256_text(json.dumps(identity, sort_keys=True))
    return identity


def expected_preparation_identity(
    config_path: str | os.PathLike[str], *, source_sha256: str, source_lang: str
) -> dict[str, Any]:
    """Describe what a group's configuration expects from a reusable snapshot."""
    import copy

    from wenyi_core.config_rebuild import load_rebuilt_config
    from wenyi_core.i18n.policy.models import content_hash
    from wenyi_core.pipeline.prepare.input_preparation import ingest_config

    config = load_rebuilt_config(config_path)
    ingestion = ingest_config(config)
    resolved = copy.deepcopy(config)
    resolved.source_lang = source_lang
    resolved.freeze_language_policies(source_sha256)
    plans = {phase: resolved.language_policy(phase) for phase in ("analysis", "translation")}
    digest_policy = content_hash(
        [
            resolved.language_policy("analysis").task_fingerprint("chapter_digest"),
            resolved.llm.model_dump(mode="json"),
        ]
    )
    return {
        "source_sha256": source_sha256,
        "target_lang": resolved.target_lang,
        "ingest_config": ingestion,
        "analysis_style_policy": plans["analysis"].task_fingerprint("analyzer"),
        "language_policies": {
            phase: f"language-policies/{plan.fingerprint}.json" for phase, plan in plans.items()
        },
        "chapter_digest_policy": digest_policy,
        "book_understanding": bool(resolved.pipeline.book_understanding),
    }


def compare_preparation_identity(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Return every reason a group may not reuse the shared prepared snapshot."""
    reasons: list[str] = []
    if actual.get("source_sha256") != expected["source_sha256"]:
        reasons.append("source identity differs")
    if actual.get("target_lang") != expected["target_lang"]:
        reasons.append("target language differs")
    if actual.get("ingest_config") != expected["ingest_config"]:
        reasons.append("ingestion settings differ")
    if actual.get("analysis_style_policy") != expected["analysis_style_policy"]:
        reasons.append("analysis policy differs or was not recorded")
    actual_policies = actual.get("language_policies") or {}
    analysis_reference = expected["language_policies"].get("analysis")
    if actual_policies.get("analysis") != analysis_reference:
        reasons.append("analysis language policy differs")
    actual_digests = actual.get("chapter_digest_policies") or []
    if expected["book_understanding"]:
        if not actual_digests:
            reasons.append("prescan chapter digest policy was not recorded")
        elif any(policy != expected["chapter_digest_policy"] for policy in actual_digests):
            reasons.append("prescan chapter digest policy differs")
    return reasons


def expected_reflection_identity(
    config_path: str | os.PathLike[str],
    *,
    source_sha256: str,
    source_lang: str,
) -> dict[str, Any]:
    """Reconstruct the reflection policy and configured route evidence.

    Everything here is derived from the group configuration and the product's
    route resolver, never from the artifact being validated. The artifact reader
    then compares its stored hashes against this independently produced identity.
    """
    from wenyi_core.config_rebuild import load_rebuilt_config
    from wenyi_core.llm.routing import resolve_routes
    from wenyi_core.pipeline.translate.reflection_records import CODE_IDENTITY

    config = load_rebuilt_config(config_path)
    config.source_lang = source_lang
    config.freeze_language_policies(source_sha256)
    plan = config.language_policy("translation")
    pipeline = config.pipeline
    routes = resolve_routes(config.llm)
    operations = ("reflect.body", "polish.body", "translation.body")
    return {
        "policy": {
            "code_identity": CODE_IDENTITY,
            "reflective_polish": True,
            "min_severity": pipeline.reflect_min_severity,
            "clean_action": pipeline.reflect_clean_batches,
            "reflector_fingerprint": plan.task_fingerprint("reflector"),
            "translation_fingerprint": plan.fingerprint,
        },
        "configured_route_fingerprints": {
            operation: routes[operation].fingerprint for operation in operations
        },
        "configured_routes": {
            operation: {
                "profile": routes[operation].profile,
                "model": routes[operation].model,
                "provider": routes[operation].provider,
                "origin": routes[operation].origin,
                "inference_fingerprint": routes[operation].fingerprint,
            }
            for operation in operations
        },
    }
