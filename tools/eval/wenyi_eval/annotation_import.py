"""Publish one coherent annotation import and verify its persisted evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from .common import EvalError, PathGuard, read_json, read_jsonl, sha256_file, write_json_atomic
from .pairwise import RatingError, _read_csv, _side_hash, _target_index, verify_output_identity


def _replace_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_imports(root: Path, payloads: list[dict[str, Any]]) -> None:
    """Validate before this call; a pending marker rejects interrupted generations."""
    guard = PathGuard(root)
    directory = guard.child("samples")
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / "annotation_import.pending.json"
    receipt = directory / "annotation_import.json"
    files: dict[Path, bytes] = {}
    for payload in payloads:
        ratings = guard.ensure(payload["ratings_path"], label="ratings")
        summary = guard.ensure(payload["summary_path"], label="import summary")
        files[ratings] = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payload["rows"]
        ).encode("utf-8")
        files[summary] = (
            json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
    backups = {path: path.read_bytes() if path.exists() else None for path in [*files, receipt]}
    already_pending = pending.exists()
    generation = uuid4().hex
    write_json_atomic(pending, {"generation": generation})
    try:
        for path, data in files.items():
            _replace_bytes(path, data)
        write_json_atomic(
            receipt,
            {"generation": generation, "files": {p.name: sha256_file(p) for p in files}},
        )
        pending.unlink()
    except BaseException:
        restored = True
        for path, data in backups.items():
            try:
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    _replace_bytes(path, data)
            except OSError:
                restored = False
        if restored and not already_pending:
            pending.unlink(missing_ok=True)
        raise


def import_annotations(
    root: Path, ratings: Path, edit_ratings: Path | None, *, extraction: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from .edit_review import _prepare_edit_import
    from .pairwise import _prepare_pairwise_import

    payloads = [_prepare_pairwise_import(root, ratings, extraction=extraction)]
    if edit_ratings is not None:
        payloads.append(_prepare_edit_import(root, edit_ratings, extraction=extraction))
    try:
        publish_imports(root, payloads)
    except OSError as error:
        raise RatingError(f"Annotation publication failed: {type(error).__name__}") from error
    return payloads[0]["summary"], payloads[1]["summary"] if len(payloads) > 1 else None


def verify_imports(root: Path, extraction: dict[str, Any]) -> dict[str, Any]:
    directory = Path(root) / "samples"
    if (directory / "annotation_import.pending.json").exists():
        return {"status": "invalid", "reason": "Annotation publication was interrupted."}
    receipt_path = directory / "annotation_import.json"
    if not receipt_path.is_file():
        return {"status": "missing", "reason": "No coherent annotation import is recorded."}
    try:
        receipt = read_json(receipt_path)
        if (
            not isinstance(receipt, dict)
            or not isinstance(receipt.get("files"), dict)
            or not isinstance(receipt.get("generation"), str)
            or not receipt["generation"]
            or not {"pairwise_ratings.jsonl", "pairwise_import.json"} <= receipt["files"].keys()
        ):
            raise ValueError("Invalid annotation import receipt")
        for name, digest in receipt["files"].items():
            if (
                name
                not in {
                    "pairwise_ratings.jsonl",
                    "pairwise_import.json",
                    "edit_ratings.jsonl",
                    "edit_import.json",
                }
                or sha256_file(directory / name) != digest
            ):
                raise ValueError(
                    "Annotation import files changed or belong to different generations"
                )
        targets = _target_index(extraction)
        source_sha = (extraction.get("expected") or {}).get("source_sha256")
        for kind, id_field in (("pairwise", "pair_id"), ("edit", "edit_id")):
            summary_path = directory / f"{kind}_import.json"
            if not summary_path.is_file():
                continue
            required_files = {f"{kind}_import.json", f"{kind}_ratings.jsonl"}
            if not required_files <= receipt["files"].keys():
                raise ValueError("Annotation import contains an uncommitted packet")
            summary = read_json(summary_path)
            for label, name in (
                ("key_sha256", f"{kind}_key.csv"),
                ("preregistration_sha256", f"{kind}_preregistration.json"),
                ("ratings_sha256", f"{kind}_ratings.jsonl"),
            ):
                if not summary.get(label) or sha256_file(directory / name) != summary[label]:
                    raise ValueError(f"{kind} annotation packet or ratings binding changed")
            key_rows = _read_csv(directory / f"{kind}_key.csv", (id_field,), f"{kind} key")
            keys = {row[id_field]: row for row in key_rows}
            if len(keys) != len(key_rows):
                raise ValueError("Duplicate annotation keys")
            seen = set()
            for row in read_jsonl(directory / f"{kind}_ratings.jsonl"):
                identity = row.get(id_field)
                if identity in seen or identity not in keys:
                    raise ValueError("Duplicate or missing annotation identity")
                seen.add(identity)
                key = keys[identity]
                fields = ["source_sha256", "source_text_sha256"] + (
                    ["left_sha256", "right_sha256"]
                    if kind == "pairwise"
                    else ["before_sha256", "after_sha256"]
                )
                if any(not key.get(f) or row.get(f) != key[f] for f in fields):
                    raise ValueError("Annotation no longer matches its text identity")
                if not source_sha or source_sha != key["source_sha256"]:
                    raise ValueError("Annotation source inventory changed or is unavailable")
                if kind == "pairwise":
                    stale = verify_output_identity(key_row=key, targets=targets)
                    if stale:
                        raise ValueError(stale)
                    sides = [(key["left_group"], "left"), (key["right_group"], "right")]
                else:
                    sides = [(key["group"], "after")]
                for group, side in sides:
                    segment = targets.get(f"{group}|{key['segment_id']}")
                    if not segment or segment.get("status") != "translated":
                        raise ValueError("Current annotation segment is missing or stale")
                    if _side_hash(segment["source"]) != key["source_text_sha256"]:
                        raise ValueError("Annotation segment source changed")
                    if kind == "edit" and (
                        not isinstance(segment.get("target_before_polish"), str)
                        or _side_hash(segment["target_before_polish"]) != key["before_sha256"]
                        or _side_hash(segment["target"]) != key["after_sha256"]
                    ):
                        raise ValueError("Edited draft or translation changed after annotation")
        return {
            "status": "valid",
            "reason": "Annotation generation and current text bindings match.",
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EvalError) as error:
        return {"status": "invalid", "reason": str(error)}
