from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_annotations import _extraction, _manifest, _read, _strata, _write
from wenyi_eval import annotation_import
from wenyi_eval.annotation_import import import_annotations, verify_imports
from wenyi_eval.edit_review import export_edit_review
from wenyi_eval.pairwise import RatingError, export_pairwise


def _packet(root: Path) -> tuple[dict, Path, Path]:
    _strata(root)
    extraction = _extraction()
    extraction["expected"] = {"source_sha256": "a" * 64}
    export_pairwise(root=root, manifest=_manifest(), extraction=extraction)
    export_edit_review(root=root, manifest=_manifest(), extraction=extraction)
    return extraction, root / "samples" / "pairwise.csv", root / "samples" / "edit_review.csv"


def _saved(root: Path) -> dict[str, bytes]:
    return {
        name: (root / "samples" / name).read_bytes()
        for name in (
            "pairwise_ratings.jsonl",
            "pairwise_import.json",
            "edit_ratings.jsonl",
            "edit_import.json",
            "annotation_import.json",
        )
    }


def _change_preference(pair: Path) -> None:
    rows = _read(pair)
    rows[0]["preference"] = "left"
    _write(pair, rows, list(rows[0]))


def test_invalid_second_table_leaves_entire_previous_generation_unchanged(tmp_path: Path) -> None:
    extraction, pair, edit = _packet(tmp_path)
    import_annotations(tmp_path, pair, edit, extraction=extraction)
    previous = _saved(tmp_path)
    _change_preference(pair)
    rows = _read(edit)
    rows[0]["worsened"] = "invalid"
    _write(edit, rows, list(rows[0]))

    with pytest.raises(RatingError, match="worsened"):
        import_annotations(tmp_path, pair, edit, extraction=extraction)

    assert _saved(tmp_path) == previous
    assert verify_imports(tmp_path, extraction)["status"] == "valid"


def test_publish_failure_restores_previous_files_and_receipt(tmp_path: Path, monkeypatch) -> None:
    extraction, pair, edit = _packet(tmp_path)
    import_annotations(tmp_path, pair, edit, extraction=extraction)
    previous = _saved(tmp_path)
    _change_preference(pair)
    real_replace = annotation_import._replace_bytes
    calls = 0

    def fail_second(path, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        real_replace(path, data)

    monkeypatch.setattr(annotation_import, "_replace_bytes", fail_second)
    with pytest.raises(RatingError, match="publication failed"):
        import_annotations(tmp_path, pair, edit, extraction=extraction)

    assert _saved(tmp_path) == previous
    assert not (tmp_path / "samples" / "annotation_import.pending.json").exists()
    assert verify_imports(tmp_path, extraction)["status"] == "valid"


def test_interrupted_or_torn_generation_is_rejected(tmp_path: Path) -> None:
    extraction, pair, edit = _packet(tmp_path)
    import_annotations(tmp_path, pair, edit, extraction=extraction)
    pending = tmp_path / "samples" / "annotation_import.pending.json"
    pending.write_text('{"generation":"interrupted"}', encoding="utf-8")
    assert verify_imports(tmp_path, extraction)["status"] == "invalid"
    pending.unlink()
    (tmp_path / "samples" / "edit_ratings.jsonl").write_text("{}\n", encoding="utf-8")
    assert verify_imports(tmp_path, extraction)["status"] == "invalid"


@pytest.mark.parametrize("field", ["target", "target_before_polish", "source"])
def test_saved_text_changes_invalidate_imports(tmp_path: Path, field: str) -> None:
    extraction, pair, edit = _packet(tmp_path)
    import_annotations(tmp_path, pair, edit, extraction=extraction)
    assert verify_imports(tmp_path, extraction)["status"] == "valid"
    extraction["groups"]["C"]["chapters"][0]["segments"][0][field] += " changed"
    assert verify_imports(tmp_path, extraction)["status"] == "invalid"


def test_reexport_and_new_source_inventory_invalidate_imports(tmp_path: Path) -> None:
    extraction, pair, edit = _packet(tmp_path)
    import_annotations(tmp_path, pair, edit, extraction=extraction)
    extraction["expected"]["source_sha256"] = "b" * 64
    assert verify_imports(tmp_path, extraction)["status"] == "invalid"
    extraction["expected"]["source_sha256"] = "a" * 64
    key = tmp_path / "samples" / "pairwise_key.csv"
    key.write_bytes(key.read_bytes() + b"\n")
    assert verify_imports(tmp_path, extraction)["status"] == "invalid"


@pytest.mark.parametrize("receipt", [{}, {"generation": "new", "files": {}}, {"files": {}}])
def test_empty_receipt_cannot_validate_an_import(tmp_path: Path, receipt: dict) -> None:
    extraction, pair, edit = _packet(tmp_path)
    import_annotations(tmp_path, pair, edit, extraction=extraction)
    (tmp_path / "samples" / "annotation_import.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    assert verify_imports(tmp_path, extraction)["status"] == "invalid"
