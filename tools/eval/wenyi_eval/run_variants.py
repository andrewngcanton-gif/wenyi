"""Plan, run, extract, export and report Wenyi variant evaluations.

Every command runs as a real subprocess through the declared executable and
argv. Provider mode imports the selected production factory without any
evaluation monkeypatch; the offline adapter is used only when explicitly
selected. Terminal records preserve the command, cwd, binding and captured logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .common import (
    CorruptIdentityError,
    EvalError,
    OwnershipError,
    PathGuard,
    bounded_text,
    hash_tree,
    new_eval_id,
    ownership_lock,
    read_json,
    resolve_absolute,
    sha256_file,
    sha256_text,
    subprocess_env,
    utc_now,
    write_json_atomic,
)
from .manifest import (
    SUPPORTED_GROUPS,
    credential_env_names,
    load_group_template,
    load_manifest,
    preparation_group,
    require_frozen_bindings,
    require_real_ready,
    validate_manifest,
    write_manifest,
    write_variant_configs,
)
from .snapshot import (
    PreparationIncompatibilityError,
    compare_preparation_identity,
    create_snapshot,
    expected_preparation_identity,
    find_run_dir,
    preparation_identity,
    read_snapshot_receipt_at,
    seed_state,
    write_group_prepare_config,
    write_prepare_config,
)

TERMINAL_PROTOCOL = 1
GROUP_STAGES = ("translate", "status", "inspect")
CAPTURE_LIMIT = 2 * 1024 * 1024


@dataclass(frozen=True)
class ChildOutcome:
    command: list[str]
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    binding: dict[str, Any] | None
    binding_error: str | None

    @property
    def binding_valid(self) -> bool:
        return self.binding is not None and self.binding_error is None


class TerminalStore:
    """One exclusive JSON file per (group, stage); never overwrites a record."""

    def __init__(self, root: Path):
        self.root = root
        self.directory = root / "records" / "terminals"
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, group: str, stage: str) -> Path:
        return self.directory / f"{group}.{stage}.json"

    def write(self, terminal: dict[str, Any]) -> Path:
        group = terminal["group"]
        stage = terminal["stage"]
        path = self.path(group, stage)
        payload = json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise OwnershipError(f"Terminal record already exists: {path}") from None
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        return path

    def read_all(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.directory.glob("*.json")):
            row = read_json(path)
            if row.get("protocol_version") != TERMINAL_PROTOCOL:
                raise CorruptIdentityError(f"Unsupported terminal protocol: {path.name}")
            if path.stem != f"{row.get('group')}.{row.get('stage')}":
                raise CorruptIdentityError(
                    f"Terminal filename does not match its payload: {path.name}"
                )
            rows.append(row)
        return rows

    def validate_complete(
        self,
        groups: Iterable[str] | None = None,
        isolated_preparations: Iterable[str] = (),
    ) -> dict[tuple[str, str], dict[str, Any]]:
        rows = self.read_all()
        selected = tuple(groups) if groups is not None else SUPPORTED_GROUPS
        expected = {("preparation", "prepare")}
        expected.update(("preparation", f"prepare.{group}") for group in isolated_preparations)
        expected.update((group, stage) for group in selected for stage in GROUP_STAGES)
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["group"], row["stage"])
            if key in seen:
                raise OwnershipError(f"Duplicate terminal record: {key}")
            seen[key] = row
        missing = expected - set(seen)
        if missing:
            raise EvalError(f"Missing terminal records: {sorted(missing)}")
        extra = set(seen) - expected
        if extra:
            raise EvalError(f"Unexpected terminal records: {sorted(extra)}")
        return seen


def initialize_root(runs_dir: str | os.PathLike[str], *, eval_id: str, book: str) -> Path:
    base = Path(runs_dir)
    if not base.is_absolute():
        raise EvalError("--runs-dir must be absolute")
    for part in (eval_id, book):
        if not part or part in {".", ".."} or any(ch in part for ch in '<>:"/\\|?*'):
            raise EvalError(f"Unsafe evaluation path component: {part!r}")
    root = (base / eval_id / book).resolve()
    if root.exists() and any(root.iterdir()):
        raise EvalError(f"Refusing to overwrite an existing evaluation: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_binding(
    binding_path: Path, code_source: str
) -> tuple[dict[str, Any] | None, str | None]:
    if not binding_path.is_file():
        return None, f"missing binding record: {binding_path}"
    try:
        binding = read_json(binding_path)
    except (OSError, json.JSONDecodeError) as error:
        return None, f"unreadable binding record: {error}"
    source = Path(code_source).resolve()
    expected_core = source / "packages" / "core"
    expected_cli = source / "packages" / "cli"
    if binding.get("code_source") != str(source):
        return binding, "binding code source differs from the manifest"
    for field, parent in (
        ("core_file", expected_core),
        ("cli_file", expected_cli),
        ("factory_file", expected_core),
    ):
        value = binding.get(field)
        if not isinstance(value, str) or not Path(value).is_file():
            return binding, f"binding {field} is missing"
        try:
            Path(value).resolve().relative_to(parent)
        except ValueError:
            return binding, f"binding {field} is outside the selected code source"
    return binding, None


def _write_log(path: Path, data: bytes) -> tuple[str, bool]:
    text, truncated = bounded_text(data, CAPTURE_LIMIT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_file(path), truncated


def _provider_binding(
    *, plan: dict[str, Any], binding_path: Path, cwd: Path, env: dict[str, str]
) -> None:
    """Import the selected product modules in the same interpreter without patches."""
    code = (
        "import json, os, sys\n"
        "import wenyi_core\n"
        "import wenyi_cli\n"
        "import wenyi_core.llm.factory as factory\n"
        "payload = {\n"
        "    'adapter': 'provider',\n"
        "    'code_source': sys.argv[1],\n"
        "    'core_file': wenyi_core.__file__,\n"
        "    'cli_file': wenyi_cli.__file__,\n"
        "    'factory_file': factory.__file__,\n"
        "    'cwd': os.getcwd(),\n"
        "    'python': sys.version.split()[0],\n"
        "}\n"
        "with open(sys.argv[2], 'w', encoding='utf-8') as handle:\n"
        "    json.dump(payload, handle, sort_keys=True)\n"
    )
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.unlink(missing_ok=True)
    completed = subprocess.run(
        [plan["executable"], "-c", code, plan["code_source"], str(binding_path)],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise EvalError(f"Could not bind the provider interpreter: {detail}")


def _child_command(plan: dict[str, Any], config: str, argv: list[str]) -> list[str]:
    """Build the command that is actually executed for one group."""
    return [plan["executable"], *plan["argv_prefix"], "--config", config, *argv]


def run_child(
    *,
    group: str,
    stage: str,
    plan: dict[str, Any],
    config: str,
    argv: list[str],
    cwd: Path,
    binding_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
) -> tuple[ChildOutcome, dict[str, Any]]:
    """Execute one CLI command and return its outcome and captured-log metadata."""
    cwd.mkdir(parents=True, exist_ok=True)
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    binding_path.unlink(missing_ok=True)
    adapter = plan.get("adapter", "offline")
    command = _child_command(plan, config, argv)
    env = subprocess_env(
        offline=adapter == "offline",
        passthrough=credential_env_names(config) if adapter == "provider" else (),
        pythonpath=(
            [
                str(Path(plan["code_source"]) / "packages" / "core"),
                str(Path(plan["code_source"]) / "packages" / "cli"),
            ]
            if adapter == "provider"
            else ()
        ),
    )
    env["WENYI_EVAL_BINDING_PATH"] = str(binding_path)
    started = utc_now()
    clock = time.monotonic()
    timed_out = False
    exit_code: int | None = None
    stdout = b""
    stderr = b""
    binding_error: str | None = None
    try:
        if adapter == "provider":
            _provider_binding(plan=plan, binding_path=binding_path, cwd=cwd, env=env)
    except (EvalError, OSError, subprocess.TimeoutExpired) as error:
        binding_error = f"binding probe failed: {error}"
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        stdout, stderr = completed.stdout, completed.stderr
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout if isinstance(error.stdout, bytes) else b""
        stderr = error.stderr if isinstance(error.stderr, bytes) else b""
    except OSError as error:
        stderr = f"could not start child process: {error}".encode("utf-8", errors="replace")
    duration = round(time.monotonic() - clock, 3)
    stdout_sha, stdout_truncated = _write_log(stdout_path, stdout)
    stderr_sha, stderr_truncated = _write_log(stderr_path, stderr)
    binding, validated_error = _validate_binding(binding_path, plan["code_source"])
    binding_error = binding_error or validated_error
    if timed_out:
        status = "timed_out"
    elif binding_error:
        status = "binding_invalid"
    elif exit_code == 0:
        status = "succeeded"
    else:
        status = "failed"
    terminal = {
        "protocol_version": TERMINAL_PROTOCOL,
        "group": group,
        "stage": stage,
        "status": status,
        "started_at": started,
        "finished_at": utc_now(),
        "duration_seconds": duration,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "adapter": adapter,
        "command": command,
        "cwd": str(cwd),
        "code_source": plan["code_source"],
        "binding_path": str(binding_path),
        "binding": binding,
        "binding_error": binding_error,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": stdout_sha,
        "stderr_sha256": stderr_sha,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
    logs = {
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": stdout_sha,
        "stderr_sha256": stderr_sha,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
    return ChildOutcome(command, exit_code, timed_out, duration, binding, binding_error), {
        "terminal": terminal,
        "logs": logs,
    }


def _usage_from_state(run_dir: Path) -> dict[str, Any] | None:
    from wenyi_core.storage.file import FileStorage

    store = FileStorage(str(run_dir), create=False)
    try:
        return store.load_usage()
    finally:
        store.close()


def _prepare_once(
    *,
    root: Path,
    manifest: dict[str, Any],
    source: Path,
    terminals: TerminalStore,
    timeout: float,
    plan: dict[str, Any],
    config: Path,
    stage: str,
    prepared_root: Path,
    record_path: Path,
) -> dict[str, Any]:
    """Run one preparation subprocess and build its sanitized snapshot."""
    guard = PathGuard(root)
    prepared_root = guard.ensure(prepared_root)
    binding_path = prepared_root / "binding.prepare.json"
    stdout_path = prepared_root / "logs" / "prepare.stdout.log"
    stderr_path = prepared_root / "logs" / "prepare.stderr.log"
    source_hash = sha256_file(source)
    outcome, logs = run_child(
        group="preparation",
        stage=stage,
        plan=plan,
        config=str(config),
        argv=["prepare", str(source)],
        cwd=prepared_root / "cwd",
        binding_path=binding_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=timeout,
    )
    terminal_path = terminals.write(logs["terminal"])
    if outcome.exit_code != 0 or not outcome.binding_valid:
        raise EvalError(f"Preparation failed ({logs['terminal']['status']}); see {stderr_path}")
    staging = prepared_root / "staging"
    run_dir = find_run_dir(staging)
    try:
        run_relative_path = run_dir.relative_to(staging).as_posix()
    except ValueError:
        run_relative_path = ""
    if not run_relative_path:
        raise EvalError("Prepared run directory is not inside the staging state")
    usage = _usage_from_state(run_dir)
    snapshot = create_snapshot(
        root=root,
        run_dir=run_dir,
        expected_source_sha256=source_hash,
        expected_target_lang=manifest["source"]["language"]["target"],
        snapshot_dir=prepared_root,
    )
    record = {
        "protocol_version": TERMINAL_PROTOCOL,
        "stage": "preparation",
        "terminal_stage": stage,
        "terminal_path": str(terminal_path),
        "terminal_status": logs["terminal"]["status"],
        "adapter": plan["adapter"],
        "executable": plan["executable"],
        "argv_prefix": plan["argv_prefix"],
        "group": plan.get("group_key"),
        "source_sha256": source_hash,
        "source_unchanged": sha256_file(source) == source_hash,
        "run_dir": str(run_dir),
        "run_relative_path": run_relative_path,
        "snapshot": snapshot,
        "snapshot_dir": str(prepared_root),
        "usage": usage,
        "usage_source": (
            "synthetic_offline_adapter; not provider metering"
            if plan["adapter"] == "offline"
            else "provider_ledger; priced only with declared rules"
        ),
        "cost": {
            "status": "unknown",
            "amount": None,
            "reason": (
                "offline adapter; no provider price rule applies"
                if plan["adapter"] == "offline"
                else "preparation cost is unknown until a declared price rule is applied"
            ),
        },
        "records_preparation_cost_only": True,
    }
    write_json_atomic(record_path, record)
    return record


def prepare(
    *,
    root: Path,
    manifest: dict[str, Any],
    source: Path,
    terminals: TerminalStore,
    timeout: float,
    base_config: dict[str, Any] | None = None,
    reflect_model: str | None = None,
) -> dict[str, Any]:
    producer = preparation_group(manifest["groups"])
    plan = dict(manifest["groups"][producer])
    plan["group_key"] = producer
    config = write_prepare_config(
        root, group=producer, base_config=base_config, reflect_model=reflect_model
    )
    record = _prepare_once(
        root=root,
        manifest=manifest,
        source=source,
        terminals=terminals,
        timeout=timeout,
        plan=plan,
        config=config,
        stage="prepare",
        prepared_root=root / "_prepared",
        record_path=root / "_prepared" / "preparation.json",
    )
    record["producer_group"] = producer
    record["isolated"] = False
    write_json_atomic(root / "_prepared" / "preparation.json", record)
    return record


def prepare_isolated(
    *,
    root: Path,
    manifest: dict[str, Any],
    group: str,
    source: Path,
    terminals: TerminalStore,
    timeout: float,
    base_config: dict[str, Any] | None = None,
    reflect_model: str | None = None,
) -> dict[str, Any]:
    """Prepare one group's incompatible contract without touching shared state."""
    guard = PathGuard(root)
    plan = dict(manifest["groups"][group])
    plan["group_key"] = group
    config = write_group_prepare_config(
        root, group, base_config=base_config, reflect_model=reflect_model
    )
    record = _prepare_once(
        root=root,
        manifest=manifest,
        source=source,
        terminals=terminals,
        timeout=timeout,
        plan=plan,
        config=config,
        stage=f"prepare.{group}",
        prepared_root=guard.child(group, "_prepared"),
        record_path=guard.child(group, "_prepared", "preparation.json"),
    )
    record["producer_group"] = group
    record["isolated"] = True
    write_json_atomic(guard.child(group, "_prepared", "preparation.json"), record)
    return record


def _receipt_validation(
    *,
    plan: dict[str, Any],
    source_hash: str,
    output: Path,
) -> dict[str, Any]:
    receipt_path = Path(plan["result_path"])
    progress_path = Path(plan["progress_path"])
    if not receipt_path.is_file():
        return {"status": "missing", "reason": "result receipt was not published"}
    try:
        receipt = read_json(receipt_path)
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "invalid", "reason": f"result receipt unreadable: {error}"}
    if receipt.get("protocol_version") != 1 or receipt.get("status") != "success":
        return {"status": "invalid", "reason": "result receipt is not a protocol-1 success"}
    if receipt.get("source_sha256") != source_hash:
        return {"status": "invalid", "reason": "result receipt source hash differs"}
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return {"status": "invalid", "reason": "result receipt has no outputs"}
    resolved_outputs = []
    for item in outputs:
        value = item.get("path") if isinstance(item, dict) else None
        if not isinstance(value, str):
            return {"status": "invalid", "reason": "result receipt output has no path"}
        path = Path(value).resolve()
        try:
            path.relative_to(Path(plan["output_dir"]).resolve())
        except ValueError:
            return {"status": "invalid", "reason": "output escaped the variant output directory"}
        if not path.is_file():
            return {"status": "invalid", "reason": f"output file is missing: {path}"}
        if item.get("sha256") != sha256_file(path):
            return {"status": "invalid", "reason": f"output hash differs: {path}"}
        resolved_outputs.append(str(path))
    state_dir = receipt.get("state_dir")
    if not isinstance(state_dir, str):
        return {"status": "invalid", "reason": "result receipt has no state directory"}
    try:
        Path(state_dir).resolve().relative_to(Path(plan["state_dir"]).resolve())
    except ValueError:
        return {"status": "invalid", "reason": "receipt state directory escaped the variant"}
    if resolved_outputs != [str(output.resolve())]:
        return {
            "status": "invalid",
            "reason": f"expected exactly the declared output {output}, got {resolved_outputs}",
        }
    progress = {}
    if progress_path.is_file():
        try:
            progress = read_json(progress_path)
        except (OSError, json.JSONDecodeError):
            progress = {"status": "unreadable"}
    return {
        "status": "valid",
        "run_id": receipt.get("run_id"),
        "state_dir": str(Path(state_dir).resolve()),
        "outputs": resolved_outputs,
        "progress_status": progress.get("status"),
    }


def _translate_arguments(
    *,
    group: str,
    source: Path,
    output: Path,
    plan: dict[str, Any],
) -> list[str]:
    polish = (
        plan["polish"]
        if isinstance(plan.get("polish"), bool)
        else bool(load_group_template(group)["pipeline"].get("polish"))
    )
    return [
        "translate",
        str(source),
        "--format",
        "txt",
        "--out",
        str(output),
        "--polish" if polish else "--no-polish",
        "--review",
        "--mono",
        "--no-bilingual",
        "--progress-json",
        plan["progress_path"],
        "--result-json",
        plan["result_path"],
    ]


def check_preparation_compatibility(
    *,
    root: Path,
    manifest: dict[str, Any],
    source_hash: str,
    allow_isolated_reprepare: bool = False,
) -> dict[str, Any]:
    """Compare the shared snapshot with each group's persisted preparation contract.

    A mismatch aborts before any translation unless isolated re-preparation was
    explicitly authorized. Persisted fingerprints are never relaxed to make a
    reuse succeed, and an isolated run never overwrites the shared snapshot.
    """
    actual = preparation_identity(root / "_prepared" / "state")
    groups: dict[str, Any] = {}
    for group, plan in manifest["groups"].items():
        try:
            expected = expected_preparation_identity(
                plan["config"],
                source_sha256=source_hash,
                source_lang=manifest["source"]["language"]["source"],
            )
            reasons = compare_preparation_identity(actual, expected)
            expected_summary = {
                key: value
                for key, value in expected.items()
                if key not in {"ingest_config", "language_policies"}
            }
        except Exception as error:
            reasons = [f"group configuration cannot be resolved: {type(error).__name__}: {error}"]
            expected_summary = None
        if reasons and allow_isolated_reprepare:
            status = "isolated_required"
            reprepare = "authorized: an isolated preparation will run under the group directory"
            cost = "separate preparation cost; never attributed to the shared snapshot"
        elif reasons:
            status = "incompatible"
            reprepare = "not attempted; pass --allow-isolated-reprepare to authorize one"
            cost = "not incurred; translation did not start"
        else:
            status = "reused"
            reprepare = "not required"
            cost = "not incurred"
        groups[group] = {
            "status": status,
            "reasons": reasons,
            "fingerprint": actual.get("fingerprint"),
            "expected": expected_summary,
            "reprepare": reprepare,
            "cost": cost,
            "snapshot": "shared",
        }
    payload = {
        "protocol_version": TERMINAL_PROTOCOL,
        "snapshot_fingerprint": actual.get("fingerprint"),
        "identity": actual,
        "groups": groups,
        "translation_started": False,
    }
    incompatible = {
        key: value for key, value in groups.items() if value["status"] == "incompatible"
    }
    if incompatible:
        path = root / "records" / "preparation_incompatibility.json"
        write_json_atomic(path, payload)
        detail = "; ".join(
            f"{key}: {', '.join(value['reasons'])}" for key, value in incompatible.items()
        )
        raise PreparationIncompatibilityError(
            f"Prepared state is incompatible with {len(incompatible)} group(s): {detail}. "
            "Pass --allow-isolated-reprepare to prepare those groups in isolation, or abort. "
            f"Incompatibility record: {path}"
        )
    payload["translation_started"] = True
    write_json_atomic(root / "records" / "preparation_compatibility.json", payload)
    return payload


def run_group(
    *,
    root: Path,
    manifest: dict[str, Any],
    group: str,
    source: Path,
    terminals: TerminalStore,
    timeout: float,
    output: Path,
    preparation: dict[str, Any],
    prepared: dict[str, Any],
    snapshot_dir: Path,
) -> dict[str, Any]:
    plan = manifest["groups"][group]
    guard = PathGuard(root)
    for field in ("output_dir", "cache_dir", "cwd"):
        Path(plan[field]).mkdir(parents=True, exist_ok=True)
    for field in ("progress_path", "result_path", "status_path", "inspect_path"):
        Path(plan[field]).parent.mkdir(parents=True, exist_ok=True)

    snapshot = read_snapshot_receipt_at(snapshot_dir)
    source_hash = sha256_file(source)
    if snapshot.get("source_sha256") != source_hash:
        raise EvalError(
            f"Prepared state is incompatible with group {group}: source SHA differs; "
            "re-prepare is required and its cost must be recorded explicitly"
        )
    if snapshot.get("target_lang") != manifest["source"]["language"]["target"]:
        raise EvalError(
            f"Prepared state is incompatible with group {group}: target language differs; "
            "re-prepare is required and its cost must be recorded explicitly"
        )
    run_relative_path = prepared.get("run_relative_path")
    if not isinstance(run_relative_path, str) or not run_relative_path:
        raise CorruptIdentityError("Preparation record has no run-relative state path")
    destination = guard.child(group, "state", *run_relative_path.split("/"))
    seed_state(
        Path(snapshot_dir) / "state",
        destination,
        expected_digest=snapshot["tree_sha256"],
    )
    config = Path(plan["config"])
    translate_outcome, translate_logs = run_child(
        group=group,
        stage="translate",
        plan=plan,
        config=str(config),
        argv=_translate_arguments(
            group=group,
            source=source,
            output=output,
            plan=plan,
        ),
        cwd=Path(plan["cwd"]),
        binding_path=guard.child(group, "receipts", "binding.translate.json"),
        stdout_path=Path(plan["stdout_path"]),
        stderr_path=Path(plan["stderr_path"]),
        timeout=timeout,
    )
    if translate_outcome.exit_code == 0 and translate_outcome.binding_valid:
        validation = _receipt_validation(plan=plan, source_hash=source_hash, output=output)
        translate_logs["terminal"]["receipt_validation"] = validation
        if validation["status"] != "valid":
            translate_logs["terminal"]["status"] = "receipt_invalid"
    else:
        validation = {"status": "not_checked"}
    translate_path = terminals.write(translate_logs["terminal"])

    status_outcome, status_logs = run_child(
        group=group,
        stage="status",
        plan=plan,
        config=str(config),
        argv=["status", str(source)],
        cwd=Path(plan["cwd"]),
        binding_path=guard.child(group, "receipts", "binding.status.json"),
        stdout_path=Path(plan["status_path"]),
        stderr_path=guard.child(group, "logs", "status.stderr.log"),
        timeout=timeout,
    )
    terminals.write(status_logs["terminal"])

    inspect_outcome, inspect_logs = run_child(
        group=group,
        stage="inspect",
        plan=plan,
        config=str(config),
        argv=["inspect-run", str(source)],
        cwd=Path(plan["cwd"]),
        binding_path=guard.child(group, "receipts", "binding.inspect.json"),
        stdout_path=Path(plan["inspect_path"]),
        stderr_path=guard.child(group, "logs", "inspect.stderr.log"),
        timeout=timeout,
    )
    terminals.write(inspect_logs["terminal"])

    return {
        "group": group,
        "state_dir": plan["state_dir"],
        "output": str(output),
        "preparation": preparation["groups"][group],
        "prepared_snapshot": str(snapshot_dir),
        "prepared_isolated": bool(prepared.get("isolated")),
        "translate_terminal": str(translate_path),
        "receipt_validation": validation,
        "exit_codes": {
            "translate": translate_outcome.exit_code,
            "status": status_outcome.exit_code,
            "inspect": inspect_outcome.exit_code,
        },
        "durations": {
            "translate": translate_outcome.duration_seconds,
            "status": status_outcome.duration_seconds,
            "inspect": inspect_outcome.duration_seconds,
        },
    }


def execute(
    *,
    root: Path,
    manifest: dict[str, Any],
    source: Path,
    terminals: TerminalStore,
    timeout: float,
    base_config: dict[str, Any] | None = None,
    reflect_model: str | None = None,
    allow_isolated_reprepare: bool = False,
) -> dict[str, Any]:
    source_hash = sha256_file(source)
    if source_hash != manifest["source"]["sha256"]:
        raise CorruptIdentityError("Fixture source hash changed before execution")
    preparation = prepare(
        root=root,
        manifest=manifest,
        source=source,
        terminals=terminals,
        timeout=timeout,
        base_config=base_config,
        reflect_model=reflect_model,
    )
    compatibility = check_preparation_compatibility(
        root=root,
        manifest=manifest,
        source_hash=source_hash,
        allow_isolated_reprepare=allow_isolated_reprepare,
    )
    isolated: dict[str, Any] = {}
    isolated_groups = [
        group
        for group, entry in compatibility["groups"].items()
        if entry["status"] == "isolated_required"
    ]
    for group in isolated_groups:
        record = prepare_isolated(
            root=root,
            manifest=manifest,
            group=group,
            source=source,
            terminals=terminals,
            timeout=timeout,
            base_config=base_config,
            reflect_model=reflect_model,
        )
        isolated[group] = record
        compatibility["groups"][group]["snapshot"] = "isolated"
        compatibility["groups"][group]["status"] = "isolated"
        compatibility["groups"][group]["reprepare"] = "completed in an isolated group snapshot"
        compatibility["groups"][group]["isolated_snapshot"] = record["snapshot_dir"]
        compatibility["groups"][group]["isolated_terminal"] = record["terminal_path"]
        compatibility["groups"][group]["isolated_fingerprint"] = record["snapshot"]["tree_sha256"]
        compatibility["groups"][group]["cost"] = record["cost"]
    if isolated_groups:
        compatibility["translation_started"] = True
        write_json_atomic(root / "records" / "preparation_compatibility.json", compatibility)
    snapshot_before = preparation["snapshot"]["tree_sha256"]
    isolated_before = {
        group: hash_tree(Path(record["snapshot_dir"]) / "state")
        for group, record in isolated.items()
    }
    if sha256_file(source) != source_hash:
        raise CorruptIdentityError("Fixture source changed during preparation")
    groups: dict[str, Any] = {}
    output_name = f"{source.stem}.zh.txt"
    for group, plan in manifest["groups"].items():
        output = Path(plan["output_dir"]) / output_name
        if group in isolated:
            snapshot_dir = Path(isolated[group]["snapshot_dir"])
            prepared = isolated[group]
        else:
            snapshot_dir = Path(preparation["snapshot_dir"])
            prepared = preparation
        groups[group] = run_group(
            root=root,
            manifest=manifest,
            group=group,
            source=source,
            terminals=terminals,
            timeout=timeout,
            output=output,
            preparation=compatibility,
            prepared=prepared,
            snapshot_dir=snapshot_dir,
        )
    if sha256_file(source) != source_hash:
        raise CorruptIdentityError("Fixture source changed during execution")
    snapshot = read_snapshot_receipt_at(Path(preparation["snapshot_dir"]))
    if not snapshot.get("tree_sha256"):
        raise CorruptIdentityError("Preparation snapshot receipt has no digest")
    snapshot_after = hash_tree(Path(preparation["snapshot_dir"]) / "state")
    if snapshot_after != snapshot_before or snapshot.get("tree_sha256") != snapshot_before:
        raise CorruptIdentityError("Preparation snapshot changed during variant runs")
    for group, record in isolated.items():
        current = hash_tree(Path(record["snapshot_dir"]) / "state")
        if current != isolated_before[group]:
            raise CorruptIdentityError(f"Isolated preparation snapshot for {group} changed")
    rows = terminals.validate_complete(
        groups=manifest["groups"], isolated_preparations=isolated_groups
    )
    _validate_receipt_identities(rows, manifest["groups"])
    return {
        "preparation": preparation,
        "isolated_preparation": isolated,
        "preparation_compatibility": compatibility,
        "groups": groups,
        "terminals": {f"{group}.{stage}": row for (group, stage), row in rows.items()},
        "source_unchanged": True,
        "preparation_unchanged": True,
        "snapshot_tree_sha256": snapshot_after,
    }


def _validate_receipt_identities(
    rows: dict[tuple[str, str], dict[str, Any]], groups: Iterable[str]
) -> None:
    run_ids: dict[str, str] = {}
    for group in groups:
        terminal = rows[(group, "translate")]
        validation = terminal.get("receipt_validation") or {}
        if validation.get("status") != "valid":
            continue
        run_id = validation.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise CorruptIdentityError(f"Group {group} receipt has no run_id")
        if run_id in run_ids:
            raise CorruptIdentityError(
                f"Groups {run_ids[run_id]} and {group} share run_id {run_id}"
            )
        run_ids[run_id] = group


_SENSITIVE_KEY = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|credential)")
_URL_CREDENTIALS = re.compile(r"(?i)(://)[^/@\s:]+(?::[^/@\s]*)?@")


def redact_metadata(value: Any) -> Any:
    """Remove credential-like values while preserving environment-variable references."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key)
            if _SENSITIVE_KEY.search(name) and not name.lower().endswith("_env"):
                result[name] = "<redacted>"
            else:
                result[name] = redact_metadata(item)
        return result
    if isinstance(value, list):
        return [redact_metadata(item) for item in value]
    if isinstance(value, str):
        return _URL_CREDENTIALS.sub(r"\1<redacted>@", value)
    return value


def _run_metadata_command(
    *, plan: dict[str, Any], config: str, argv: list[str], timeout: float
) -> tuple[Any | None, str | None]:
    command = _child_command(plan, config, argv)
    env = subprocess_env(
        offline=False,
        passthrough=credential_env_names(config),
        pythonpath=[
            str(Path(plan["code_source"]) / "packages" / "core"),
            str(Path(plan["code_source"]) / "packages" / "cli"),
        ],
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(plan["cwd"])),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"{type(error).__name__}: {error}"
    text, truncated = bounded_text(completed.stdout)
    if completed.returncode != 0 or truncated:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        return None, detail or f"exit code {completed.returncode}"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as error:
        return None, f"metadata output is not JSON: {error}"


def capture_runtime_metadata(
    *, root: Path, plans: dict[str, dict[str, Any]], timeout: float = 120.0
) -> None:
    """Record real model routes and the selected interpreter binding per group."""
    PathGuard(root).child("records").mkdir(parents=True, exist_ok=True)
    binding_cache: dict[str, dict[str, Any]] = {}
    for key, plan in plans.items():
        reflective = bool((plan.get("reflection") or {}).get("enabled"))
        reflect_status = "unknown" if reflective else "not_enabled"
        if plan["adapter"] == "offline":
            plan["routes"] = {
                "status": "unknown",
                "reason": "the explicit offline adapter does not expose provider routes",
                "translation.body": "unknown",
                "polish.body": "disabled" if not plan["polish"] else "unknown",
                "reflect.body": reflect_status,
            }
            plan["route_fingerprints"] = {
                "status": "unknown",
                "reason": "model metadata commands were not run for the offline adapter",
            }
            continue
        config = plan["config"]
        Path(plan["cwd"]).mkdir(parents=True, exist_ok=True)
        listed, list_error = _run_metadata_command(
            plan=plan, config=config, argv=["models", "list", "--json"], timeout=timeout
        )
        operations: dict[str, Any] = {}
        reasons = [list_error] if list_error else []
        for operation in ("translation.body", "polish.body", "review.scan"):
            described, explain_error = _run_metadata_command(
                plan=plan,
                config=config,
                argv=["models", "explain", "--operation", operation],
                timeout=timeout,
            )
            if explain_error:
                reasons.append(f"{operation}: {explain_error}")
            else:
                operations[operation] = redact_metadata(described)
        if reflective:
            described, explain_error = _run_metadata_command(
                plan=plan,
                config=config,
                argv=["models", "explain", "--operation", "reflect.body"],
                timeout=timeout,
            )
            if explain_error:
                reasons.append(f"reflect.body: {explain_error}")
            else:
                operations["reflect.body"] = redact_metadata(described)
        if listed is None or reasons:
            plan["routes"] = {
                "status": "unknown",
                "reason": "; ".join(reasons)[:2000] or "model route metadata is unavailable",
                "translation.body": "unknown",
                "polish.body": "disabled" if not plan["polish"] else "unknown",
                "reflect.body": reflect_status,
            }
            plan["route_fingerprints"] = {
                "status": "unknown",
                "reason": "at least one route description could not be captured",
            }
            continue
        routes = redact_metadata(listed)
        plan["routes"] = {
            "status": "known",
            "translation.body": routes.get("translation.body", "unknown"),
            "polish.body": "disabled"
            if not plan["polish"]
            else routes.get("polish.body", "unknown"),
            "reflect.body": routes.get("reflect.body", "unknown") if reflective else "not_enabled",
            "operations": operations,
            "listed": routes,
        }
        plan["route_fingerprints"] = {
            "status": "known",
            "fingerprint": sha256_text(json.dumps(operations, sort_keys=True)),
        }
        if reflective:
            reflect = routes.get("reflect.body") or {}
            polish = routes.get("polish.body") or {}
            inherited = reflect.get("origin") == "inherits polish.body"
            same_model = (
                reflect.get("model") is not None
                and reflect.get("model") == polish.get("model")
                and reflect.get("provider") == polish.get("provider")
            )
            plan["reflection"]["observed_route"] = {
                "reflect_model": reflect.get("model"),
                "reflect_provider": reflect.get("provider"),
                "polish_model": polish.get("model"),
                "polish_provider": polish.get("provider"),
                "reflect_origin": reflect.get("origin"),
                "separation": "inherited" if inherited else ("same" if same_model else "distinct"),
                "source": "wenyi models list --json",
            }
            if key == "D" and same_model:
                raise EvalError(
                    "Group D reflect.body and polish.body resolve to the same model identity"
                )
        if plan["code_source"] not in binding_cache:
            binding_path = PathGuard(root).child(
                "records",
                f"binding.{Path(plan['code_source']).name}.{sha256_text(plan['code_source'])[:8]}.json",
            )
            env = subprocess_env(
                offline=False,
                passthrough=credential_env_names(config),
                pythonpath=[
                    str(Path(plan["code_source"]) / "packages" / "core"),
                    str(Path(plan["code_source"]) / "packages" / "cli"),
                ],
            )
            _provider_binding(
                plan=plan,
                binding_path=binding_path,
                cwd=Path(plan["cwd"]),
                env=env,
            )
            binding_cache[plan["code_source"]] = read_json(binding_path)
        plan["runtime_binding"] = binding_cache[plan["code_source"]]


def prepare_manifest(
    *,
    root: Path,
    manifest: dict[str, Any],
    repo_root: Path,
    timeout: float = 900.0,
    base_config: dict[str, Any] | None = None,
    reflect_model: str | None = None,
) -> dict[str, Any]:
    """Bind configs, capture real routes and commit the manifest exactly once."""
    groups = tuple(manifest["groups"])
    config_hashes = write_variant_configs(
        root, base_config=base_config, groups=groups, reflect_model=reflect_model
    )
    manifest["root"] = str(root)
    for key, group in manifest["groups"].items():
        group["config_sha256"] = config_hashes[key]
    if not manifest.get("diagnostic"):
        require_frozen_bindings(manifest)
    capture_runtime_metadata(root=root, plans=manifest["groups"])
    validate_manifest(manifest, repo_root=repo_root)
    producer = manifest["preparation"]["producer_group"]
    manifest["llm"]["routes"] = manifest["groups"][producer]["routes"]
    manifest["llm"]["route_fingerprints"] = manifest["groups"][producer]["route_fingerprints"]
    if not manifest.get("diagnostic"):
        require_real_ready(manifest)
    write_manifest(root, manifest)
    return load_manifest(root / "manifest.json")


def execute_planned(
    *,
    root: Path,
    manifest: dict[str, Any],
    source: Path,
    timeout: float,
    base_config: dict[str, Any] | None = None,
    reflect_model: str | None = None,
    allow_isolated_reprepare: bool = False,
) -> dict[str, Any]:
    execution = execute(
        root=root,
        manifest=manifest,
        source=source,
        terminals=TerminalStore(root),
        timeout=timeout,
        base_config=base_config,
        reflect_model=reflect_model,
        allow_isolated_reprepare=allow_isolated_reprepare,
    )
    write_json_atomic(root / "records" / "execution.json", execution)
    return execution


def start_evaluation(
    *,
    root: Path,
    source: Path,
    manifest: dict[str, Any],
    repo_root: Path,
    timeout: float = 900.0,
    base_config: dict[str, Any] | None = None,
    reflect_model: str | None = None,
    allow_isolated_reprepare: bool = False,
) -> dict[str, Any]:
    """Write the manifest and run all variants inside an already initialized root."""
    loaded = prepare_manifest(
        root=root,
        manifest=manifest,
        repo_root=repo_root,
        timeout=timeout,
        base_config=base_config,
        reflect_model=reflect_model,
    )
    execution = execute_planned(
        root=root,
        manifest=loaded,
        source=source,
        timeout=timeout,
        base_config=base_config,
        reflect_model=reflect_model,
        allow_isolated_reprepare=allow_isolated_reprepare,
    )
    return {
        "root": root,
        "manifest": loaded,
        "execution": execution,
    }


_SOURCE_FORMATS = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".epub": "epub",
    ".fb2": "fb2",
    ".docx": "docx",
    ".pdf": "pdf",
}


def _source_format(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    return _SOURCE_FORMATS.get(path.suffix.lower(), "text")


def _binding_from_args(args: argparse.Namespace, arm: str) -> dict[str, Any]:
    adapter = args.adapter
    executable = getattr(args, f"{arm}_executable") or sys.executable
    argv = getattr(args, f"{arm}_argv") or []
    if adapter == "offline":
        return {"adapter": "offline", "executable": sys.executable}
    if not argv:
        argv = ["-m", "wenyi_cli"]
    return {"adapter": "provider", "executable": executable, "argv_prefix": argv}


def _language(args: argparse.Namespace, base: dict[str, Any] | None) -> tuple[str, str]:
    configured = (base or {}).get("language") or {}
    source = args.source_language or configured.get("source") or "en"
    target = args.target_language or configured.get("target") or "zh"
    if base is not None:
        base.setdefault("language", {}).update({"source": source, "target": target})
    return source, target


def _build_evaluation(
    args: argparse.Namespace, *, execute_after: bool
) -> tuple[Path, dict[str, Any], dict[str, Any] | None]:
    """Create one evaluation root from CLI arguments, optionally executing it."""
    from .cost import load_prices
    from .fixture import fixture_metadata
    from .manifest import (
        BASELINE_SHA,
        DEFAULT_GROUPS,
        REFLECTIVE_GROUPS,
        WP0_SHA,
        build_group_plans,
        build_manifest,
        load_user_config,
        normalize_groups,
    )

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    runs_dir = resolve_absolute(args.runs_dir, label="--runs-dir")
    source = resolve_absolute(args.source, label="--source")
    if not source.is_file():
        raise EvalError(f"Source does not exist: {source}")
    if not args.license:
        raise EvalError("--license is required so the source provenance is explicit")
    requested = normalize_groups(args.groups or DEFAULT_GROUPS)
    if "D" in requested and not args.heterogeneous_reflect_model:
        raise EvalError(
            "Group D requires --heterogeneous-reflect-model naming a model profile defined by "
            "the user configuration; no model ID is invented"
        )
    diagnostic = bool(args.diagnostic or args.adapter == "offline")
    if not diagnostic and not args.candidate_sha:
        raise EvalError("A real comparison requires --candidate-sha for the candidate code source")
    book = args.book or source.stem
    eval_id = args.eval_id or new_eval_id()
    root = initialize_root(runs_dir, eval_id=eval_id, book=book)
    base_config = load_user_config(args.config) if args.config else None
    source_lang, target_lang = _language(args, base_config)
    with ownership_lock(root):
        copied = root / "source" / source.name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
        config_hashes = write_variant_configs(
            root,
            base_config=base_config,
            groups=requested,
            reflect_model=args.heterogeneous_reflect_model,
        )
        plans = build_group_plans(
            root,
            baseline_source=Path(args.baseline_code)
            if args.baseline_code
            else _default_baseline(repo_root),
            wp0_source=Path(args.wp0_code) if args.wp0_code else _default_wp0(repo_root),
            candidate_source=Path(args.candidate_code or repo_root),
            baseline_expected_sha=args.baseline_sha or BASELINE_SHA,
            wp0_expected_sha=args.wp0_sha or WP0_SHA,
            candidate_expected_sha=args.candidate_sha or "unfrozen",
            diagnostic=diagnostic,
            requested_groups=requested,
            baseline_binding=_binding_from_args(args, "baseline"),
            wp0_binding=_binding_from_args(args, "wp0"),
            candidate_binding=_binding_from_args(args, "candidate"),
            base_config=base_config,
            reflect_model=args.heterogeneous_reflect_model,
        )
        fixture = (
            fixture_metadata()
            if args.synthetic
            else {
                "license": args.license,
                "provenance": args.provenance or "Supplied by the evaluation operator.",
                "synthetic": False,
                "source_language": source_lang,
                "target_language": target_lang,
            }
        )
        fixture["license"] = args.license
        fixture["source_language"] = source_lang
        fixture["target_language"] = target_lang
        notes = [
            "Missing prices leave cost unknown; they never disable non-cost tooling.",
            "Human-review rows are suggestions until an operator submits ratings.",
        ]
        if diagnostic:
            notes.append(
                "Diagnostic run: results do not support translation-quality or provider-cost claims."
            )
        candidate_groups = [group for group in requested if group in REFLECTIVE_GROUPS]
        if candidate_groups:
            notes.append(
                "C/C_keep/D bind the current workspace source; its dirty or unfrozen identity is "
                "diagnostic and is not a frozen quality candidate or an A_prime relabel."
            )
            notes.append(
                "Reflection artifacts are diagnostic records; a current artifact is not a clean "
                "quality verdict."
            )
        if "D" in requested:
            notes.append(
                "D family heterogeneity is operator-declared; the harness verifies only that the "
                "resolved reflect.body and polish.body identities differ."
            )
        if args.allow_isolated_reprepare:
            notes.append(
                "Incompatible preparation contracts may run isolated preparation under the group "
                "directory; that cost is charged separately and never reuses incompatible state."
            )
        if args.candidate_sha in (None, "unfrozen"):
            notes.append("The candidate code SHA is unfrozen and cannot support a real comparison.")
        notes.extend(args.unknown_note or [])
        producer = preparation_group(plans)
        manifest = build_manifest(
            root=root,
            runs_dir=runs_dir,
            eval_id=eval_id,
            book=book,
            source_path=copied,
            source_format=_source_format(source, args.source_format),
            language={"source": source_lang, "target": target_lang},
            fixture=fixture,
            plans=plans,
            config_hashes=config_hashes,
            mode=args.mode or ("offline_smoke" if args.adapter == "offline" else "evaluation"),
            diagnostic=diagnostic,
            repo_root=repo_root,
            seed=args.seed,
            prices=load_prices(args.prices),
            preparation_entry=(
                f"{plans[producer]['code_source']} prepare via {plans[producer]['executable']}"
            ),
            unknown_notes=notes,
            user_config=(
                {"path": str(Path(args.config).resolve()), "sha256": sha256_file(args.config)}
                if args.config
                else None
            ),
            sampling={"min": args.human_min, "max": args.human_max},
        )
        loaded = prepare_manifest(
            root=root,
            manifest=manifest,
            repo_root=repo_root,
            timeout=args.timeout,
            base_config=base_config,
            reflect_model=args.heterogeneous_reflect_model,
        )
        execution = None
        if execute_after:
            execution = execute_planned(
                root=root,
                manifest=loaded,
                source=copied,
                timeout=args.timeout,
                base_config=base_config,
                reflect_model=args.heterogeneous_reflect_model,
                allow_isolated_reprepare=args.allow_isolated_reprepare,
            )
    return root, loaded, execution


def _finish(root: Path, manifest: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    from .cost import build_cost_summary
    from .extract import extract_all
    from .pairwise import export_pairwise, import_pairwise_ratings
    from .report import write_report
    from .sample import build_samples

    extraction = extract_all(root=root, manifest=manifest)
    selection = build_samples(
        root=root, manifest=manifest, extraction=extraction, limits=manifest.get("sampling")
    )
    export_pairwise(root=root, manifest=manifest, extraction=extraction)
    import_pairwise_ratings(root, root / "samples" / "pairwise.csv")
    cost = build_cost_summary(
        root=root, manifest=manifest, extraction=extraction, execution=execution
    )
    report = write_report(
        root=root,
        manifest=manifest,
        extraction=extraction,
        execution=execution,
        cost=cost,
    )
    return {"extraction": extraction, "selection": selection, "report": report}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_baseline(repo_root: Path) -> Path:
    from .manifest import BASELINE_REF

    for candidate in (repo_root / BASELINE_REF, repo_root.parent / BASELINE_REF):
        if candidate.is_dir():
            return candidate
    raise EvalError(
        f"Cannot find the read-only baseline checkout at {BASELINE_REF}; pass --baseline-code"
    )


def _default_wp0(repo_root: Path) -> Path:
    from .manifest import WP0_REF

    for candidate in (repo_root / WP0_REF, repo_root.parent / WP0_REF):
        if candidate.is_dir():
            return candidate
    raise EvalError(f"Cannot find the frozen WP0 checkout at {WP0_REF}; pass --wp0-code")


def _load_root_manifest(root: str) -> tuple[Path, dict[str, Any]]:
    resolved = resolve_absolute(root, label="--root")
    return resolved, load_manifest(resolved / "manifest.json")


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs-dir", required=True, help="Absolute disposable runs directory")
    parser.add_argument("--source", required=True, help="Book source file (license-clear)")
    parser.add_argument("--source-format", default=None, help="Override the inferred format")
    parser.add_argument("--source-language", default=None, help="Source language code")
    parser.add_argument("--target-language", default=None, help="Target language code")
    parser.add_argument("--license", required=True, help="Source license identifier")
    parser.add_argument("--provenance", default=None, help="Source provenance statement")
    parser.add_argument("--synthetic", action="store_true", help="Mark the source as synthetic")
    parser.add_argument("--book", default=None, help="Book name (default: source stem)")
    parser.add_argument("--eval-id", default=None, help="New evaluation id")
    parser.add_argument("--repo-root", default=None, help="Repository root for baseline checks")
    parser.add_argument("--baseline-code", default=None, help="Read-only baseline checkout")
    parser.add_argument("--baseline-sha", default=None, help="Expected baseline commit")
    parser.add_argument("--wp0-code", default=None, help="Frozen WP0 checkout")
    parser.add_argument("--wp0-sha", default=None, help="Expected WP0 commit")
    parser.add_argument("--candidate-code", default=None, help="Candidate code source")
    parser.add_argument(
        "--candidate-sha", default=None, help="Expected candidate commit (required for real runs)"
    )
    parser.add_argument(
        "--groups",
        default=None,
        help=(
            "Comma-separated groups (default: A,A2,A_prime,A_prime2,B). "
            "C/C_keep/D add reflective polish; E is not implemented."
        ),
    )
    parser.add_argument(
        "--heterogeneous-reflect-model",
        default=None,
        help="Model profile used as D's explicit reflect.body route",
    )
    parser.add_argument(
        "--allow-isolated-reprepare",
        action="store_true",
        help="Authorize isolated re-preparation for incompatible preparation contracts",
    )
    parser.add_argument("--config", default=None, help="User configuration YAML")
    parser.add_argument("--prices", default=None, help="Optional provider price rule JSON")
    parser.add_argument("--seed", type=int, default=20261116, help="Sampling seed")
    parser.add_argument("--human-min", type=int, default=200, help="Minimum human sample rows")
    parser.add_argument("--human-max", type=int, default=300, help="Maximum human sample rows")
    parser.add_argument(
        "--adapter",
        choices=("provider", "offline"),
        default="provider",
        help="Explicitly select the production provider path or the offline adapter",
    )
    parser.add_argument("--diagnostic", action="store_true", help="Mark results as not comparable")
    parser.add_argument("--mode", default=None, help="Recorded evaluation mode label")
    parser.add_argument("--timeout", type=float, default=900.0, help="Per-command timeout")
    parser.add_argument(
        "--baseline-executable", default=None, help="Interpreter that runs the baseline source"
    )
    parser.add_argument(
        "--baseline-argv",
        action="append",
        default=[],
        help="Repeatable argv element preceding the Wenyi command for the baseline",
    )
    parser.add_argument(
        "--candidate-executable", default=None, help="Interpreter that runs the candidate source"
    )
    parser.add_argument(
        "--candidate-argv",
        action="append",
        default=[],
        help="Repeatable argv element preceding the Wenyi command for the candidate",
    )
    parser.add_argument(
        "--wp0-executable", default=None, help="Interpreter that runs the frozen WP0 source"
    )
    parser.add_argument(
        "--wp0-argv",
        action="append",
        default=[],
        help="Repeatable argv element preceding the Wenyi command for WP0",
    )
    parser.add_argument(
        "--unknown-note", action="append", default=[], help="Explicit known limitation"
    )


def _print_summary(root: Path, result: dict[str, Any]) -> None:
    report = result.get("report") or {}
    print(f"Eval root: {root}")
    print(f"Manifest: {root / 'manifest.json'}")
    print(f"Report: {root / 'report' / 'summary.md'}")
    print(f"Conclusion: {report.get('conclusion', 'not run')}")
    coverage = (result.get("extraction") or {}).get("groups", {})
    for group, item in coverage.items():
        print(f"Coverage {group}: {item.get('coverage')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wenyi_eval.run_variants",
        description=(
            "Plan, run, extract, export, import and report a Wenyi variant evaluation. "
            "The offline adapter is only used when --adapter offline is selected."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("plan", "Write a manifest using local CLI metadata without translating"),
        ("run", "Plan, execute, extract, export and report an evaluation"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        _add_plan_arguments(sub)
    for name, help_text in (
        ("extract", "Re-extract an existing evaluation root"),
        ("export", "Rebuild human-review samples and pairwise material"),
        ("report", "Rebuild cost and report artifacts"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--root", required=True, help="Existing evaluation root")
    import_parser = subparsers.add_parser("import", help="Import pairwise ratings")
    import_parser.add_argument("--root", required=True, help="Existing evaluation root")
    import_parser.add_argument(
        "--ratings", default=None, help="Ratings CSV (default: samples/pairwise.csv)"
    )
    args = parser.parse_args(argv)

    try:
        if args.command in {"plan", "run"}:
            root, manifest, execution = _build_evaluation(args, execute_after=args.command == "run")
            if execution is None:
                print(f"Planned evaluation: {root}")
                print(f"Manifest: {root / 'manifest.json'}")
                return 0
            result = _finish(root, manifest, execution)
            _print_summary(root, result)
            return 0
        root, manifest = _load_root_manifest(args.root)
        if args.command == "extract":
            from .extract import extract_all

            extraction = extract_all(root=root, manifest=manifest)
            print(f"Extraction: {root / 'extraction.json'}")
            for group, item in extraction["groups"].items():
                print(f"Coverage {group}: {item.get('coverage')}")
            return 0
        if args.command == "export":
            from .extract import load_extraction
            from .pairwise import export_pairwise
            from .sample import build_samples

            extraction = load_extraction(root)
            build_samples(root=root, manifest=manifest, extraction=extraction)
            export_pairwise(root=root, manifest=manifest, extraction=extraction)
            print(f"Samples: {root / 'samples'}")
            return 0
        if args.command == "import":
            from .pairwise import import_pairwise_ratings

            ratings = Path(args.ratings) if args.ratings else root / "samples" / "pairwise.csv"
            result = import_pairwise_ratings(root, ratings)
            print(f"Imported ratings: {result['rated_rows']} of {result['denominator']}")
            return 0
        if args.command == "report":
            from .cost import build_cost_summary
            from .extract import load_extraction
            from .report import write_report

            extraction = load_extraction(root)
            execution = read_json(root / "records" / "execution.json")
            cost = build_cost_summary(
                root=root, manifest=manifest, extraction=extraction, execution=execution
            )
            report = write_report(
                root=root,
                manifest=manifest,
                extraction=extraction,
                execution=execution,
                cost=cost,
            )
            print(f"Report: {root / 'report' / 'summary.md'}")
            print(f"Conclusion: {report['conclusion']}")
            return 0
    except EvalError as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
