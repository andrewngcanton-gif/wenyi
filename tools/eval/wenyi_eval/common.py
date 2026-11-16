"""Shared filesystem, identity and safety helpers for the evaluation harness."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


class EvalError(RuntimeError):
    """A harness precondition failed; callers report the message without a traceback."""


class UnsafePathError(EvalError):
    """An evaluation path escaped its disposable root or points at real user state."""


class UnsafeSnapshotError(EvalError):
    """A preparation snapshot would carry body targets, locks or foreign paths."""


class OwnershipError(EvalError):
    """Another process already owns this evaluation directory."""


class CorruptIdentityError(EvalError):
    """A persisted identity, receipt or source hash is inconsistent."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_eval_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root: str | os.PathLike[str]) -> str:
    """Hash relative paths and file bytes; directory ordering is stable."""
    base = Path(root)
    if not base.is_dir():
        raise EvalError(f"Directory does not exist: {base}")
    digest = hashlib.sha256()
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str | os.PathLike[str], value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)


def write_text_atomic(path: str | os.PathLike[str], text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, target)


def append_jsonl(path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise EvalError(f"JSONL row is not an object in {path}")
                rows.append(row)
    return rows


def resolve_absolute(path: str | os.PathLike[str], *, label: str = "path") -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise UnsafePathError(f"{label} must be absolute: {path}")
    return value.resolve()


class PathGuard:
    """Require every runtime path to stay inside one disposable evaluation root."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = resolve_absolute(root, label="evaluation root")

    def ensure(self, path: str | os.PathLike[str], *, label: str = "path") -> Path:
        resolved = resolve_absolute(path, label=label)
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise UnsafePathError(
                f"{label} escapes evaluation root {self.root}: {resolved}"
            ) from None
        return resolved

    def child(self, *parts: str) -> Path:
        return self.ensure(self.root.joinpath(*parts))


@contextmanager
def ownership_lock(root: str | os.PathLike[str]) -> Iterator[Path]:
    """Claim one evaluation root with an exclusive-create record.

    The record is never overwritten silently: a second active or abandoned
    owner is reported to the operator, who decides whether to keep it.
    """
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".owner.json"
    payload = json.dumps(
        {"pid": os.getpid(), "started_at": utc_now(), "cwd": os.getcwd()},
        ensure_ascii=False,
    ).encode("utf-8")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        previous = read_json(lock_path) if lock_path.is_file() else {}
        raise OwnershipError(
            f"Evaluation directory is already owned: {directory} ({previous}). "
            "Remove .owner.json only after confirming no run is active."
        ) from None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def bounded_text(data: bytes, limit: int = 64 * 1024) -> tuple[str, bool]:
    """Decode captured output and report whether it was truncated."""
    truncated = len(data) > limit
    chunk = data[:limit]
    return chunk.decode("utf-8", errors="replace"), truncated


@dataclass(frozen=True)
class GitIdentity:
    code_source: str
    head_sha: str
    branch: str
    dirty: bool
    status_sha256: str
    worktree_diff_sha256: str
    diff_identity: str
    describe: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code_source": self.code_source,
            "head_sha": self.head_sha,
            "branch": self.branch,
            "dirty": self.dirty,
            "status_sha256": self.status_sha256,
            "worktree_diff_sha256": self.worktree_diff_sha256,
            "diff_identity": self.diff_identity,
            "describe": self.describe,
        }


def _git(code_source: str, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", code_source, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise EvalError(f"git {' '.join(args)} failed in {code_source}: {detail}")
    return result.stdout


def git_identity(code_source: str | os.PathLike[str]) -> GitIdentity:
    source = str(resolve_absolute(code_source, label="code source"))
    head_sha = _git(source, "rev-parse", "HEAD").decode("ascii").strip()
    branch = _git(source, "branch", "--show-current").decode("utf-8", errors="replace").strip()
    status = _git(source, "status", "--porcelain=v1", "--untracked-files=normal")
    diff = _git(source, "diff", "--binary", "HEAD")
    describe = (
        _git(source, "describe", "--tags", "--always", "--dirty")
        .decode("utf-8", errors="replace")
        .strip()
    )
    status_sha = sha256_bytes(status)
    diff_sha = sha256_bytes(diff)
    identity = sha256_text(f"{head_sha}\0{status_sha}\0{diff_sha}")
    return GitIdentity(
        code_source=source,
        head_sha=head_sha,
        branch=branch,
        dirty=bool(status.strip()),
        status_sha256=status_sha,
        worktree_diff_sha256=diff_sha,
        diff_identity=identity,
        describe=describe,
    )


def software_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    def package(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "unknown"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wenyi_core": package("wenyi-core"),
        "wenyi_cli": package("wenyi-cli"),
        "wenyi_eval": package("wenyi-eval"),
        "git": _optional_git_version(),
    }


def _optional_git_version() -> str:
    try:
        result = subprocess.run(
            ["git", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.decode("utf-8", errors="replace").strip() or "unknown"


_ENV_ALLOWLIST = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
)


def subprocess_env(
    *,
    offline: bool = False,
    passthrough: list[str] | tuple[str, ...] = (),
    extra: dict[str, str] | None = None,
    pythonpath: list[str] | tuple[str, ...] = (),
) -> dict[str, str]:
    """Build an allowlisted child environment without serializing credentials.

    ``passthrough`` names credential variables that the selected provider config
    references by environment name. Only the variable names are recorded; their
    values are copied into the child and never written to a receipt.
    """
    env = {key: value for key, value in os.environ.items() if key in _ENV_ALLOWLIST}
    for name in passthrough:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            # The old baseline must stay byte-for-byte read-only; -B plus this
            # prevents __pycache__ writes into .review/upstream.
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
        }
    )
    if offline:
        env["WENYI_EVAL_OFFLINE"] = "1"
    if pythonpath:
        existing = env.get("PYTHONPATH")
        parts = [str(item) for item in pythonpath]
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


def copy_without_following_symlinks(source: Path, destination: Path) -> None:
    """Copy a file after refusing symlinks; snapshots must stay ordinary files."""
    if source.is_symlink():
        raise UnsafeSnapshotError(f"Refusing to copy symlink into snapshot: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
