"""Evaluation manifest, group bindings and disposable path layout.

The manifest is an identity receipt, not a result: it is written before any
child process starts and records only observed values or explicit unknowns.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .common import (
    EvalError,
    PathGuard,
    UnsafePathError,
    git_identity,
    hash_tree,
    read_json,
    resolve_absolute,
    sha256_file,
    software_versions,
    utc_now,
    write_json_atomic,
)

PROTOCOL_VERSION = 1
BASELINE_SHA = "a27a176f42755432478a33db5af0dcf6a1649f97"
BASELINE_REF = ".review/upstream"
SUPPORTED_GROUPS = ("A", "A2", "A_prime", "A_prime2", "B")
UNSUPPORTED_GROUPS = {
    "C": "reflective polish requires the WP2 implementation",
    "C_keep": "reflective polish with clean batches kept requires the WP2 implementation",
    "D": "heterogeneous reflect.body routing requires the WP2 implementation",
    "E": "the optional three-draft reference group is not implemented",
}

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "variants"
TEMPLATE_FILES = {
    "A": "A.yaml",
    "A2": "A.yaml",
    "A_prime": "A_prime.yaml",
    "A_prime2": "A_prime.yaml",
    "B": "B.yaml",
}


@dataclass(frozen=True)
class GroupSpec:
    key: str
    arm: str
    purpose: str
    rerun: bool

    @property
    def code_role(self) -> str:
        return "baseline" if self.arm == "baseline" else "candidate"


GROUP_SPECS = {
    "A": GroupSpec("A", "baseline", "baseline primary", False),
    "A2": GroupSpec("A2", "baseline", "baseline reproducibility rerun", True),
    "A_prime": GroupSpec("A_prime", "candidate", "candidate primary", False),
    "A_prime2": GroupSpec("A_prime2", "candidate", "candidate reproducibility rerun", True),
    "B": GroupSpec("B", "candidate", "candidate with polish disabled", False),
}


def require_group_supported(group: str) -> None:
    """Fail explicitly for groups whose implementation belongs to WP2."""
    reason = UNSUPPORTED_GROUPS.get(group)
    if reason is not None:
        raise EvalError(f"Group {group} is unsupported: {reason}")
    if group not in GROUP_SPECS:
        raise EvalError(f"Unknown evaluation group: {group}")


def load_group_template(group: str) -> dict[str, Any]:
    """Load the checked-in configuration template used by one group."""
    require_group_supported(group)
    path = TEMPLATE_DIR / TEMPLATE_FILES[group]
    if not path.is_file():
        raise EvalError(f"Variant configuration template is missing: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("pipeline"), dict):
        raise EvalError(f"Variant configuration template is invalid: {path}")
    return data


def load_user_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a user configuration without rewriting it and reject literal secrets."""
    source = Path(path)
    if not source.is_file():
        raise EvalError(f"User configuration does not exist: {source}")
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise EvalError(f"User configuration is not valid YAML: {error}") from None
    if not isinstance(data, dict):
        raise EvalError("User configuration must be a YAML mapping")
    for name, provider in (data.get("llm") or {}).get("providers", {}).items():
        if not isinstance(provider, dict):
            raise EvalError(f"llm.providers.{name} must be a mapping")
        for key in ("api_key", "apikey", "token", "secret", "password", "credentials"):
            if provider.get(key):
                raise EvalError(
                    f"llm.providers.{name}.{key} must not contain a literal credential; "
                    "use api_key_env with an environment variable name"
                )
    return data


_REVIEW_OVERRIDE_KEYS = (
    "review",
    "review_autofix",
    "review_concurrency",
    "review_output_retries",
    "review_agent_max_evidence_rounds",
    "review_fix_max_rounds",
)


def variant_config(
    group: str, state_dir: str | os.PathLike[str], *, base: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Bind a disposable state path; overlay only polish and Review settings.

    A user configuration is preserved otherwise, so language, model routing and
    unrelated pipeline options reach both preparation and translation unchanged.
    """
    template = load_group_template(group)
    if base is None:
        config = copy.deepcopy(template)
    else:
        config = copy.deepcopy(base)
        pipeline = config.setdefault("pipeline", {})
        pipeline["polish"] = template["pipeline"]["polish"]
        for key in _REVIEW_OVERRIDE_KEYS:
            if key in template["pipeline"]:
                pipeline[key] = template["pipeline"][key]
    config.setdefault("paths", {})["state_dir"] = Path(state_dir).as_posix()
    return config


def _routes(polish: bool) -> dict[str, Any]:
    return {
        "status": "unknown",
        "reason": "the offline adapter does not expose effective provider routes",
        "translation.body": "unknown",
        "polish.body": "unknown" if polish else "disabled",
        "reflect.body": "unsupported",
    }


def _route_fingerprints() -> dict[str, Any]:
    return {
        "status": "unknown",
        "reason": "wenyi models explain was not invoked for the offline adapter",
    }


def _template_polish(group: str) -> bool:
    return bool(load_group_template(group)["pipeline"].get("polish"))


def credential_env_names(config_path: str | os.PathLike[str]) -> list[str]:
    """Return credential environment-variable names referenced by a config.

    Only names are returned. For presets the provider adapter's default variable
    name is added so a normal provider run can read its existing credential.
    """
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    names = []
    for provider in (data.get("llm") or {}).get("providers", {}).values():
        if isinstance(provider, dict):
            name = provider.get("api_key_env")
            if isinstance(name, str) and name:
                names.append(name)
    try:
        from wenyi_core.llm.configuration import LLMConfig
        from wenyi_core.llm.registry import provider_spec

        resolved = LLMConfig.model_validate(data.get("llm") or {})
        for provider in resolved.providers.values():
            name = provider.api_key_env
            if not name:
                name = getattr(
                    provider_spec(provider.kind).adapter_type(), "default_api_key_env", None
                )
            if isinstance(name, str) and name:
                names.append(name)
    except (ImportError, ValueError, TypeError):
        pass
    return sorted(set(names))


def _runtime_binding(source: Path, binding: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve the command that will actually execute for one arm.

    The offline adapter is only used when it was explicitly selected; a normal
    evaluation keeps the production factory and any user-supplied executable.
    """
    resolved = dict(binding or {})
    adapter = resolved.get("adapter", "provider")
    if adapter not in {"provider", "offline"}:
        raise EvalError(f"Unknown evaluation adapter: {adapter}")
    if adapter == "offline":
        executable = str(resolved.get("executable") or sys.executable)
        argv_prefix = [
            "-m",
            "wenyi_eval.offline_cli",
            "--code-source",
            str(source),
            "--",
        ]
    else:
        executable = resolved.get("executable")
        argv_prefix = list(resolved.get("argv_prefix") or [])
        if not executable:
            raise EvalError(
                "Provider mode requires an explicit executable that can run the selected "
                "code source (for example a virtualenv's python with argv -m wenyi_cli)"
            )
    if not Path(executable).exists():
        raise EvalError(f"Group executable does not exist: {executable}")
    if not argv_prefix and Path(executable).name.lower().startswith("python"):
        argv_prefix = ["-m", "wenyi_cli"]
    return {
        "adapter": adapter,
        "executable": executable,
        "argv_prefix": argv_prefix,
        "command": [executable, *argv_prefix],
    }


def build_group_plans(
    root: Path,
    *,
    baseline_source: Path,
    candidate_source: Path,
    baseline_expected_sha: str = BASELINE_SHA,
    candidate_expected_sha: str,
    diagnostic: bool,
    requested_groups: Iterable[str] | None = None,
    baseline_binding: dict[str, Any] | None = None,
    candidate_binding: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve one explicit binding and one isolated path set per group."""
    requested = tuple(requested_groups or SUPPORTED_GROUPS)
    for group in requested:
        require_group_supported(group)
    if set(requested) != set(SUPPORTED_GROUPS):
        raise EvalError(
            "WP1a requires exactly the groups "
            f"{sorted(SUPPORTED_GROUPS)}; requested {sorted(set(requested))}"
        )
    guard = PathGuard(root)
    plans: dict[str, dict[str, Any]] = {}
    for key, spec in GROUP_SPECS.items():
        source = baseline_source if spec.arm == "baseline" else candidate_source
        expected = baseline_expected_sha if spec.arm == "baseline" else candidate_expected_sha
        identity = git_identity(source)
        status = _binding_status(identity, expected, diagnostic)
        polish = _template_polish(key)
        runtime = _runtime_binding(
            Path(identity.code_source),
            baseline_binding if spec.arm == "baseline" else candidate_binding,
        )
        plans[key] = {
            "label": spec.purpose,
            "arm": spec.arm,
            "rerun": spec.rerun,
            "polish": polish,
            "code_source": identity.code_source,
            "git": identity.as_dict(),
            "expected_head_sha": expected,
            "binding_status": status,
            "adapter": runtime["adapter"],
            "executable": runtime["executable"],
            "argv_prefix": runtime["argv_prefix"],
            "command": runtime["command"],
            "prompt_policy_hashes": _policy_hashes(Path(identity.code_source)),
            "routes": _routes(polish),
            "route_fingerprints": _route_fingerprints(),
            "config": str(guard.child(key, "config.yaml")),
            "cwd": str(guard.child(key, "cwd")),
            "state_dir": str(guard.child(key, "state")),
            "output_dir": str(guard.child(key, "output")),
            "cache_dir": str(guard.child(key, "cache")),
            "progress_path": str(guard.child(key, "receipts", "progress.json")),
            "result_path": str(guard.child(key, "receipts", "result.json")),
            "status_path": str(guard.child("records", f"{key}.status.txt")),
            "inspect_path": str(guard.child("records", f"{key}.inspect-run.txt")),
            "stdout_path": str(guard.child(key, "logs", "translate.stdout.log")),
            "stderr_path": str(guard.child(key, "logs", "translate.stderr.log")),
        }
    if plans["A"]["code_source"] == plans["A_prime"]["code_source"]:
        raise EvalError(
            "Baseline and candidate must use distinct code sources; "
            "the current working tree must never be imported as the old baseline."
        )
    return plans


def _binding_status(identity, expected: str, diagnostic: bool) -> str:
    if expected == "unfrozen":
        return "unfrozen"
    if identity.head_sha != expected:
        return "mismatch"
    if identity.dirty:
        return "dirty"
    return "clean"


def _policy_hashes(code_source: Path) -> dict[str, str]:
    data = code_source / "packages" / "core" / "wenyi_core" / "i18n" / "data"
    result: dict[str, str] = {}
    for name in ("tasks", "rules", "languages"):
        path = data / name
        result[name] = hash_tree(path) if path.is_dir() else "missing"
    return result


def write_variant_configs(
    root: Path, *, base_config: dict[str, Any] | None = None
) -> dict[str, str]:
    """Instantiate each group with isolated paths and a shared user configuration."""
    guard = PathGuard(root)
    hashes: dict[str, str] = {}
    for key in SUPPORTED_GROUPS:
        config = variant_config(key, guard.child(key, "state"), base=base_config)
        path = guard.child(key, "config.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        hashes[key] = sha256_file(path)
    return hashes


def build_manifest(
    *,
    root: Path,
    runs_dir: Path,
    eval_id: str,
    book: str,
    source_path: Path,
    source_format: str,
    language: dict[str, str],
    fixture: dict[str, Any],
    plans: dict[str, dict[str, Any]],
    config_hashes: dict[str, str],
    mode: str,
    diagnostic: bool,
    repo_root: Path,
    seed: int,
    prices: dict[str, Any],
    preparation_entry: str,
    unknown_notes: list[str],
    user_config: dict[str, Any] | None = None,
    sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest from observed identities and explicit unknowns."""
    guard = PathGuard(root)
    source = resolve_absolute(source_path, label="book source")
    guard.ensure(source, label="book source")
    versions = software_versions()
    adapter = plans["A_prime"]["adapter"]
    groups: dict[str, Any] = {}
    for key, plan in plans.items():
        entry = dict(plan)
        entry["config_sha256"] = config_hashes[key]
        groups[key] = entry
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "eval_id": eval_id,
        "book": book,
        "created_at": utc_now(),
        "mode": mode,
        "diagnostic": diagnostic,
        "adapter": adapter,
        "comparability": {
            "status": "not_comparable" if diagnostic else "pending",
            "reason": (
                "Offline adapter replaces the external model boundary; this run validates "
                "the harness and never supports quality or cost conclusions."
                if diagnostic
                else None
            ),
        },
        "root": str(root),
        "runs_dir": str(runs_dir),
        "repo": {
            "root": str(repo_root),
            "baseline_sha": plans["A"]["expected_head_sha"],
            "baseline_ref": BASELINE_REF,
        },
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "format": source_format,
            "size": source.stat().st_size,
            "language": dict(language),
        },
        "fixture": fixture,
        "user_config": (
            {"path": str(user_config.get("path")), "sha256": user_config.get("sha256")}
            if user_config
            else None
        ),
        "sampling": sampling or {},
        "groups": groups,
        "preparation": {
            "entry": preparation_entry,
            "producer_group": "A_prime",
            "snapshot": str(guard.child("_prepared")),
            "cost": "unknown",
            "cost_reason": (
                "offline preparation cost is not a provider cost"
                if adapter == "offline"
                else "preparation usage is recorded separately when the selected CLI exposes it"
            ),
        },
        "llm": {
            "external_provider": adapter == "provider",
            "adapter": (
                "wenyi_eval.offline_llm (deterministic subprocess bootstrap)"
                if adapter == "offline"
                else "selected product factory (no evaluation monkeypatch)"
            ),
            "preset": "configured" if adapter == "provider" else "fake",
            "routes": plans["A_prime"]["routes"],
            "route_fingerprints": plans["A_prime"]["route_fingerprints"],
            "model_params": {
                "temperature": "provider default",
                "max_output_tokens": "provider default",
                "tiers": ["strong", "cheap", "fast"],
            },
            "seed": seed,
        },
        "prices": prices,
        "software": versions,
        "unknowns": unknown_notes,
    }
    validate_manifest(manifest, repo_root=repo_root)
    return manifest


def validate_manifest(manifest: dict[str, Any], *, repo_root: Path) -> None:
    """Reject unsafe paths, duplicate ownership and unresolved group bindings."""
    root = resolve_absolute(manifest.get("root", ""), label="manifest root")
    guard = PathGuard(root)
    required_groups = set(SUPPORTED_GROUPS)
    groups = manifest.get("groups")
    if not isinstance(groups, dict) or set(groups) != required_groups:
        raise EvalError(f"Manifest must contain exactly groups {sorted(required_groups)}")
    forbidden = [
        repo_root / "state",
        repo_root / "output",
        repo_root / "logs",
        repo_root / "apps",
    ]
    path_fields = (
        "config",
        "cwd",
        "state_dir",
        "output_dir",
        "cache_dir",
        "progress_path",
        "result_path",
        "status_path",
        "inspect_path",
        "stdout_path",
        "stderr_path",
    )
    ownership: dict[tuple[str, str], str] = {}
    for key, group in groups.items():
        for field in path_fields:
            value = group.get(field)
            if not isinstance(value, str):
                raise EvalError(f"Group {key} lacks explicit {field}")
            path = guard.ensure(value, label=f"groups.{key}.{field}")
            _reject_forbidden(path, forbidden, label=f"groups.{key}.{field}")
            if field in {"state_dir", "result_path", "progress_path", "config", "cwd"}:
                marker = (field, str(path).casefold())
                if marker in ownership:
                    raise EvalError(f"Groups {ownership[marker]} and {key} share {field}: {path}")
                ownership[marker] = key
        source = resolve_absolute(group.get("code_source", ""), label=f"groups.{key}.code_source")
        expected = group.get("expected_head_sha")
        if not expected:
            raise EvalError(f"Group {key} has no expected code SHA")
        actual = group.get("git", {}).get("head_sha")
        if expected != "unfrozen" and actual != expected:
            raise EvalError(f"Group {key} code SHA {actual} does not match expected {expected}")
        if group.get("binding_status") not in {"clean", "dirty", "unfrozen"}:
            raise EvalError(f"Group {key} binding status is unresolved")
        if group.get("adapter") not in {"provider", "offline"}:
            raise EvalError(f"Group {key} must select the provider or offline adapter")
        if not isinstance(group.get("executable"), str) or not group["executable"]:
            raise EvalError(f"Group {key} has no executable binding")
        if not isinstance(group.get("argv_prefix"), list):
            raise EvalError(f"Group {key} has no argv binding")
        if group.get("arm") == "baseline" and actual != manifest["repo"]["baseline_sha"]:
            raise EvalError(f"Baseline group {key} is not at {manifest['repo']['baseline_sha']}")
        if not isinstance(group.get("polish"), bool):
            raise EvalError(f"Group {key} must declare whether polish is enabled")
        if key == "B" and group["polish"]:
            raise EvalError("Group B must have polish disabled")
        for field in ("routes", "route_fingerprints"):
            if not isinstance(group.get(field), dict):
                raise EvalError(f"Group {key} lacks explicit {field}")
        if group["routes"].get("status") not in {"known", "unknown"}:
            raise EvalError(f"Group {key} routes status is unresolved")
        if group["route_fingerprints"].get("status") not in {"known", "unknown"}:
            raise EvalError(f"Group {key} route fingerprint status is unresolved")
    source = manifest.get("source", {})
    source_path = guard.ensure(source.get("path", ""), label="source.path")
    if source.get("sha256") != sha256_file(source_path):
        raise EvalError("Manifest source hash does not match the file on disk")
    preparation = manifest.get("preparation", {})
    guard.ensure(preparation.get("snapshot", ""), label="preparation.snapshot")
    if manifest.get("diagnostic") and not manifest.get("comparability", {}).get("reason"):
        raise EvalError("Diagnostic manifests must state why results are not comparable")
    if manifest.get("prices", {}).get("status") not in {"known", "unknown", "partial"}:
        raise EvalError("Prices must carry an explicit known/unknown status")
    llm = manifest.get("llm", {})
    for field in ("routes", "route_fingerprints"):
        if not isinstance(llm.get(field), dict):
            raise EvalError(f"Manifest llm metadata lacks explicit {field}")


def require_frozen_bindings(manifest: dict[str, Any]) -> None:
    """Reject dirty or unpinned code before any metadata or model process starts."""
    unresolved = []
    for key, group in manifest.get("groups", {}).items():
        status = group.get("binding_status")
        if status != "clean":
            unresolved.append(f"{key}:{status}")
        if group.get("expected_head_sha") == "unfrozen":
            unresolved.append(f"{key}:unfrozen")
    if unresolved:
        raise EvalError(
            "Real runs require clean, pinned commits; unresolved bindings: "
            + ", ".join(sorted(unresolved))
        )


def require_real_ready(manifest: dict[str, Any]) -> None:
    """Reject a real comparative run that would use dirty or unpinned code."""
    if manifest.get("diagnostic"):
        raise EvalError("Refusing to treat a diagnostic manifest as a real comparison")
    require_frozen_bindings(manifest)
    unresolved = []
    for key, group in manifest.get("groups", {}).items():
        if (group.get("routes") or {}).get("status") != "known":
            unresolved.append(f"{key}:routes_unknown")
        if (group.get("route_fingerprints") or {}).get("status") != "known":
            unresolved.append(f"{key}:route_fingerprint_unknown")
    if unresolved:
        raise EvalError(
            "Real runs require clean, pinned commits; unresolved bindings: "
            + ", ".join(sorted(unresolved))
        )
    # Missing or partial price rules leave cost unknown; they must not disable
    # non-cost tooling such as extraction, sampling or reporting.
    if manifest.get("prices", {}).get("status") not in {"known", "partial", "unknown"}:
        raise EvalError("Prices must carry an explicit known/unknown status")


def _reject_forbidden(path: Path, forbidden: list[Path], *, label: str) -> None:
    for item in forbidden:
        try:
            path.relative_to(item.resolve())
        except ValueError:
            continue
        raise UnsafePathError(f"{label} points at non-disposable user data: {path}")


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise EvalError("Unsupported manifest protocol version")
    return manifest


def write_manifest(root: Path, manifest: dict[str, Any]) -> Path:
    path = Path(root) / "manifest.json"
    if path.exists():
        raise EvalError(f"Refusing to overwrite an existing evaluation manifest: {path}")
    write_json_atomic(path, manifest)
    return path
