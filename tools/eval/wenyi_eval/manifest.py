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
# Frozen WP0 checkout. A_prime/A_prime2/B compare against this clean source,
# never against the mutable working tree that carries WP2.
WP0_SHA = "4267537c9f28208606282f5ec085d723c3c270bc"
WP0_REF = ".review/wp0-baseline"
# Sources are distinct bindings: upstream baseline, frozen WP0, current workspace.
SOURCE_ROLES = ("baseline", "wp0", "candidate")
REFLECTIVE_GROUPS = ("C", "C_keep", "D")
UNSUPPORTED_GROUPS = {"E": "the optional three-draft reference group is not implemented"}

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "variants"
TEMPLATE_FILES = {
    "A": "A.yaml",
    "A2": "A.yaml",
    "A_prime": "A_prime.yaml",
    "A_prime2": "A_prime.yaml",
    "B": "B.yaml",
    "C": "C.yaml",
    "C_keep": "C_keep.yaml",
    "D": "D.yaml",
}


@dataclass(frozen=True)
class GroupSpec:
    key: str
    arm: str
    source: str
    purpose: str
    rerun: bool

    @property
    def code_role(self) -> str:
        return self.source


GROUP_SPECS = {
    "A": GroupSpec("A", "baseline", "baseline", "baseline primary", False),
    "A2": GroupSpec("A2", "baseline", "baseline", "baseline reproducibility rerun", True),
    "A_prime": GroupSpec("A_prime", "candidate", "wp0", "WP0 baseline primary", False),
    "A_prime2": GroupSpec(
        "A_prime2", "candidate", "wp0", "WP0 baseline reproducibility rerun", True
    ),
    "B": GroupSpec("B", "candidate", "wp0", "WP0 baseline with polish disabled", False),
    "C": GroupSpec(
        "C", "candidate", "candidate", "reflective polish, clean batches polished", False
    ),
    "C_keep": GroupSpec(
        "C_keep", "candidate", "candidate", "reflective polish, clean batches kept", False
    ),
    "D": GroupSpec(
        "D", "candidate", "candidate", "reflective polish with a heterogeneous reflect route", False
    ),
}
SUPPORTED_GROUPS = tuple(GROUP_SPECS)
# The historical five-group evaluation remains the default, so existing commands
# keep planning exactly the runs they always planned.
DEFAULT_GROUPS = ("A", "A2", "A_prime", "A_prime2", "B")


def require_group_supported(group: str) -> None:
    """Fail explicitly for groups that are not implemented."""
    reason = UNSUPPORTED_GROUPS.get(group)
    if reason is not None:
        raise EvalError(f"Group {group} is unsupported: {reason}")
    if group not in GROUP_SPECS:
        raise EvalError(f"Unknown evaluation group: {group}")


def normalize_groups(requested: Iterable[str] | None) -> tuple[str, ...]:
    """Resolve an explicit selection; defaults preserve the five-group workflow."""
    if requested is None:
        return DEFAULT_GROUPS
    if isinstance(requested, str):
        values = [part.strip() for part in requested.split(",")]
    else:
        values = [str(part).strip() for part in requested]
    values = [value for value in values if value]
    if not values:
        raise EvalError("At least one evaluation group must be selected")
    for group in values:
        require_group_supported(group)
    selected = set(values)
    return tuple(group for group in SUPPORTED_GROUPS if group in selected)


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
# These fields only exist in the WP2 code. They must never reach a baseline or
# WP0 configuration, whose pinned config model rejects unknown pipeline keys.
_WP2_PIPELINE_KEYS = ("reflective_polish", "reflect_min_severity", "reflect_clean_batches")


def variant_config(
    group: str,
    state_dir: str | os.PathLike[str],
    *,
    base: dict[str, Any] | None = None,
    reflect_model: str | None = None,
) -> dict[str, Any]:
    """Bind a disposable state path and the group's treatment switches.

    A user configuration is preserved otherwise, so language, model routing and
    unrelated pipeline options reach both preparation and translation unchanged.
    Control groups never inherit reflection from the user configuration, and
    legacy code sources never receive WP2-only fields.
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
    pipeline = config.setdefault("pipeline", {})
    for key in _WP2_PIPELINE_KEYS:
        pipeline.pop(key, None)
    llm = config.setdefault("llm", {})
    routes = llm.get("routes")
    if isinstance(routes, dict):
        routes.pop("reflect.body", None)

    if group in REFLECTIVE_GROUPS:
        if pipeline.get("translation_mode", "standard") != "standard":
            raise EvalError(
                f"Group {group} requires pipeline.translation_mode=standard; "
                "reflective polish replaces ordinary polish only in that mode"
            )
        pipeline["reflective_polish"] = True
        pipeline["reflect_min_severity"] = template["pipeline"]["reflect_min_severity"]
        pipeline["reflect_clean_batches"] = template["pipeline"]["reflect_clean_batches"]
        if group == "D":
            if not reflect_model:
                raise EvalError(
                    "Group D requires --heterogeneous-reflect-model naming a model profile "
                    "defined by the user configuration; no model ID is invented"
                )
            models = llm.get("models")
            if not isinstance(models, dict) or reflect_model not in models:
                raise EvalError(
                    f"Group D reflect model profile {reflect_model!r} is not defined in "
                    "llm.models; supply a valid routing/model configuration"
                )
            routes = llm.setdefault("routes", {})
            routes["reflect.body"] = {"model": reflect_model}
    if isinstance(llm.get("routes"), dict) and not llm["routes"]:
        llm.pop("routes")
    config.setdefault("paths", {})["state_dir"] = Path(state_dir).as_posix()
    return config


def _routes(polish: bool, *, reflection: dict[str, Any] | None = None) -> dict[str, Any]:
    reflect_status = "not_enabled"
    if reflection and reflection.get("enabled"):
        reflect_status = "unknown"
    return {
        "status": "unknown",
        "reason": "the offline adapter does not expose effective provider routes",
        "translation.body": "unknown",
        "polish.body": "unknown" if polish else "disabled",
        "reflect.body": reflect_status,
    }


def _route_fingerprints() -> dict[str, Any]:
    return {
        "status": "unknown",
        "reason": "wenyi models explain was not invoked for the offline adapter",
    }


def _template_polish(group: str) -> bool:
    return bool(load_group_template(group)["pipeline"].get("polish"))


def reflection_plan(group: str, config: dict[str, Any]) -> dict[str, Any]:
    """Describe the configured reflection treatment and prove route separation.

    ``C``/``C_keep`` keep the product default where ``reflect.body`` inherits
    ``polish.body``. ``D`` must resolve an explicit heterogeneous profile; the
    separation is proven here from resolved route identities, never by relabeling
    an unknown or by inventing a model ID.
    """
    pipeline = config.get("pipeline") or {}
    if group not in REFLECTIVE_GROUPS:
        return {
            "enabled": False,
            "status": "not_enabled",
            "clean_action": None,
            "min_severity": None,
            "route_separation": {"status": "not_applicable"},
            "configured_routes": None,
        }
    from wenyi_core.llm.configuration import LLMConfig
    from wenyi_core.llm.routing import resolve_routes

    try:
        resolved = resolve_routes(LLMConfig.model_validate(config.get("llm") or {}))
    except Exception as error:
        raise EvalError(
            f"Group {group} reflection routing cannot be resolved: {type(error).__name__}: {error}"
        ) from None
    reflect = resolved["reflect.body"]
    polish = resolved["polish.body"]
    configured = {
        operation: {
            "profile": resolved[operation].profile,
            "model": resolved[operation].model,
            "provider": resolved[operation].provider,
            "provider_kind": resolved[operation].provider_kind,
            "origin": resolved[operation].origin,
            "inference_fingerprint": resolved[operation].fingerprint,
        }
        for operation in ("reflect.body", "polish.body", "translation.body")
    }
    if group == "D":
        if reflect.origin == "inherits polish.body":
            raise EvalError(
                "Group D requires an explicit reflect.body route; it must not inherit polish.body"
            )
        if reflect.provider == polish.provider and reflect.model == polish.model:
            raise EvalError(
                "Group D requires a different model family for reflect.body; the supplied "
                "configuration resolves the same provider and model as polish.body"
            )
        separation = {
            "status": "distinct",
            "reflect_profile": reflect.profile,
            "reflect_model": reflect.model,
            "reflect_provider": reflect.provider,
            "polish_profile": polish.profile,
            "polish_model": polish.model,
            "polish_provider": polish.provider,
            "reflect_inference_fingerprint": reflect.fingerprint,
            "polish_inference_fingerprint": polish.fingerprint,
            "family_heterogeneity": "operator-declared; verified as a distinct model identity",
        }
    else:
        if (
            reflect.origin != "inherits polish.body"
            or reflect.profile != polish.profile
            or reflect.model != polish.model
            or reflect.provider != polish.provider
        ):
            raise EvalError(f"Group {group} must let reflect.body inherit its polish.body route")
        separation = {
            "status": "inherited",
            "reflect_profile": reflect.profile,
            "reflect_model": reflect.model,
            "reflect_provider": reflect.provider,
            "polish_profile": polish.profile,
            "polish_model": polish.model,
            "polish_provider": polish.provider,
            "reflect_inference_fingerprint": reflect.fingerprint,
        }
    return {
        "enabled": True,
        "status": "enabled",
        "clean_action": pipeline.get("reflect_clean_batches"),
        "min_severity": pipeline.get("reflect_min_severity"),
        "route_separation": separation,
        "configured_routes": configured,
    }


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
    wp0_source: Path | None = None,
    baseline_expected_sha: str = BASELINE_SHA,
    wp0_expected_sha: str | None = None,
    candidate_expected_sha: str,
    diagnostic: bool,
    requested_groups: Iterable[str] | None = None,
    baseline_binding: dict[str, Any] | None = None,
    candidate_binding: dict[str, Any] | None = None,
    wp0_binding: dict[str, Any] | None = None,
    base_config: dict[str, Any] | None = None,
    reflect_model: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve explicit bindings and one isolated path set per selected group.

    ``A``/``A2`` bind the upstream baseline, ``A_prime``/``A_prime2``/``B`` bind
    the frozen WP0 checkout, and ``C``/``C_keep``/``D`` bind the current
    workspace. Selecting a WP0 group requires its own explicit source; missing
    bindings must never relabel the candidate working tree as the frozen WP0.
    """
    requested = normalize_groups(requested_groups)
    if wp0_source is None and any(GROUP_SPECS[group].source == "wp0" for group in requested):
        raise EvalError("An explicit frozen WP0 source is required for A_prime/A_prime2/B")
    wp0_expected_sha = wp0_expected_sha or WP0_SHA
    sources = {
        "baseline": (baseline_source, baseline_expected_sha, baseline_binding),
        "wp0": (wp0_source, wp0_expected_sha, wp0_binding),
        "candidate": (candidate_source, candidate_expected_sha, candidate_binding),
    }
    guard = PathGuard(root)
    plans: dict[str, dict[str, Any]] = {}
    for key in requested:
        spec = GROUP_SPECS[key]
        source, expected, binding = sources[spec.source]
        identity = git_identity(source)
        status = _binding_status(identity, expected, diagnostic)
        polish = _template_polish(key)
        config = variant_config(
            key,
            guard.child(key, "state"),
            base=base_config,
            reflect_model=reflect_model,
        )
        reflection = reflection_plan(key, config)
        runtime = _runtime_binding(Path(identity.code_source), binding)
        plans[key] = {
            "label": spec.purpose,
            "arm": spec.arm,
            "source_role": spec.source,
            "rerun": spec.rerun,
            "polish": polish,
            "reflection": reflection,
            "code_source": identity.code_source,
            "git": identity.as_dict(),
            "expected_head_sha": expected,
            "binding_status": status,
            "adapter": runtime["adapter"],
            "executable": runtime["executable"],
            "argv_prefix": runtime["argv_prefix"],
            "command": runtime["command"],
            "prompt_policy_hashes": _policy_hashes(Path(identity.code_source)),
            "routes": _routes(polish, reflection=reflection),
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
    distinct = {
        "baseline": str(Path(baseline_source).resolve()),
        "candidate": str(Path(candidate_source).resolve()),
    }
    if wp0_source is not None:
        distinct["wp0"] = str(Path(wp0_source).resolve())
    if len(set(distinct.values())) != len(distinct):
        raise EvalError(
            "Baseline, WP0 and candidate must use distinct code sources; a frozen checkout "
            "must never be replaced by the current working tree."
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
    root: Path,
    *,
    base_config: dict[str, Any] | None = None,
    groups: Iterable[str] | None = None,
    reflect_model: str | None = None,
) -> dict[str, str]:
    """Instantiate each selected group with isolated paths and shared user configuration."""
    guard = PathGuard(root)
    selected = normalize_groups(groups)
    hashes: dict[str, str] = {}
    for key in selected:
        config = variant_config(
            key,
            guard.child(key, "state"),
            base=base_config,
            reflect_model=reflect_model,
        )
        path = guard.child(key, "config.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        hashes[key] = sha256_file(path)
    return hashes


def preparation_group(groups: Iterable[str]) -> str:
    """Choose the producer for the shared preparation snapshot.

    Prefer a frozen baseline configuration whose LLM graph matches the controls,
    and use ``D`` only when it is the sole group because its explicit route
    changes the persisted chapter-digest preparation contract.
    """
    available = set(groups)
    for key in ("A_prime", "A_prime2", "B", "C", "C_keep", "A", "A2", "D"):
        if key in available:
            return key
    raise EvalError("Cannot choose a preparation producer without a selected group")


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
    producer = preparation_group(plans)
    adapter = plans[producer]["adapter"]
    groups: dict[str, Any] = {}
    for key, plan in plans.items():
        entry = dict(plan)
        entry["config_sha256"] = config_hashes[key]
        groups[key] = entry
    baseline_plan = plans.get("A") or next(
        (plan for plan in plans.values() if plan["source_role"] == "baseline"), None
    )
    wp0_plan = plans.get("A_prime") or next(
        (plan for plan in plans.values() if plan["source_role"] == "wp0"), None
    )
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
                "Diagnostic adapter or dirty/unfrozen code replaces a comparable model boundary; "
                "this run validates the harness and never supports quality or cost conclusions."
                if diagnostic
                else None
            ),
        },
        "root": str(root),
        "runs_dir": str(runs_dir),
        "repo": {
            "root": str(repo_root),
            "baseline_sha": baseline_plan["expected_head_sha"] if baseline_plan else BASELINE_SHA,
            "baseline_ref": BASELINE_REF,
            "wp0_sha": wp0_plan["expected_head_sha"] if wp0_plan else WP0_SHA,
            "wp0_ref": WP0_REF,
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
            "producer_group": producer,
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
            "routes": plans[producer]["routes"],
            "route_fingerprints": plans[producer]["route_fingerprints"],
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
    groups = manifest.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise EvalError("Manifest must contain at least one evaluation group")
    unknown = set(groups) - set(SUPPORTED_GROUPS)
    if unknown:
        raise EvalError(f"Manifest contains unsupported groups: {sorted(unknown)}")
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
        if group.get("source_role") not in SOURCE_ROLES:
            raise EvalError(f"Group {key} has no explicit source role")
        if group.get("source_role") == "baseline" and actual != manifest["repo"]["baseline_sha"]:
            raise EvalError(f"Baseline group {key} is not at {manifest['repo']['baseline_sha']}")
        if group.get("source_role") == "wp0" and actual != manifest["repo"]["wp0_sha"]:
            raise EvalError(f"WP0 group {key} is not at {manifest['repo']['wp0_sha']}")
        if not isinstance(group.get("polish"), bool):
            raise EvalError(f"Group {key} must declare whether polish is enabled")
        if key == "B" and group["polish"]:
            raise EvalError("Group B must have polish disabled")
        reflection = group.get("reflection")
        if not isinstance(reflection, dict) or not isinstance(reflection.get("enabled"), bool):
            raise EvalError(f"Group {key} must declare whether reflection is enabled")
        if key in REFLECTIVE_GROUPS:
            if not group["polish"] or not reflection["enabled"]:
                raise EvalError(f"Group {key} must enable reflection with polish")
            if reflection.get("clean_action") not in {"polish", "keep"}:
                raise EvalError(f"Group {key} must declare its clean-batch strategy")
            if reflection.get("min_severity") not in {"major", "critical"}:
                raise EvalError(f"Group {key} must declare its critique severity threshold")
            separation = (reflection.get("route_separation") or {}).get("status")
            expected_separation = "distinct" if key == "D" else "inherited"
            if separation != expected_separation:
                raise EvalError(
                    f"Group {key} reflection route separation is {separation!r}, "
                    f"expected {expected_separation!r}"
                )
        elif reflection["enabled"]:
            raise EvalError(f"Control group {key} must not enable reflective polish")
        for field in ("routes", "route_fingerprints"):
            if not isinstance(group.get(field), dict):
                raise EvalError(f"Group {key} lacks explicit {field}")
        if group["routes"].get("status") not in {"known", "unknown"}:
            raise EvalError(f"Group {key} routes status is unresolved")
        if group["route_fingerprints"].get("status") not in {"known", "unknown"}:
            raise EvalError(f"Group {key} route fingerprint status is unresolved")
        if key == "B" and group["routes"].get("polish.body") not in {"disabled", "unknown"}:
            raise EvalError("Group B must record polish as disabled or unknown")
    source = manifest.get("source", {})
    source_path = guard.ensure(source.get("path", ""), label="source.path")
    if source.get("sha256") != sha256_file(source_path):
        raise EvalError("Manifest source hash does not match the file on disk")
    preparation = manifest.get("preparation", {})
    guard.ensure(preparation.get("snapshot", ""), label="preparation.snapshot")
    if preparation.get("producer_group") not in groups:
        raise EvalError("Preparation producer must be one of the selected groups")
    if manifest.get("diagnostic") and not manifest.get("comparability", {}).get("reason"):
        raise EvalError("Diagnostic manifests must state why results are not comparable")
    if manifest.get("prices", {}).get("status") not in {"known", "unknown", "partial"}:
        raise EvalError("Prices must carry an explicit known/unknown status")
    llm = manifest.get("llm", {})
    for field in ("routes", "route_fingerprints"):
        if not isinstance(llm.get(field), dict):
            raise EvalError(f"Manifest llm metadata lacks explicit {field}")

    # Control groups must carry no WP2-only fields into legacy code.
    for key, group in groups.items():
        config_path = Path(group["config"])
        if not config_path.is_file():
            continue
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        pipeline = data.get("pipeline") or {}
        present = [name for name in _WP2_PIPELINE_KEYS if name in pipeline]
        if key not in REFLECTIVE_GROUPS and present:
            raise EvalError(
                f"Control group {key} configuration contains WP2-only fields: {present}"
            )
        if key not in REFLECTIVE_GROUPS and (data.get("llm") or {}).get("routes", {}).get(
            "reflect.body"
        ):
            raise EvalError(f"Control group {key} configuration routes reflect.body")


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
