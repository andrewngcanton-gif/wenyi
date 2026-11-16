"""Normal-path tests: the production factory runs behind only a local HTTP stub."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from wenyi_eval import offline_llm
from wenyi_eval.common import EvalError
from wenyi_eval.run_variants import (
    _run_metadata_command,
    redact_metadata,
    run_child,
)


class _StubHandler(BaseHTTPRequestHandler):
    server_version = "WenyiEvalStub/1.0"

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages") or []
        content = offline_llm.handler(messages, "strong", False)
        body = json.dumps(
            {
                "id": "stub-request",
                "object": "chat.completion",
                "created": 0,
                "model": payload.get("model", "stub"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        ).encode("utf-8")
        self.server.requests += 1  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        return


@pytest.fixture()
def stub_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    server.requests = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _config(tmp_path: Path, base_url: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "language": {"source": "en", "target": "zh"},
                "llm": {
                    "preset": None,
                    "providers": {
                        "stub": {
                            "kind": "openai-compatible",
                            "base_url": base_url,
                            "api_key_env": "WENYI_TEST_STUB_KEY",
                        }
                    },
                    "models": {
                        "stub-model": {"provider": "stub", "model": "stub-model"},
                    },
                    "tiers": {
                        "strong": "stub-model",
                        "cheap": "stub-model",
                        "fast": "stub-model",
                    },
                },
                "pipeline": {
                    "review": False,
                    "polish": False,
                    "book_understanding": False,
                    "review_autofix": False,
                    "annotation_alignment": False,
                    "prescan_concurrency": 1,
                    "translation_mode": "standard",
                },
                "segment": {"max_tokens_per_batch": 600, "max_tokens_per_segment": 300},
                "output": {"mono": True, "bilingual": False, "about_page": False},
                "paths": {"state_dir": str(tmp_path / "state")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _plan(tmp_path: Path, config: Path) -> dict:
    return {
        "adapter": "provider",
        "executable": __import__("sys").executable,
        "argv_prefix": ["-m", "wenyi_cli"],
        "code_source": str(Path(__file__).resolve().parents[3]),
        "cwd": str(tmp_path / "cwd"),
        "config": str(config),
    }


def test_metadata_command_reports_production_routes(
    tmp_path: Path, stub_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WENYI_TEST_STUB_KEY", "stub-key")
    config = _config(tmp_path, f"http://127.0.0.1:{stub_server.server_port}/v1")
    plan = _plan(tmp_path, config)
    Path(plan["cwd"]).mkdir(parents=True)

    listed, error = _run_metadata_command(
        plan=plan, config=str(config), argv=["models", "list", "--json"], timeout=120
    )
    described, explain_error = _run_metadata_command(
        plan=plan,
        config=str(config),
        argv=["models", "explain", "--operation", "translation.body"],
        timeout=120,
    )

    assert error is None and explain_error is None
    assert listed["translation.body"]["model"] == "stub-model"
    assert described["model"] == "stub-model"
    assert described["provider_kind"] == "openai-compatible"
    assert stub_server.requests == 0


def test_translate_uses_the_production_factory_through_the_stub(
    tmp_path: Path, stub_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WENYI_TEST_STUB_KEY", "stub-key")
    config = _config(tmp_path, f"http://127.0.0.1:{stub_server.server_port}/v1")
    plan = _plan(tmp_path, config)
    source = tmp_path / "tiny.en.md"
    source.write_text(
        "# The Harbor\n\nThe lantern glowed over the quiet harbor.\n\nA gull circled twice.\n",
        encoding="utf-8",
    )
    output = tmp_path / "out" / "tiny.zh.txt"
    progress = tmp_path / "receipts" / "progress.json"
    result = tmp_path / "receipts" / "result.json"
    binding = tmp_path / "receipts" / "binding.json"

    outcome, logs = run_child(
        group="probe",
        stage="translate",
        plan=plan,
        config=str(config),
        argv=[
            "translate",
            str(source),
            "--format",
            "txt",
            "--out",
            str(output),
            "--no-polish",
            "--no-review",
            "--mono",
            "--no-bilingual",
            "--progress-json",
            str(progress),
            "--result-json",
            str(result),
        ],
        cwd=Path(plan["cwd"]),
        binding_path=binding,
        stdout_path=tmp_path / "logs" / "stdout.log",
        stderr_path=tmp_path / "logs" / "stderr.log",
        timeout=300,
    )

    assert outcome.exit_code == 0, (tmp_path / "logs" / "stderr.log").read_text(encoding="utf-8")
    assert outcome.binding_valid
    assert logs["terminal"]["adapter"] == "provider"
    assert logs["terminal"]["binding"]["adapter"] == "provider"
    assert Path(logs["terminal"]["binding"]["factory_file"]).is_relative_to(
        Path(plan["code_source"]) / "packages" / "core"
    )
    assert stub_server.requests > 0
    assert output.is_file()
    assert "米拉" not in output.read_text(encoding="utf-8")  # no offline fixture substitution


def test_redaction_keeps_environment_references() -> None:
    redacted = redact_metadata(
        {
            "api_key": "sk-live-secret",
            "api_key_env": "OPENAI_API_KEY",
            "headers": {"Authorization": "Bearer sk-live-secret"},
            "base_url": "https://user:pass@example.invalid/v1",
        }
    )

    assert redacted["api_key"] == "<redacted>"
    assert redacted["api_key_env"] == "OPENAI_API_KEY"
    assert redacted["headers"]["Authorization"] == "<redacted>"
    assert "user:pass" not in redacted["base_url"]


def test_unpinned_candidate_is_refused_before_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.helpers import minimal_manifest
    from wenyi_eval import run_variants

    root = tmp_path / "eval"
    manifest = minimal_manifest(root, candidate_binding="unfrozen")
    called = []

    def fake_capture(*, root, plans, timeout=120.0):  # pragma: no cover - must not run
        called.append(True)

    monkeypatch.setattr(run_variants, "capture_runtime_metadata", fake_capture)
    with pytest.raises(EvalError, match="A_prime:unfrozen"):
        run_variants.prepare_manifest(
            root=root, manifest=manifest, repo_root=Path(manifest["repo"]["root"])
        )
    assert called == []
