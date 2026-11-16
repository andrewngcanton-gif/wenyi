"""Child-process bootstrap that selects one code source and installs the offline client.

Usage::

    python -m wenyi_eval.offline_cli --code-source <checkout> -- <wenyi argv...>

Only ``build_client`` is replaced. The binding record written before the CLI runs
proves which ``wenyi_core`` and ``wenyi_cli`` modules were actually imported.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _fail(message: str, code: int = 2) -> None:
    print(f"offline bootstrap: {message}", file=sys.stderr)
    raise SystemExit(code)


def _require_inside(path: str | os.PathLike[str], root: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{label} resolved outside the selected code source: {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" not in arguments:
        _fail("expected --code-source PATH -- <wenyi argv>")
    split = arguments.index("--")
    parser = argparse.ArgumentParser(prog="wenyi_eval.offline_cli")
    parser.add_argument("--code-source", required=True)
    parsed = parser.parse_args(arguments[:split])
    command = arguments[split + 1 :]
    if not command:
        _fail("no Wenyi command was supplied")

    code_source = Path(parsed.code_source).resolve()
    core = code_source / "packages" / "core"
    cli = code_source / "packages" / "cli"
    if not (core / "wenyi_core").is_dir() or not (cli / "wenyi_cli").is_dir():
        _fail(f"code source does not contain packages/core and packages/cli: {code_source}")
    sys.path.insert(0, str(cli))
    sys.path.insert(0, str(core))

    import wenyi_core.llm.factory as factory

    from wenyi_eval.offline_llm import build_client

    factory.build_client = build_client
    # The runtime imported build_client at module scope in older revisions; patch
    # the already-bound symbol too so every supported code source uses the adapter.
    try:
        import wenyi_core.pipeline.run.runtime as runtime

        runtime.build_client = build_client
    except ImportError:
        pass

    import wenyi_cli.cli as cli_module

    core_file = _require_inside(sys.modules["wenyi_core"].__file__ or "", core, "wenyi_core")
    cli_file = _require_inside(cli_module.__file__ or "", cli, "wenyi_cli")
    factory_file = _require_inside(factory.__file__ or "", core, "wenyi_core.llm.factory")

    binding_path = os.environ.get("WENYI_EVAL_BINDING_PATH")
    if binding_path:
        from wenyi_eval.common import write_json_atomic

        write_json_atomic(
            binding_path,
            {
                "code_source": str(code_source),
                "core_file": str(core_file),
                "cli_file": str(cli_file),
                "factory_file": str(factory_file),
                "cwd": os.getcwd(),
                "python": sys.version.split()[0],
                "command": command,
            },
        )

    sys.argv = ["wenyi", *command]
    cli_module.main()


if __name__ == "__main__":
    main()
