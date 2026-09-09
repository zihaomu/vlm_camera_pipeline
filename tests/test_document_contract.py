from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt

WORKSPACE = Path(__file__).resolve().parents[1]


def test_deployment_document_parses_and_enforces_uv_only_python_commands() -> None:
    document = (WORKSPACE / "doc/amd_395_pipeline.md").read_text(encoding="utf-8")
    tokens = MarkdownIt("commonmark").parse(document)

    assert tokens
    assert sum(token.type == "fence" for token in tokens) >= 50
    for forbidden in (
        "python3.12 -m venv",
        "source .venv/bin/activate",
        "python -m pip install",
        "pipx install",
    ):
        assert forbidden not in document
    assert "uv sync --frozen" in document
    assert "uv run --frozen python scripts/run_camera.py" in document


def test_documented_cli_contract_is_present_in_the_real_parser() -> None:
    from scripts.run_camera import build_parser

    documented_flags = {
        "--backend",
        "--device",
        "--fourcc",
        "--width",
        "--height",
        "--camera-fps",
        "--nms-mode",
        "--display-mode",
        "--display",
        "--record",
        "--record-path",
        "--vlm",
        "--vlm-interval",
        "--vlm-server",
        "--vlm-model",
        "--vlm-mmproj",
        "--vlm-port",
        "--vlm-timeout",
        "--vlm-startup-timeout",
        "--vlm-caption-expiry",
        "--vlm-context-size",
        "--vlm-image-max-tokens",
        "--vlm-max-tokens",
        "--vlm-prompt",
        "--vlm-log",
        "--max-latency-ms",
        "--metrics-json",
    }
    actual_flags = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert documented_flags <= actual_flags


def test_documented_vlm_only_demo_contract_is_present() -> None:
    from scripts.run_vlm_demo import build_parser

    documented_flags = {
        "--device",
        "--video-file",
        "--loop-video",
        "--video-fps",
        "--vlm-interval",
        "--vlm-server",
        "--vlm-model",
        "--vlm-mmproj",
        "--subtitle-height",
        "--subtitle-font",
        "--window-scale",
        "--duration-seconds",
        "--metrics-json",
    }
    actual_flags = {
        option for action in build_parser()._actions for option in action.option_strings
    }
    assert documented_flags <= actual_flags
