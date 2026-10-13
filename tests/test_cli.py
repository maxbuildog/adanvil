from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import adanvil.cli as cli_module
from adanvil.cli import main
from adanvil.qc.report import Check, QCReport


def test_cli_init_index_and_recipes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["init", str(tmp_path), "CLI demo", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["created"] is True

    assert main(["index", str(tmp_path), "--json"]) == 0
    indexed = json.loads(capsys.readouterr().out)
    assert indexed["added"] == []

    assert main(["recipes", "--json"]) == 0
    recipes = json.loads(capsys.readouterr().out)
    assert recipes and all("checklist" not in recipe for recipe in recipes)


def test_cli_index_passes_the_transcribe_choice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["init", str(tmp_path), "CLI speech"]) == 0
    capsys.readouterr()
    seen: list[object] = []
    real = cli_module.index_library

    def spy(root, source, **options):
        seen.append(options["transcribe"])
        return real(root, source, **options)

    monkeypatch.setattr(cli_module, "index_library", spy)
    for choice in ("auto", "true", "false"):
        assert main(["index", str(tmp_path), "--transcribe", choice, "--json"]) == 0
    assert seen == ["auto", True, False]
    assert main(["index", str(tmp_path), "--json"]) == 0 and seen[-1] == "auto"
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(["index", str(tmp_path), "--transcribe", "maybe"])


def test_cli_cat_reads_project_text_and_reports_truncation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["init", str(tmp_path), "CLI text"]) == 0
    capsys.readouterr()
    (tmp_path / "music" / "LICENCES.txt").write_text("é" * 100, encoding="utf-8")

    assert main(["cat", str(tmp_path), "music/LICENCES.txt", "--max-bytes", "5", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["truncated"] is True
    assert "Content truncated" in value["note"]


def test_cli_validate_reports_exit_two_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["init", str(tmp_path), "CLI demo"])
    capsys.readouterr()
    storyboard = tmp_path / "invalid.json"
    storyboard.write_text(
        json.dumps({"version": 1, "name": "bad", "segments": []}), encoding="utf-8"
    )

    assert main(["validate", str(tmp_path), "invalid.json", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["errors"]


def test_cli_storyboard_paths_accept_project_absolute_and_cwd_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    assert main(["init", str(project), "CLI paths"]) == 0
    capsys.readouterr()
    storyboard = {"version": 1, "name": "bad", "segments": []}
    inside = project / "inside.json"
    inside.write_text(json.dumps(storyboard), encoding="utf-8")
    assert main(["validate", str(project), str(inside), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["errors"]

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(storyboard), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["validate", str(project), "outside.json", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["errors"]

    assert main(["validate", str(project), str(tmp_path / "missing.json")]) == 1
    assert "storyboard absolute path must be inside the project" in capsys.readouterr().err


def test_cli_qc_loads_automatic_report_and_prints_human_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    assert main(["init", str(project), "CLI QC"]) == 0
    capsys.readouterr()
    video = project / "renders" / "ad.mp4"
    video.write_bytes(b"placeholder")
    report_path = project / "renders" / "ad.report.json"
    report_path.write_text(
        json.dumps(
            {
                "path": str(video),
                "sheet": str(project / "renders" / "sheet.jpg"),
                "duration": 2.0,
                "timeline": [],
                "overlays": [],
                "warnings": [],
                "quality": "final",
            }
        ),
        encoding="utf-8",
    )
    image = project / "renders" / "qc.jpg"
    image.write_bytes(b"image")
    captured: dict[str, object] = {}

    def fake_qc(video_path: Path, *, render_report: object = None) -> QCReport:
        captured["video"] = video_path
        captured["report"] = render_report
        return QCReport(
            ok=True,
            checks=[Check("duration", "pass", "duration is 2.000s")],
            images=[image],
            summary={},
        )

    monkeypatch.setattr(cli_module, "qc_report", fake_qc)
    assert main(["qc", str(project), "renders/ad.mp4"]) == 0
    output = capsys.readouterr().out
    assert captured["report"] is not None
    assert "PASS  duration  duration is 2.000s" in output
    assert "images:" in output
    assert str(image) in output
    assert output.rstrip().endswith("OK")


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert "adanvil 0.2.0" in capsys.readouterr().out


def test_cli_render_preview_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init", str(tmp_path), "CLI render"]) == 0
    capsys.readouterr()
    fixture = Path(__file__).parent / "fixtures" / "out" / "ref_ad.mp4"
    shutil.copy2(fixture, tmp_path / "footage" / "ref_ad.mp4")
    storyboard = {
        "version": 1,
        "name": "cli_preview",
        "segments": [
            {
                "type": "clip",
                "source": "footage/ref_ad.mp4",
                "in": 0,
                "out": 1.2,
            }
        ],
    }
    storyboard_path = tmp_path / "storyboard.json"
    storyboard_path.write_text(json.dumps(storyboard), encoding="utf-8")
    assert (
        main(
            [
                "render",
                str(tmp_path),
                "storyboard.json",
                "--preview",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert Path(result["report_path"]).is_file()
    assert Path(result["report_path"]).name == "cli_preview.preview.report.json"
    assert result["report"]["quality"] == "preview"
