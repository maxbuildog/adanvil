"""Human and machine-facing command line wrappers for AdAnvil."""

from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from adanvil import __version__
from adanvil.endcard import write_end_card
from adanvil.library import index_library
from adanvil.onboard import onboard
from adanvil.project import init_project, read_project_text, resolve_in_project
from adanvil.qc import qc_report
from adanvil.reference import decode_reference
from adanvil.render import compose
from adanvil.server import _validate, render_caption_preview
from adanvil.sfx import SfxError, fetch_pack
from adanvil.storyboard import get_recipe, list_recipes
from adanvil.vision import vision_setup
from adanvil.voice import VoiceSpec, generate_voice, voice_setup


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _print(value: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_jsonable(value), ensure_ascii=False, indent=2))
    elif isinstance(value, str):
        print(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                print(f"{key}: {json.dumps(_jsonable(item), ensure_ascii=False)}")
            else:
                print(f"{key}: {item}")
    else:
        print(value)


def _root(value: str, *, init: bool = False) -> Path:
    root = Path(value).resolve()
    if init:
        if root.exists() and not root.is_dir():
            raise ValueError(f"project path is not a directory: {root}")
    else:
        if not root.is_dir():
            raise ValueError(f"project directory does not exist: {root}")
        if not (root / "adanvil.json").is_file():
            raise ValueError(f"not an AdAnvil project (adanvil.json is missing): {root}")
    return root


def _report_path(root: Path, name: str) -> Path:
    return root / "renders" / f"{name}.report.json"


def _storyboard_argument(root: Path, value: str) -> dict[str, Any] | str:
    """Resolve a CLI storyboard from the project or current directory."""

    requested = Path(value)
    if requested.is_absolute():
        # A human on the command line may pass an absolute path: accept it when it
        # lies inside the project (checked lexically, like every project path).
        try:
            value = requested.relative_to(root).as_posix()
            requested = Path(value)
        except ValueError:
            raise ValueError(
                f"storyboard absolute path must be inside the project: {value}"
            ) from None

    try:
        project_candidate = resolve_in_project(root, value)
    except (TypeError, ValueError):
        project_candidate = None
    if project_candidate is not None and project_candidate.is_file():
        return project_candidate.relative_to(root).as_posix()

    cwd_candidate = (Path.cwd() / requested).resolve()
    if cwd_candidate.is_file():
        try:
            source = json.loads(cwd_candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not read storyboard {value!r}: {exc}") from exc
        if not isinstance(source, dict):
            raise ValueError(f"storyboard JSON must contain an object: {cwd_candidate}")
        return source

    raise ValueError(
        f"storyboard file does not exist: {value}; checked the project and current directory"
    )


def _load_render_report(path: Path) -> Any:
    from adanvil.render.compose import RenderReport

    try:
        return RenderReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load render report {path}: {exc}") from exc


def _print_qc(result: Any) -> None:
    for check in result.checks:
        print(f"{check.status.upper()}  {check.id}  {check.message}")
    print("images:")
    for path in result.images:
        print(path)
    print("OK" if result.ok else "FAILED")


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit machine-readable JSON",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS,
        help="show a full traceback for unexpected errors",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adanvil")
    parser.add_argument("--version", action="version", version=f"adanvil {__version__}")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--debug", action="store_true", help="show unexpected tracebacks")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("mcp", help="run the MCP server over stdio")
    _add_output_flags(command)

    command = commands.add_parser("voice-setup", help="install the optional voice environment")
    command.add_argument("--target", type=Path)
    command.add_argument("--python")
    command.add_argument(
        "--engine",
        choices=("chatterbox", "kokoro"),
        default="chatterbox",
        help="chatterbox (cloned voice, slow) or kokoro (fixed voices, fast)",
    )
    _add_output_flags(command)

    command = commands.add_parser(
        "music-setup", help="install the optional local music generator (ACE-Step, Apache-2.0)"
    )
    command.add_argument("--target", type=Path)
    command.add_argument("--python")
    _add_output_flags(command)

    command = commands.add_parser(
        "vision-setup",
        help="install the optional vision environment (shot search by sentence, faces)",
    )
    command.add_argument("--target", type=Path)
    command.add_argument("--python")
    _add_output_flags(command)

    command = commands.add_parser("init", help="create a project")
    command.add_argument("dir", metavar="DIR")
    command.add_argument("name")
    _add_output_flags(command)

    command = commands.add_parser("cat", help="read a project text file")
    command.add_argument("dir", metavar="DIR")
    command.add_argument("path", metavar="PATH")
    command.add_argument("--max-bytes", type=int, default=65536)
    _add_output_flags(command)

    command = commands.add_parser("onboard", help="discover or write a brand profile")
    command.add_argument("dir", metavar="DIR")
    command.add_argument("--url")
    command.add_argument("--answers", type=Path, help="JSON answers file")
    _add_output_flags(command)

    command = commands.add_parser("index", help="index project media")
    command.add_argument("dir", metavar="DIR")
    command.add_argument("--source", default="footage")
    command.add_argument(
        "--transcribe",
        choices=["auto", "true", "false"],
        default="auto",
        help="speech of the clips: auto (clips with sound), true (every clip with an audio "
        "stream) or false (none)",
    )
    command.add_argument(
        "--vision",
        choices=["auto", "true", "false"],
        default="auto",
        help="vision pass (embeddings, faces, automatic tags, product versions): auto (when "
        "the vision environment is installed), true (redo every entry) or false (none)",
    )
    _add_output_flags(command)

    command = commands.add_parser("decode", help="decode a reference video")
    command.add_argument("dir", metavar="DIR")
    command.add_argument("video")
    command.add_argument("--transcribe", action="store_true")
    command.add_argument("--language")
    _add_output_flags(command)

    command = commands.add_parser("recipes", help="list recipes or show one recipe")
    command.add_argument("id", nargs="?")
    _add_output_flags(command)

    command = commands.add_parser("validate", help="validate a storyboard")
    command.add_argument("dir", metavar="DIR")
    command.add_argument("storyboard", metavar="STORYBOARD.json")
    command.add_argument("--recipe")
    _add_output_flags(command)

    command = commands.add_parser(
        "export", help="export a storyboard as a timeline for DaVinci Resolve or Premiere Pro"
    )
    command.add_argument("dir", metavar="DIR")
    command.add_argument("storyboard", metavar="STORYBOARD.json")
    command.add_argument(
        "--formats",
        nargs="+",
        choices=["otio", "fcp7xml", "edl", "srt"],
        help="default: all four",
    )
    command.add_argument("--relative", action="store_true", help="relative media paths")
    command.add_argument("--copy-media", action="store_true", help="copy sources to media/")
    command.add_argument("--out", type=Path, help="write here instead of exports/<name>/")
    command.add_argument(
        "--media-root",
        help="absolute folder a relative export will live in (written in the media paths)",
    )
    command.add_argument(
        "--overwrite", action="store_true", help="replace an export of another media mode"
    )
    command.add_argument("--cache", type=Path, help="cache folder (default: the project's cache/)")
    _add_output_flags(command)

    command = commands.add_parser("render", help="render a storyboard synchronously")
    command.add_argument("dir", metavar="DIR")
    command.add_argument("storyboard", metavar="STORYBOARD.json")
    command.add_argument("--preview", action="store_true")
    _add_output_flags(command)

    command = commands.add_parser("voice", help="generate a voice track synchronously")
    command.add_argument("dir", metavar="DIR")
    command.add_argument("name")
    command.add_argument("spec", type=Path, metavar="SPEC.json")
    _add_output_flags(command)

    command = commands.add_parser("qc", help="run QC on a video")
    command.add_argument("dir", metavar="DIR")
    command.add_argument("video")
    command.add_argument("--report")
    _add_output_flags(command)

    command = commands.add_parser("preview-caption", help="preview a caption over a real frame")
    command.add_argument("dir", metavar="DIR")
    command.add_argument("source")
    command.add_argument("t", type=float)
    command.add_argument("overlay", type=Path, metavar="OVERLAY.json")
    command.add_argument("out", type=Path, metavar="OUT.png")
    _add_output_flags(command)

    command = commands.add_parser(
        "sfx-fetch", help="build the CC0 sound-effects pack in the project's sfx/ folder"
    )
    command.add_argument("dir", metavar="DIR")
    command.add_argument(
        "--only", nargs="+", metavar="NAME", help="build only these sounds (default: the pack)"
    )
    command.add_argument(
        "--offline", action="store_true", help="synthesise every sound, download nothing"
    )
    command.add_argument(
        "--force", action="store_true", help="rebuild sounds that are already in sfx/"
    )
    _add_output_flags(command)

    command = commands.add_parser("end-card", help="compose an end card image from a product photo")
    command.add_argument("spec", type=Path, metavar="SPEC.json")
    command.add_argument("--project", required=True, metavar="DIR")
    command.add_argument("--name", required=True)
    _add_output_flags(command)
    return parser


def _run(args: argparse.Namespace) -> int:
    as_json = bool(getattr(args, "json", False))

    if args.command == "mcp":
        from adanvil.server import build_server

        build_server().run("stdio")
        return 0

    if args.command == "music-setup":
        from adanvil.musicgen import music_setup

        if as_json:
            from contextlib import redirect_stdout

            with redirect_stdout(io.StringIO()):
                interpreter = music_setup(target=args.target, python=args.python)
        else:
            interpreter = music_setup(target=args.target, python=args.python)
        _print({"python": str(interpreter)}, as_json)
        return 0

    if args.command == "voice-setup":
        if as_json:
            from contextlib import redirect_stdout

            with redirect_stdout(io.StringIO()):
                interpreter = voice_setup(
                    target=args.target, python=args.python, engine=args.engine
                )
        else:
            interpreter = voice_setup(target=args.target, python=args.python, engine=args.engine)
        _print({"python": str(interpreter)}, as_json)
        return 0

    if args.command == "vision-setup":
        if as_json:
            from contextlib import redirect_stdout

            with redirect_stdout(io.StringIO()):
                interpreter = vision_setup(target=args.target, python=args.python)
        else:
            interpreter = vision_setup(target=args.target, python=args.python)
        _print({"python": str(interpreter)}, as_json)
        return 0

    if args.command == "init":
        root = _root(args.dir, init=True)
        init_project(root, args.name)
        _print({"project_dir": str(root), "name": args.name, "created": True}, as_json)
        return 0

    if args.command == "cat":
        root = _root(args.dir)
        value = read_project_text(root, args.path, args.max_bytes)
        _print(value if as_json else value["content"], as_json)
        return 0

    if args.command == "recipes":
        if args.id is None:
            value = [
                {
                    "id": recipe.id,
                    "name": recipe.name,
                    "summary": recipe.summary,
                    "duration": recipe.duration.model_dump(mode="json"),
                }
                for recipe in list_recipes()
            ]
        else:
            value = get_recipe(args.id).model_dump(mode="json")
        _print(value, as_json)
        return 0

    if args.command == "end-card":
        root = _root(args.project)
        spec_path = args.spec
        if not spec_path.is_file():
            spec_path = resolve_in_project(root, spec_path.as_posix())
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        output, payload = write_end_card(root, spec, args.name)
        if as_json:
            _print(payload, True)
        else:
            print(f"End card written: {output}")
            print(f"title: {payload['title_lines']}; photo card: {payload['photo_card']}")
        return 0

    root = _root(args.dir)

    if args.command == "sfx-fetch":
        try:
            result = fetch_pack(root, names=args.only, offline=args.offline, force=args.force)
        except SfxError as exc:
            raise ValueError(str(exc)) from exc
        if as_json:
            _print(result, True)
        else:
            print(
                f"Sound pack in {root / result['folder']}: {len(result['built'])} built, "
                f"{len(result['kept'])} kept"
            )
            print(f"downloaded (CC0): {', '.join(result['downloaded']) or 'none'}")
            print(f"generated: {', '.join(result['generated']) or 'none'}")
            for warning in result["warnings"]:
                print(f"Warning: {warning}")
            print("Sources and licences: sfx/LICENCES.txt")
        return 0

    if args.command == "export":
        from adanvil.export import export_timeline

        board = _storyboard_argument(root, args.storyboard)
        if isinstance(board, str):
            board = json.loads((root / board).read_text(encoding="utf-8"))
        result = export_timeline(
            root,
            board,
            formats=args.formats,
            relative=args.relative,
            copy_media=args.copy_media,
            media_root=args.media_root,
            overwrite=args.overwrite,
            out_dir=args.out,
            cache_dir=args.cache,
        )
        if as_json:
            _print(result, True)
        else:
            print(f"Export written to {result['folder']}")
            for kind, file in result["files"].items():
                print(f"  {kind}: {file}")
            for warning in result["warnings"]:
                print(f"Warning: {warning}")
            for item in result["not_exported"]:
                print(f"Not exported: {item}")
        return 0

    if args.command == "onboard":
        answers = None
        if args.answers is not None:
            answers = json.loads(args.answers.read_text(encoding="utf-8"))
        result = onboard(root, url=args.url, answers=answers)
        value = {
            "facts": result.facts.model_dump(mode="json") if result.facts else None,
            "questionnaire": [item.model_dump(mode="json") for item in result.questionnaire],
            "brand_path": str(result.brand_path) if result.brand_path else None,
            "brief_path": str(result.brief_path) if result.brief_path else None,
            "open_questions": result.open_questions,
        }
        _print(value, as_json)
        return 0

    if args.command == "index":
        transcribe = args.transcribe if args.transcribe == "auto" else args.transcribe == "true"
        vision = args.vision if args.vision == "auto" else args.vision == "true"
        result = index_library(root, args.source, transcribe=transcribe, vision=vision)
        _print(result, as_json)
        return 0

    if args.command == "decode":
        result = decode_reference(
            root,
            args.video,
            transcribe=args.transcribe,
            language=args.language,
        )
        _print(result, as_json)
        return 0

    if args.command == "voice":
        spec = VoiceSpec.model_validate(json.loads(args.spec.read_text(encoding="utf-8")))
        track = generate_voice(root, spec, args.name)
        summary = {
            "manifest": track.manifest,
            "lines": [
                {
                    "index": line.index,
                    "text": line.text,
                    "display": line.display,
                    "start": line.start,
                    "duration": line.duration,
                    "status": line.status,
                    "transcript": line.transcript,
                    "unmatched_words": line.unmatched_words,
                }
                for line in track.lines
            ],
            "total_duration": track.total_duration,
            "warnings": track.warnings,
        }
        if as_json:
            _print(summary, True)
        else:
            print(f"Manifest: {track.manifest}")
            print("index  start    duration  status   transcript")
            for line in track.lines:
                print(
                    f"{line.index:>5}  {line.start:>7.2f}  {line.duration:>8.2f}  "
                    f"{line.status:<7}  {line.transcript}"
                )
            print(f"Total: {track.total_duration:.2f}s")
            for warning in track.warnings:
                print(f"Warning: {warning}")
        return 0

    if args.command == "validate":
        _, report = _validate(root, _storyboard_argument(root, args.storyboard), args.recipe)
        _print(report, as_json)
        return 0 if report.ok else 2
    if args.command == "render":
        model, report = _validate(root, _storyboard_argument(root, args.storyboard), None)
        if not report.ok:
            _print(report, as_json)
            return 2
        quality = "preview" if args.preview else model.output.quality
        rendered = compose(model, root=root, quality=quality)
        path = Path(str(rendered.path)).with_suffix(".report.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(rendered.model_dump(mode="json"), ensure_ascii=False, indent=2) + chr(10),
            encoding="utf-8",
        )
        value = {"report_path": str(path), "report": rendered}
        if as_json:
            _print(value, True)
        else:
            print(f"Render complete: {rendered.path}")
            print(f"Report: {path}")
        return 0
    if args.command == "qc":
        report = None
        if args.report:
            report_path = resolve_in_project(root, args.report)
            report = _load_render_report(report_path)
        else:
            video_path = resolve_in_project(root, args.video)
            automatic_report = video_path.with_suffix(".report.json")
            if automatic_report.is_file():
                report = _load_render_report(automatic_report)
        result = qc_report(resolve_in_project(root, args.video), render_report=report)
        if as_json:
            _print(result, True)
        else:
            _print_qc(result)
        return 0

    if args.command == "preview-caption":
        overlay = json.loads(args.overlay.read_text(encoding="utf-8"))
        output = resolve_in_project(root, args.out.as_posix())
        payload, _ = render_caption_preview(root, args.source, args.t, overlay, output=output)
        if as_json:
            _print(payload, True)
        else:
            print(f"Preview written: {output}")
            print(
                f"bbox: {payload['bbox']}; size_used: {payload['size_used']}; "
                f"overflow: {payload['overflow']}"
            )
        return 0

    raise ValueError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    """Run the ``adanvil`` command and return its process exit code."""

    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except SystemExit:
        raise
    except Exception as exc:
        if bool(getattr(args, "debug", False)):
            traceback.print_exc()
        else:
            message = " ".join(str(exc).splitlines())
            print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
