from __future__ import annotations

import json
import shutil
from pathlib import Path

import anyio
from mcp import Client

from adanvil.project import init_project
from adanvil.server import build_server, render_caption_preview
from adanvil.voice import FakeEngine, VoiceSpec, generate_voice

TOOL_NAMES = {
    "project_init",
    "project_read_text",
    "brand_onboard",
    "brand_get",
    "library_index",
    "library_view",
    "library_annotate",
    "library_search",
    "speech_view",
    "speech_takes",
    "speech_refine",
    "reference_decode",
    "reference_breakdown",
    "library_overview",
    "frame_view",
    "recipe_list",
    "recipe_get",
    "editing_guide",
    "storyboard_schema",
    "storyboard_save",
    "storyboard_patch",
    "caption_preview",
    "end_card",
    "storyboard_validate",
    "render_start",
    "voice_generate",
    "voice_view",
    "music_profile",
    "music_search",
    "music_fetch",
    "music_generate",
    "music_beats",
    "sfx_list",
    "storyboard_snap",
    "job_status",
    "qc_run",
    "ad_score",
    "compare_pair",
    "vision_setup",
    "variants_build",
    "render_batch",
    "render_formats",
    "timeline_export",
}


def _texts(result):
    return [item.text for item in result.content if item.type == "text"]


def _json(result):
    text = _texts(result)[0]
    return json.loads(text)


def _has_image(result) -> bool:
    return any(item.type == "image" for item in result.content)


async def client_schema(server) -> dict:
    async with Client(server) as client:
        result = await client.call_tool("storyboard_schema", {})
    return json.loads(_texts(result)[0])


async def _list_tools() -> None:
    server = build_server()
    async with Client(server) as client:
        result = await client.list_tools()
    tools = {tool.name: tool for tool in result.tools}
    assert set(tools) == TOOL_NAMES
    assert all(tool.description for tool in tools.values())
    assert all(tool.input_schema for tool in tools.values())
    # The speech tools of the talking-head workflow and the new index parameter.
    properties = {name: set(tool.input_schema["properties"]) for name, tool in tools.items()}
    assert {"project_dir", "entry_id", "source", "max_pause"} <= properties["speech_view"]
    assert {"words", "max_phrases", "first_phrase"} <= properties["speech_view"]
    assert {"project_dir", "sources", "model", "language"} <= properties["speech_refine"]
    assert {"project_dir", "source", "entry_ids", "max_pause"} <= properties["speech_takes"]
    assert "transcribe" in properties["library_index"]
    assert "vision" in properties["library_index"]
    assert {"query", "limit", "auto_tags"} <= properties["library_search"]
    assert "SENTENCE" in tools["library_search"].description
    assert "vision_setup" in tools["library_search"].description
    assert "speech_captions" in tools["caption_preview"].description
    assert "clean_segments" in tools["speech_view"].description
    assert "best take" in tools["speech_takes"].description
    assert "stronger model" in tools["speech_refine"].description
    assert "compact" in tools["speech_view"].description
    assert "compact" in tools["brand_onboard"].description
    assert "separate_audio_candidates" in tools["library_index"].description
    schema = (await client_schema(server))
    assert "SpeechCaptionsOverlay" in schema["$defs"] and "AudioFrom" in schema["$defs"]
    assert "clips" in schema["$defs"]["AudioSpec"]["properties"]


def test_server_lists_exact_tool_contract() -> None:
    anyio.run(_list_tools)


def test_packaged_editing_guide_matches_source() -> None:
    source = Path(__file__).parents[1] / "docs" / "EDITING_GUIDE.md"
    packaged = Path(__file__).parents[1] / "src" / "adanvil" / "assets" / "EDITING_GUIDE.md"
    assert packaged.read_bytes() == source.read_bytes()


def test_caption_preview_expands_voice_captions_from_manifest(
    media: dict[str, dict[str, object]], tmp_path: Path
) -> None:
    init_project(tmp_path, "voice preview")
    shutil.copy2(Path(str(media["ref_ad.mp4"]["path"])), tmp_path / "footage" / "ref_ad.mp4")
    track = generate_voice(
        tmp_path,
        VoiceSpec(engine="fake", lines=["One two three four"]),
        "preview",
        engine=FakeEngine(),
    )

    payload, image_path = render_caption_preview(
        tmp_path,
        "footage/ref_ad.mp4",
        1.0,
        {
            "type": "voice_captions",
            "manifest": track.manifest,
            "mode": "chunks",
            "max_words": 2,
            "chunk_index": 1,
            "style": "outline",
        },
    )

    assert image_path.is_file()
    assert payload["bbox"]
    assert payload["lines"] == ["three four"]
    assert payload["expanded"]
    # Captions show the script text (here "One two three four"), timed on the words.
    assert [item["text"] for item in payload["expanded"]] == ["One two", "three four"]
    assert payload["expanded"][1]["start"] < payload["expanded"][1]["end"]


async def _end_to_end(root: Path, fixture_root: Path) -> None:
    server = build_server()
    async with Client(server) as client:
        missing_project = await client.call_tool(
            "library_index",
            {"project_dir": str(root / "missing")},
        )
        assert missing_project.is_error
        assert "does not exist" in _texts(missing_project)[0]

        init = await client.call_tool(
            "project_init", {"project_dir": str(root), "name": "MCP demo"}
        )
        assert not init.is_error
        footage = root / "footage"
        shutil.copy2(fixture_root / "ref_ad.mp4", footage / "ref_ad.mp4")
        (root / "music" / "LICENCES.txt").write_text("licensed", encoding="utf-8")

        text_file = await client.call_tool(
            "project_read_text",
            {"project_dir": str(root), "path": "music/LICENCES.txt"},
        )
        assert _json(text_file)["content"] == "licensed"

        indexed = await client.call_tool(
            "library_index", {"project_dir": str(root), "source": "footage"}
        )
        summary = _json(indexed)
        assert summary["counts"]["added"] == 1
        entry = {"id": summary["entries"][0].split(" | ")[0]}
        annotated = await client.call_tool(
            "library_annotate",
            {
                "project_dir": str(root),
                "entry_id": entry["id"],
                "tags": ["candidate"],
                "quality": 5,
            },
        )
        assert not annotated.is_error
        found = await client.call_tool(
            "library_search", {"project_dir": str(root), "tags": ["candidate"]}
        )
        assert _json(found)["count"] == 1

        decoded = await client.call_tool(
            "reference_decode", {"project_dir": str(root), "video": "footage/ref_ad.mp4"}
        )
        assert _has_image(decoded)
        decoded_json = _json(decoded)
        assert decoded_json["shot_sheets"]
        assert sum(item.type == "image" for item in decoded.content) == 1 + len(
            decoded_json["shot_sheets_inline"]
        )
        template = decoded_json["breakdown_template"]
        bad = await client.call_tool(
            "reference_breakdown",
            {"project_dir": str(root), "reference": "reference/ref-ad", "shots": template[:1]},
        )
        assert bad.is_error
        filled = [{**entry, "type": "other", "role": "hook"} for entry in template]
        good = await client.call_tool(
            "reference_breakdown",
            {"project_dir": str(root), "reference": "reference/ref-ad", "shots": filled},
        )
        assert not good.is_error
        assert _json(good)["role_sequence"] == "hook"

        overview = await client.call_tool(
            "library_overview", {"project_dir": str(root), "path": "footage"}
        )
        assert not overview.is_error
        assert _has_image(overview)
        assert _json(overview)["tree"]["videos"] >= 1

        preview = await client.call_tool(
            "caption_preview",
            {
                "project_dir": str(root),
                "source": "footage/ref_ad.mp4",
                "t": 1.0,
                "overlay": {
                    "type": "caption",
                    "style": "pill",
                    "lines": ["Try it"],
                    "start": 0,
                    "end": 2,
                },
            },
        )
        assert _has_image(preview)
        assert _json(preview)["bbox"]

        storyboard = {
            "version": 1,
            "name": "mcp_demo",
            "segments": [
                {
                    "type": "clip",
                    "source": "footage/ref_ad.mp4",
                    "in": 0,
                    "out": 1.2,
                }
            ],
        }
        valid = await client.call_tool(
            "storyboard_validate", {"project_dir": str(root), "storyboard": storyboard}
        )
        assert _json(valid)["ok"] is True
        invalid = await client.call_tool(
            "storyboard_validate",
            {
                "project_dir": str(root),
                "storyboard": {**storyboard, "segments": [{"type": "clip"}]},
            },
        )
        invalid_json = _json(invalid)
        assert invalid_json["ok"] is False
        assert any(issue["where"] for issue in invalid_json["errors"])

        started = await client.call_tool(
            "render_start",
            {"project_dir": str(root), "storyboard": storyboard, "quality": "preview"},
        )
        job = _json(started)
        assert job["status"] in {"queued", "running"}
        status_result = await client.call_tool(
            "job_status", {"job_id": job["job_id"], "wait_seconds": 180}
        )
        status = _json(status_result)
        assert status["status"] == "done", status
        assert _has_image(status_result)
        assert Path(status["report"]["path"]).name == "mcp_demo.preview.mp4"
        assert (root / "renders" / "mcp_demo.preview.report.json").is_file()

        qc = await client.call_tool(
            "qc_run",
            {
                "project_dir": str(root),
                # Use the path the job reports: previews get a ".preview" suffix.
                "video": Path(status["report"]["path"]).relative_to(root).as_posix(),
                "render_report": job["job_id"],
            },
        )
        assert _json(qc)["ok"] is True
        assert sum(item.type == "image" for item in qc.content) >= 2

        escaped = await client.call_tool(
            "frame_view", {"project_dir": str(root), "source": "../x.mp4", "t": 0}
        )
        assert escaped.is_error
        assert "escapes" in _texts(escaped)[0]


def test_mcp_end_to_end(media: dict[str, dict[str, object]], tmp_path: Path) -> None:
    anyio.run(_end_to_end, tmp_path, Path(str(media["ref_ad.mp4"]["path"])).parent)
