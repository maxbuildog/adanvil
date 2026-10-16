"""V2.9: music search (Openverse, Freesound), fetch with a traced licence, the licence in the
render report, ``ad_score`` and the batch manifest, ranking against a storyboard. The network is
mocked with recorded-shape responses."""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
from mcp import Client

from adanvil import music
from adanvil.batch import MANIFEST_COLUMNS, write_manifest
from adanvil.project import init_project
from adanvil.qc import ad_score
from adanvil.render import compose
from adanvil.server import build_server
from adanvil.storyboard import Storyboard
from tests.test_visual_grammar import FORMAT, _project


def _ov(identifier: str, title: str, licence: str, seconds: float, **extra: Any) -> dict:
    version = "1.0" if licence in ("cc0", "pdm") else "4.0"
    return {
        "id": identifier,
        "title": title,
        "foreign_landing_url": f"https://example.org/{identifier}",
        "url": f"https://cdn.example.org/{identifier}.mp3",
        "creator": "guitarman213",
        "license": licence,
        "license_version": version,
        "license_url": "",
        "provider": "freesound",
        "source": "freesound",
        "tags": [{"name": "acoustic"}, {"name": "upbeat"}],
        "filetype": "mp3",
        "duration": int(seconds * 1000),
        "attribution": f'"{title}" by guitarman213 is licensed under CC BY {version}.',
        **extra,
    }


OPENVERSE = {
    "result_count": 8,
    "results": [
        _ov("aaaaaaaa-0001", "Short sunny strum", "cc0", 20),
        _ov("aaaaaaaa-0002", "Long upbeat acoustic", "by", 95),
        _ov("aaaaaaaa-0003", "Remix share alike", "by-sa", 60),
        _ov("aaaaaaaa-0004", "Non commercial", "by-nc", 60),
        _ov("aaaaaaaa-0005", "No derivatives", "by-nd", 60),
        _ov("aaaaaaaa-0006", "NC share alike", "by-nc-sa", 60),
        _ov("aaaaaaaa-0007", "Public domain calm piano", "pdm", 45,
            tags=[{"name": "calm"}, {"name": "piano"}]),
        _ov("aaaaaaaa-0008", "No file", "cc0", 30, url=None),
    ],
}  # fmt: skip

FREESOUND = {
    "count": 4,
    "results": [
        {"id": 111, "name": "Free CC0 loop", "username": "ana", "duration": 50.0,
         "license": "http://creativecommons.org/publicdomain/zero/1.0/",
         "url": "https://freesound.org/people/ana/sounds/111/", "tags": ["music", "calm"],
         "previews": {"preview-hq-mp3": "https://cdn.freesound.org/previews/111_1-hq.mp3"}},
        {"id": 222, "name": "Attribution beat", "username": "bo", "duration": 70.0,
         "license": "http://creativecommons.org/licenses/by/4.0/",
         "url": "https://freesound.org/people/bo/sounds/222/", "tags": ["energetic"],
         "previews": {"preview-hq-mp3": "https://cdn.freesound.org/previews/222_1-hq.mp3"}},
        {"id": 333, "name": "NC beat", "username": "cy", "duration": 70.0,
         "license": "http://creativecommons.org/licenses/by-nc/4.0/",
         "url": "https://freesound.org/people/cy/sounds/333/", "tags": [],
         "previews": {"preview-hq-mp3": "https://cdn.freesound.org/previews/333_1-hq.mp3"}},
        {"id": 444, "name": "Sampling plus", "username": "di", "duration": 70.0,
         "license": "http://creativecommons.org/licenses/sampling+/1.0/",
         "url": "https://freesound.org/people/di/sounds/444/", "tags": [],
         "previews": {"preview-hq-mp3": "https://cdn.freesound.org/previews/444_1-hq.mp3"}},
    ],
}  # fmt: skip


class _Asked(list):
    """The URLs asked, in order; ``headers`` holds the headers sent with each JSON request."""

    def __init__(self) -> None:
        super().__init__()
        self.headers: list[dict | None] = []


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> _Asked:
    """Mock the HTTP layer; the list receives every URL asked."""

    asked = _Asked()

    def fake_json(url: str, headers: dict | None = None) -> dict:
        asked.append(url)
        asked.headers.append(headers)
        if "api.openverse.org" in url:
            tail = url.split("/v1/audio/")[1].split("?")[0].strip("/")
            if tail:
                match = [r for r in OPENVERSE["results"] if r["id"] == tail]
                if not match:
                    raise music.MusicError("HTTP 404")
                return match[0]
            return OPENVERSE
        if "/search/text/" in url:
            return FREESOUND
        if "/sounds/" in url:
            number = url.split("/sounds/")[1].split("/")[0]
            return next(r for r in FREESOUND["results"] if str(r["id"]) == number)
        raise AssertionError(url)

    def fake_download(url: str, destination: Path) -> int:
        asked.append(url)
        wave_form = 0.3 * np.sin(2 * np.pi * 330 * np.arange(48_000 * 3) / 48_000)
        samples = (wave_form * 32767).astype("<i2")
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(48_000)
            handle.writeframes(samples.tobytes())
        return destination.stat().st_size

    monkeypatch.setattr(music, "_http_json", fake_json)
    monkeypatch.setattr(music, "_http_download", fake_download)
    monkeypatch.delenv("FREESOUND_API_KEY", raising=False)
    return asked


# ------------------------------------------------------------------------------- search


def test_search_keeps_only_licences_that_allow_commercial_use_and_modification(network) -> None:
    result = music.music_search(None, "upbeat acoustic")
    titles = {track["title"] for track in result["tracks"]}
    assert titles == {
        "Short sunny strum", "Long upbeat acoustic", "Remix share alike",
        "Public domain calm piano",
    }  # fmt: skip
    assert result["licences_kept"] == ["CC0", "PDM", "CC BY", "CC BY-SA"]
    by_title = {track["title"]: track for track in result["tracks"]}
    assert by_title["Short sunny strum"]["licence"] == "CC0 1.0"
    assert by_title["Short sunny strum"]["attribution_required"] is False
    long = by_title["Long upbeat acoustic"]
    assert long["licence"] == "CC BY 4.0" and long["attribution_required"] is True
    assert "licensed under CC BY" in long["attribution_text"]
    assert long["duration"] == 95.0 and long["preview_url"].endswith(".mp3")
    assert long["source_url"] == "https://example.org/aaaaaaaa-0002" and long["author"]
    assert by_title["Remix share alike"]["licence"] == "CC BY-SA 4.0"
    url = network[0]
    assert "license=cc0%2Cpdm%2Cby%2Cby-sa" in url and "category=music" in url


def test_the_user_agent_is_clear() -> None:
    assert music._headers()["User-Agent"].startswith("AdAnvil-music/")


def test_duration_filter_mood_and_limit(network) -> None:
    result = music.music_search(None, "acoustic", mood="upbeat", min_duration=40, max_duration=80)
    assert {t["title"] for t in result["tracks"]} == {
        "Remix share alike",
        "Public domain calm piano",
    }
    assert "acoustic upbeat" in network[0].replace("+", " ")
    assert music.music_search(None, "acoustic", limit=2)["count"] == 2


def test_freesound_without_a_key_says_so_and_never_fails(network) -> None:
    result = music.music_search(None, "calm piano", source="freesound")
    assert result["tracks"] == [] and any("FREESOUND_API_KEY" in n for n in result["notes"])
    both = music.music_search(None, "calm piano", source="all")
    assert both["count"] == 4 and any("FREESOUND_API_KEY" in n for n in both["notes"])
    assert not any("freesound.org/apiv2" in url for url in network)


def test_freesound_with_a_key_filters_its_licences(
    network, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FREESOUND_API_KEY", "secret")
    result = music.music_search(None, "calm", source="freesound")
    assert {t["title"] for t in result["tracks"]} == {"Free CC0 loop", "Attribution beat"}
    cc0 = next(t for t in result["tracks"] if t["id"] == "111")
    assert cc0["licence"] == "CC0 1.0" and not cc0["attribution_required"]
    by = next(t for t in result["tracks"] if t["id"] == "222")
    assert by["licence"] == "CC BY 4.0" and by["attribution_required"]
    assert "Attribution beat" in by["attribution_text"] and "bo" in by["attribution_text"]
    # The key travels in a header: it is never in a URL (logged, shown in errors).
    assert "secret" not in network[0] and "NonCommercial" not in network[0]
    assert network.headers[0] == {"Authorization": "Token secret"}


def test_search_rejects_bad_arguments(network) -> None:
    with pytest.raises(music.MusicError):
        music.music_search(None, "x", source="spotify")
    with pytest.raises(music.MusicError):
        music.music_search(None, "  ")
    with pytest.raises(music.MusicError):
        music.music_search(None, "x", fit_to="storyboard")


# ------------------------------------------------------------------------------- ranking


def test_ranking_prefers_tracks_that_cover_the_ad_then_the_feel(network, tmp_path: Path) -> None:
    root = tmp_path
    init_project(root, "kettles")
    board = {
        "name": "ad", "format": FORMAT, "emoji": "none",
        "segments": [{"type": "image", "source": "footage/a.png", "duration": 50.0}],
        "audio": {"music_feel": "calm"},
        "output": {"path": "renders/ad.mp4"},
    }  # fmt: skip
    (root / "storyboards").mkdir()
    (root / "storyboards" / "ad.json").write_text(json.dumps(board), encoding="utf-8")
    result = music.music_search(
        root, "piano", fit_to="storyboard", storyboard="storyboards/ad.json"
    )
    assert result["fit_to"]["duration"] == 50.0 and result["fit_to"]["feel"] == "calm"
    titles = [t["title"] for t in result["tracks"]]
    # the ad lasts 50 s: the 60 s and 95 s tracks cover it (the nearest length first),
    # the 45 s and 20 s ones are too short (the longest first)
    assert titles == [
        "Remix share alike", "Long upbeat acoustic", "Public domain calm piano",
        "Short sunny strum",
    ]  # fmt: skip
    top = result["tracks"][0]["fit"]
    assert top == {"covers_ad": True, "feel_guess": "energetic"}
    assert result["tracks"][2]["fit"] == {"covers_ad": False, "feel_guess": "calm"}


def test_an_explicit_feel_puts_the_other_feel_last(network) -> None:
    ranked = music.rank_tracks(
        music.music_search(None, "x")["tracks"], duration=None, feel="energetic"
    )
    assert ranked[0]["fit"]["feel_guess"] in ("unknown", "energetic")
    assert ranked[-1]["title"] == "Public domain calm piano"


# -------------------------------------------------------------------------------- fetch


def test_fetch_writes_the_file_a_sidecar_and_the_licence_line(network, tmp_path: Path) -> None:
    init_project(tmp_path, "kettles")
    result = music.music_fetch(tmp_path, "openverse", "aaaaaaaa-0002")
    path = tmp_path / result["path"]
    assert path.is_file() and result["path"].startswith("music/")
    sidecar = json.loads((tmp_path / (result["path"] + ".licence.json")).read_text("utf-8"))
    assert sidecar["source"] == "openverse" and sidecar["id"] == "aaaaaaaa-0002"
    assert sidecar["title"] == "Long upbeat acoustic" and sidecar["author"] == "guitarman213"
    assert sidecar["licence"] == "CC BY 4.0" and sidecar["licence_code"] == "by"
    assert sidecar["licence_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert sidecar["url"] == "https://example.org/aaaaaaaa-0002"
    assert sidecar["attribution_required"] is True and "CC BY 4.0" in sidecar["attribution_text"]
    assert sidecar["fetch_date"] and sidecar["download_url"].endswith(".mp3")
    assert result["attribution_required"] is True
    assert result["feel"] in ("calm", "medium", "energetic")
    assert result["duration"] and "tempo_bpm" in result and "loudness_lufs" in result
    # a second track appends a line
    music.music_fetch(tmp_path, "openverse", "aaaaaaaa-0001")
    lines = (tmp_path / "music" / "LICENCES.txt").read_text("utf-8").splitlines()
    assert len(lines) == 3 and "CC BY 4.0" in lines[1] and "CC0 1.0" in lines[2]
    assert "Attribution required" in lines[1] and "No attribution required" in lines[2]


def test_fetch_refuses_a_licence_that_is_not_allowed(network, tmp_path: Path) -> None:
    init_project(tmp_path, "kettles")
    for identifier in ("aaaaaaaa-0004", "aaaaaaaa-0005", "aaaaaaaa-0008"):
        with pytest.raises(music.MusicError, match="commercial use and modification"):
            music.music_fetch(tmp_path, "openverse", identifier)
    assert not (tmp_path / "music").exists() or not list((tmp_path / "music").glob("*.mp3"))
    with pytest.raises(music.MusicError, match="not an Openverse id"):
        music.music_fetch(tmp_path, "openverse", "../etc")


def test_fetch_from_freesound_needs_the_key(
    network, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_project(tmp_path, "kettles")
    with pytest.raises(music.MusicError, match="FREESOUND_API_KEY"):
        music.music_fetch(tmp_path, "freesound", "111")
    monkeypatch.setenv("FREESOUND_API_KEY", "secret")
    result = music.music_fetch(tmp_path, "freesound", "111")
    assert result["licence"] == "CC0 1.0" and not result["attribution_required"]
    with pytest.raises(music.MusicError):
        music.music_fetch(tmp_path, "freesound", "333")


# ------------------------------------------------------------- the licence in the render


def test_licence_summary_unknown_attribution_and_none_needed(network, tmp_path: Path) -> None:
    init_project(tmp_path, "kettles")
    (tmp_path / "music").mkdir(exist_ok=True)
    (tmp_path / "music" / "mine.mp3").write_bytes(b"x")
    unknown = music.music_licence(tmp_path, "music/mine.mp3")
    assert unknown["status"] == "unknown" and unknown["licence"] is None
    assert unknown["warnings"][0].startswith("music_licence_unknown")
    assert music.licence_label(unknown) == "unknown"
    by = music.music_fetch(tmp_path, "openverse", "aaaaaaaa-0002")
    info = music.music_licence(tmp_path, by["path"])
    assert info["status"] == "recorded" and info["attribution_required"]
    assert info["warnings"][0].startswith("music_attribution_required")
    assert by["attribution_text"] in info["warnings"][0]
    assert music.licence_label(info) == "CC BY 4.0; attribution required"
    cc0 = music.music_fetch(tmp_path, "openverse", "aaaaaaaa-0001")
    clean = music.music_licence(tmp_path, cc0["path"])
    assert clean["warnings"] == [] and music.licence_label(clean) == "CC0 1.0"
    assert music.music_licence(tmp_path, None) is None and music.licence_label(None) == ""


def _ad(root: Path, track: str | None) -> dict[str, Any]:
    audio: dict[str, Any] = {} if track is None else {"music": track, "music_volume": 0.35}
    return {
        "name": "ad", "format": FORMAT, "emoji": "none",
        "segments": [{"type": "clip", "source": "footage/red.mp4", "in": 0, "out": 3}],
        "overlays": [], "audio": audio, "output": {"path": "renders/ad.mp4", "quality": "final"},
    }  # fmt: skip


@pytest.fixture(scope="module")
def renders(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = _project(tmp_path_factory.mktemp("v29"))
    audio_dir = root / "music"
    audio_dir.mkdir(exist_ok=True)
    import subprocess

    from adanvil._bin import ffmpeg_path

    for name in ("plain.m4a", "cc0.m4a", "by.m4a"):
        subprocess.run(
            [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
             "sine=frequency=330:sample_rate=48000:duration=4", "-c:a", "aac",
             str(audio_dir / name)],
            check=True,
        )  # fmt: skip
    base = {"source": "openverse", "id": "x", "url": "https://e.org/x", "title": "T",
            "author": "A", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
            "fetch_date": "2026-10-09"}  # fmt: skip
    music.write_licence_record(root, "music/cc0.m4a", {
        **base, "licence": "CC0 1.0", "attribution_required": False,
        "attribution_text": ""})  # fmt: skip
    music.write_licence_record(root, "music/by.m4a", {
        **base, "licence": "CC BY 4.0", "attribution_required": True,
        "attribution_text": '"T" by A is licensed under CC BY 4.0 (https://e.org/x).'})  # fmt: skip
    out: dict[str, Any] = {"root": root}
    for key, track in {"none": None, "plain": "music/plain.m4a", "cc0": "music/cc0.m4a",
                       "by": "music/by.m4a"}.items():  # fmt: skip
        raw = _ad(root, track)
        raw["name"] = f"ad_{key}"
        raw["output"]["path"] = f"renders/ad_{key}.mp4"
        report = compose(Storyboard.model_validate(raw), root=root)
        out[key] = {"raw": raw, "report": report}
    return out


def test_the_render_report_carries_the_music_licence(renders) -> None:
    none = renders["none"]["report"]
    assert none.music is None and not [w for w in none.warnings if w.startswith("music_")]
    plain = renders["plain"]["report"]
    assert plain.music["status"] == "unknown"
    assert any(w.startswith("music_licence_unknown") for w in plain.warnings)
    cc0 = renders["cc0"]["report"]
    assert cc0.music["licence"] == "CC0 1.0" and not cc0.music["attribution_required"]
    assert not [w for w in cc0.warnings if w.startswith("music_")]
    by = renders["by"]["report"]
    assert by.music["attribution_required"] and by.music["licence"] == "CC BY 4.0"
    warning = next(w for w in by.warnings if w.startswith("music_attribution_required"))
    assert by.music["attribution_text"] in warning
    json.dumps(by.model_dump(mode="json"))


def test_ad_score_carries_the_music_licence(renders) -> None:
    root = renders["root"]

    def score(key: str):
        return ad_score(root, Path(renders[key]["report"].path), storyboard=renders[key]["raw"])

    plain = score("plain")
    finding = next(f for f in plain.findings if f["id"] == "music_licence_unknown")
    assert finding["area"] == "audio" and finding["detail"]["path"] == "music/plain.m4a"
    assert plain.facts["audio"]["music_licence"]["status"] == "unknown"
    by = score("by")
    attribution = next(f for f in by.findings if f["id"] == "music_attribution_required")
    assert "CC BY 4.0" in attribution["detail"]["attribution_text"]
    assert attribution["impact"] == 0.0
    cc0 = score("cc0")
    assert "audio.music_licence" in cc0.passed
    assert not [f for f in cc0.findings if f["id"].startswith("music_")]
    assert cc0.facts["audio"]["music_licence"]["licence"] == "CC0 1.0"
    assert score("none").facts.get("audio", {}).get("music_licence") is None


# ------------------------------------------------------------------------------ manifest


def _entry(name: str, label: str) -> dict[str, Any]:
    return {"status": "done", "file": f"renders/{name}.mp4", "duration": 5.0, "score": 80,
            "ids": {}, "format": "9:16", "hook_text": "h", "cta_text": "c",
            "music_licence": label}  # fmt: skip


def test_manifest_has_a_music_licence_column(tmp_path: Path) -> None:
    path = write_manifest(
        tmp_path, "b", [_entry("a", "CC0 1.0"), _entry("b", "unknown"),
                        _entry("c", "CC BY 4.0; attribution required")]
    )  # fmt: skip
    rows = list(csv.reader(path.read_text("utf-8").splitlines()))
    assert rows[0] == MANIFEST_COLUMNS and rows[0][-2] == "music licence"
    assert [row[-2] for row in rows[1:]] == [
        "CC0 1.0",
        "unknown",
        "CC BY 4.0; attribution required",
    ]


def test_a_manifest_without_the_column_is_migrated(tmp_path: Path) -> None:
    folder = tmp_path / "renders" / "old"
    folder.mkdir(parents=True)
    columns = [c for c in MANIFEST_COLUMNS if c not in ("music licence", "quality")]
    with (folder / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow(["old.mp4", "", "9:16", "5.00", "80", "h", "", "c"])
    path = write_manifest(tmp_path, "old", [_entry("new", "CC0 1.0")])
    rows = list(csv.reader(path.read_text("utf-8").splitlines()))
    assert rows[1][0] == "old.mp4" and rows[1][-2] == "" and rows[2][-2] == "CC0 1.0"


# ------------------------------------------------------------------------------ MCP tools


def test_the_tools_are_served(network, tmp_path: Path) -> None:
    init_project(tmp_path, "kettles")

    async def run() -> None:
        async with Client(build_server()) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            assert {"music_search", "music_fetch"} <= tools
            found = await client.call_tool(
                "music_search", {"project_dir": str(tmp_path), "query": "upbeat acoustic"}
            )
            data = json.loads(found.content[0].text)
            assert data["count"] == 4
            fetched = await client.call_tool(
                "music_fetch",
                {"project_dir": str(tmp_path), "source": "openverse", "id": "aaaaaaaa-0001"},
            )
            assert json.loads(fetched.content[0].text)["licence"] == "CC0 1.0"
            bad = await client.call_tool(
                "music_fetch",
                {"project_dir": str(tmp_path), "source": "openverse", "id": "aaaaaaaa-0004"},
            )
            assert bad.is_error

    anyio.run(run)


def test_music_feel_is_a_storyboard_field() -> None:
    board = Storyboard.model_validate(
        {"name": "x", "format": FORMAT, "emoji": "none",
         "segments": [{"type": "image", "source": "a.png", "duration": 2.0}],
         "audio": {"music_feel": "energetic"}, "output": {"path": "renders/x.mp4"}}
    )  # fmt: skip
    assert board.audio.music_feel == "energetic"


# ------------------------------------------------------------------ local generation (mocked)

FAKE_WORKER = r"""
import json, sys, wave
import numpy as np
request = json.load(open(sys.argv[1]))
cr = chr(13)
sys.stderr.write("  0%|          | 0/4 [00:01<00:03]" + cr + " 50%|#####   | 2/4 [00:02]" + cr)
sys.stderr.flush()
t = np.arange(int(48000 * request["duration"])) / 48000
data = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
with wave.open(request["out"], "wb") as handle:
    handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(48000)
    handle.writeframes(data.tobytes())
print("ADANVIL_RESULT " + json.dumps({"ok": True, "out": request["out"], "seconds": 1.5}))
"""


def test_music_setup_plan_installs_cpu_torch_and_a_pinned_ace_step(tmp_path: Path) -> None:
    from adanvil import musicgen

    plan = musicgen.music_setup(tmp_path / "env", plan=True)
    flat = [" ".join(command) for command in plan]
    assert any("torch==2.6.0" in line for line in flat)
    assert any(f"ACE-Step.git@{musicgen.ACE_STEP_REF}" in line and "--no-deps" in line
               for line in flat)  # fmt: skip
    assert not any("gradio" in line for line in flat)


def test_generate_without_the_environment_says_how_to_install_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adanvil import musicgen

    monkeypatch.setenv("ADANVIL_CACHE", str(tmp_path / "cache"))
    init_project(tmp_path / "p", "kettles")
    with pytest.raises(musicgen.MusicUnavailableError, match="adanvil music-setup"):
        musicgen.generate_music(tmp_path / "p", "upbeat acoustic", 20)
    with pytest.raises(music.MusicError, match="duration"):
        musicgen.generate_music(tmp_path / "p", "x", 500)


def test_generate_writes_the_track_and_its_licence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from adanvil import musicgen
    monkeypatch.setenv("ADANVIL_MUSIC_MIN_FREE_GB", "0")  # fake engine: no real memory need

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "music_config.json").write_text(json.dumps({"python": sys.executable}), "utf-8")
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER, encoding="utf-8")
    monkeypatch.setenv("ADANVIL_CACHE", str(cache))
    monkeypatch.setattr(musicgen, "WORKER", worker)
    root = tmp_path / "p"
    init_project(root, "kettles")
    events: list[dict] = []
    result = musicgen.generate_music(
        root, "Upbeat acoustic guitar, warm", 10, seed=7, progress=events.append
    )
    assert result["path"].startswith("music/gen-") and result["path"].endswith("-7.flac")
    assert (root / result["path"]).is_file() and not list((root / "cache").rglob("*.wav"))
    assert result["attribution_required"] is False and 9 < result["duration"] < 11
    sidecar = json.loads((root / (result["path"] + ".licence.json")).read_text("utf-8"))
    assert sidecar["source"] == "generated" and sidecar["licence_code"] == "apache-2.0"
    assert sidecar["model"] == musicgen.MODEL_ID and sidecar["seed"] == 7
    assert sidecar["prompt"] == "Upbeat acoustic guitar, warm"
    assert "Apache-2.0" in (root / "music" / "LICENCES.txt").read_text("utf-8")
    info = music.music_licence(root, result["path"])
    assert info["status"] == "recorded" and info["warnings"] == []
    assert any(e.get("done") == 2 and e.get("total") == 4 for e in events)


def test_music_generate_tool_refuses_without_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADANVIL_CACHE", str(tmp_path / "cache"))
    init_project(tmp_path / "p", "kettles")

    async def run() -> None:
        async with Client(build_server()) as client:
            refused = await client.call_tool(
                "music_generate", {"project_dir": str(tmp_path / "p"), "prompt": "calm piano"}
            )
            assert refused.is_error and "music-setup" in refused.content[0].text

    anyio.run(run)


def test_music_generation_refused_when_memory_is_short(monkeypatch: pytest.MonkeyPatch) -> None:
    from adanvil import musicgen

    class _Meminfo:
        def __init__(self, *_: Any) -> None:
            pass

        def read_text(self, encoding: str = "ascii") -> str:
            return "MemTotal:       23000000 kB\nMemAvailable:    8388608 kB\n"

    monkeypatch.setattr(musicgen, "Path", _Meminfo)
    monkeypatch.delenv("ADANVIL_MUSIC_MIN_FREE_GB", raising=False)
    with pytest.raises(musicgen.MusicUnavailableError, match="8.0 GB is free"):
        musicgen._check_memory()
    monkeypatch.setenv("ADANVIL_MUSIC_MIN_FREE_GB", "4")
    musicgen._check_memory()
