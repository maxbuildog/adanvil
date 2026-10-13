from __future__ import annotations

import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from adanvil.jobs import _ProjectLock, read_job, start_job
from adanvil.library import annotate, index_library, load_catalog
from adanvil.library.catalog import (
    CATALOG_LOCK_FILE,
    Entry,
    merge_entries,
    merge_into_catalog,
    update_catalog,
)
from adanvil.project import init_project, resolve_in_project
from adanvil.voice.engines import Alignment, FakeEngine
from adanvil.voice.generate import VoiceSpec, _unmatched_words, generate_voice


def _entry(name: str, indexed: datetime, **annotation: object) -> Entry:
    return Entry.model_validate(
        {
            "id": name,
            "path": f"footage/{name}.mp4",
            "kind": "video",
            "file_key": {},
            "indexed": indexed.isoformat(),
            "annotation": annotation,
        }
    )


def _wait(job_file: Path, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = read_job(job_file)
        if record.get("status") in {"done", "failed"}:
            return record
        time.sleep(0.1)
    return read_job(job_file)


# Item 1: catalog writes are read-merge-write under a lock.


def test_stale_snapshots_merge_instead_of_overwriting(tmp_path: Path) -> None:
    init_project(tmp_path, "Merge")
    now = datetime.now(UTC)
    merge_into_catalog(tmp_path, [_entry("a", now)])
    merge_into_catalog(tmp_path, [_entry("b", now)])
    assert set(load_catalog(tmp_path).entries) == {"a", "b"}

    merge_into_catalog(tmp_path, removed=["a"])
    assert set(load_catalog(tmp_path).entries) == {"b"}


def test_merge_keeps_newer_measurement_and_newer_annotation() -> None:
    old = datetime(2026, 1, 1, tzinfo=UTC)
    new = old + timedelta(hours=1)
    stored = _entry("a", old, description="annotated", updated=new.isoformat())
    incoming = _entry("a", new, description="stale copy", updated=old.isoformat())
    merged = merge_entries(stored, incoming)
    assert merged.indexed == incoming.indexed
    assert merged.annotation.description == "annotated"

    plain = _entry("a", old)
    assert merge_entries(plain, _entry("a", old)).annotation.description is None
    assert merge_entries(incoming, stored).indexed == incoming.indexed


def _three_writers(root: Path) -> None:
    """Three processes add 25 entries each; the lock must keep all 75."""

    code = (
        "import sys; from pathlib import Path; from adanvil.library.catalog import "
        "merge_into_catalog, Entry; from datetime import datetime, UTC; "
        "root = Path(sys.argv[1]); tag = sys.argv[2]; "
        "[merge_into_catalog(root, [Entry.model_validate({'id': f'{tag}{i}', "
        "'path': f'footage/{tag}{i}.mp4', 'kind': 'video', 'file_key': {}, "
        "'indexed': datetime.now(UTC).isoformat()})]) for i in range(25)]"
    )
    processes = [subprocess.Popen([sys.executable, "-c", code, str(root), tag]) for tag in "xyz"]
    assert [process.wait(timeout=120) for process in processes] == [0, 0, 0]
    assert len(load_catalog(root).entries) == 75


def test_update_catalog_is_exclusive(tmp_path: Path) -> None:
    init_project(tmp_path, "Lock")
    _three_writers(tmp_path)


@pytest.mark.parametrize("size", [2, 5])
def test_update_catalog_is_exclusive_whatever_the_lock_file_holds(
    tmp_path: Path, size: int
) -> None:
    # Two processes creating the lock file at the same moment each used to append a byte,
    # and a lock file longer than one byte let several writers in together (55 entries
    # kept out of 75). The lock must not depend on what the file holds.
    init_project(tmp_path, "Lock")
    (tmp_path / CATALOG_LOCK_FILE).write_bytes(b"\0" * size)
    _three_writers(tmp_path)


@pytest.mark.parametrize("size", [0, 1, 2, 5])
def test_project_lock_excludes_whatever_the_lock_file_holds(tmp_path: Path, size: int) -> None:
    path = tmp_path / "catalog.json.lock"
    path.write_bytes(b"\0" * size)
    first = _ProjectLock(path, timeout=5)
    first.acquire()
    with pytest.raises(TimeoutError):
        _ProjectLock(path, timeout=0.3, poll=0.02).acquire()
    first.release()
    again = _ProjectLock(path, timeout=5)
    again.acquire()
    again.release()
    assert path.stat().st_size == size  # taking the lock never writes to the file


def test_concurrent_index_processes_keep_all_entries(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "Concurrent")
    for folder, name in [("one", "ref_ad.mp4"), ("two", "sdr_portrait.mp4")]:
        (tmp_path / "footage" / folder).mkdir()
        shutil.copy2(Path(str(media[name]["path"])), tmp_path / "footage" / folder / name)
    code = (
        "import sys; from pathlib import Path; from adanvil.library import index_library; "
        "index_library(Path(sys.argv[1]), sys.argv[2], text_scan=False)"
    )
    processes = [
        subprocess.Popen([sys.executable, "-c", code, str(tmp_path), f"footage/{folder}"])
        for folder in ("one", "two")
    ]
    assert [process.wait(timeout=300) for process in processes] == [0, 0]
    paths = sorted(entry.path for entry in load_catalog(tmp_path).entries.values())
    assert paths == ["footage/one/ref_ad.mp4", "footage/two/sdr_portrait.mp4"]


def test_annotation_made_during_an_index_survives(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "Annotate")
    for name in ("ref_ad.mp4", "sdr_portrait.mp4"):
        shutil.copy2(Path(str(media[name]["path"])), tmp_path / "footage" / name)
    annotated: list[str] = []

    def progress(message: str) -> None:
        # The first file is already merged; annotate it while the second one is measured.
        if message.startswith("indexed ") and not annotated:
            identifier = next(iter(load_catalog(tmp_path).entries))
            annotate(tmp_path, identifier, description="done during index", quality=4)
            annotated.append(identifier)

    index_library(tmp_path, text_scan=False, progress=progress)
    catalog = load_catalog(tmp_path)
    assert len(catalog.entries) == 2
    assert catalog.entries[annotated[0]].annotation.description == "done during index"

    index_library(tmp_path, text_scan=False)  # re-run: nothing changed, annotation kept
    assert load_catalog(tmp_path).entries[annotated[0]].annotation.quality == 4


def test_annotate_unknown_entry_does_not_touch_the_catalog(tmp_path: Path) -> None:
    init_project(tmp_path, "Unknown")
    with pytest.raises(KeyError, match="unknown catalog entry"):
        annotate(tmp_path, "nope", description="x")
    update_catalog(tmp_path, lambda catalog: None)
    assert load_catalog(tmp_path).entries == {}


# Item 2: names that differ only by Unicode normalisation form.


def _make_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    if path.name not in {item.name for item in path.parent.iterdir()}:
        pytest.skip("this filesystem does not keep the Unicode form of file names")


def test_nfd_name_on_disk_is_found_from_nfc_request(tmp_path: Path) -> None:
    folder = unicodedata.normalize("NFD", "Clips été")
    name = unicodedata.normalize("NFD", "Plans canapé.mov")
    _make_file(tmp_path / "footage" / folder / name)

    requested = unicodedata.normalize("NFC", f"footage/{folder}/{name}")
    found = resolve_in_project(tmp_path, requested)
    assert found.is_file()
    assert found.parent.name == folder and found.name == name  # the real on-disk spelling

    mixed = f"footage/{unicodedata.normalize('NFC', folder)}/{name}"
    assert resolve_in_project(tmp_path, mixed).is_file()


def test_nfc_name_on_disk_is_found_from_nfd_request(tmp_path: Path) -> None:
    folder = unicodedata.normalize("NFC", "Clips été")
    name = unicodedata.normalize("NFC", "Plans canapé.mov")
    _make_file(tmp_path / "footage" / folder / name)

    requested = unicodedata.normalize("NFD", f"footage/{folder}/{name}")
    found = resolve_in_project(tmp_path, requested)
    assert found.is_file()
    assert found.name == name


def test_missing_name_still_resolves_without_error(tmp_path: Path) -> None:
    (tmp_path / "footage").mkdir()
    assert resolve_in_project(tmp_path, "footage/absent é.mov") == (
        tmp_path / "footage" / "absent é.mov"
    )
    assert not resolve_in_project(tmp_path, "nothing/at/all.mov").exists()


def test_indexed_nfd_file_resolves_from_its_catalog_path(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "Unicode")
    name = unicodedata.normalize("NFD", "Plans canapé.mp4")
    target = tmp_path / "footage" / name
    shutil.copy2(Path(str(media["sdr_portrait.mp4"]["path"])), target)
    if name not in {item.name for item in target.parent.iterdir()}:
        pytest.skip("this filesystem does not keep the Unicode form of file names")

    index_library(tmp_path, text_scan=False)
    entry = next(iter(load_catalog(tmp_path).entries.values()))
    assert resolve_in_project(tmp_path, entry.path).is_file()
    assert resolve_in_project(tmp_path, entry.path).name == name


# Item 3: index jobs do not wait for the project lock.


def test_index_job_completes_while_a_voice_job_holds_the_project_lock(
    tmp_path: Path, media: dict[str, dict[str, object]]
) -> None:
    init_project(tmp_path, "Parallel")
    shutil.copy2(Path(str(media["sdr_portrait.mp4"]["path"])), tmp_path / "footage" / "clip.mp4")
    slow_voice = _ProjectLock(tmp_path / "renders" / "jobs" / ".lock")
    slow_voice.acquire()
    try:
        waiting_voice = start_job(
            tmp_path,
            "voice",
            {"name": "late", "spec": {"engine": "fake", "lines": ["one"]}},
            progress={"total": 1, "unit": "lines"},
        )
        index = start_job(tmp_path, "index", {"source": "footage"}, progress={"total": 1})
        result = _wait(Path(index["job_file"]), timeout=180)

        assert result["status"] == "done", result
        assert result["index"]["counts"]["added"] == 1
        assert read_job(Path(waiting_voice["job_file"]))["status"] != "done"
    finally:
        slow_voice.release()
    assert _wait(Path(waiting_voice["job_file"]))["status"] == "done"


# Item 4: word-level pronunciation check.


class _HeardEngine(FakeEngine):
    """A fake engine whose transcriber hears something else for chosen lines."""

    def __init__(self, heard: dict[str, str]) -> None:
        super().__init__()
        self.heard = heard

    def align(self, paths: list[Path], language: str, model: str) -> list[Alignment]:
        results = super().align(paths, language, model)
        for path, result in zip(paths, results, strict=True):
            expected = self._text.get(Path(path).resolve(), "")
            result.text = self.heard.get(expected, result.text)
        return results


def test_unmatched_words_rules() -> None:
    assert _unmatched_words("Il coûte quarante-neuf euros", "il coûte 49 euros") == []
    assert _unmatched_words("Découvre l'application", "découvre lapplication") == []
    assert _unmatched_words("Un petit pas, peut-être", "un petit pas peutetre") == []
    assert _unmatched_words("le Brio est là", "le Briao est là") == []  # ratio 0.89
    assert _unmatched_words("le Brio est là", "le Breeoh est là") == ["brio"]
    assert _unmatched_words(
        "En ce moment, le Brio est à prix réduit", "le brioche est à pré-réduit"
    ) == ["moment", "brio", "prix"]


def test_two_unmatched_key_words_make_a_line_suspect_despite_high_similarity(
    tmp_path: Path,
) -> None:
    line = "Le Brio garde le café chaud toute la journée, à prix doux, en trente jours"
    heard = "le brio garde le cave chaud toute la journée à pris doux en 30 jours"
    engine = _HeardEngine({line: heard})
    track = generate_voice(tmp_path, VoiceSpec(engine="fake", lines=[line]), "two", engine=engine)

    result = track.lines[0]
    assert result.similarity >= 0.85
    assert result.unmatched_words == ["cafe", "prix"]
    assert result.status == "suspect"
    assert result.attempts == 3  # retried like any suspect line
    assert any("cafe, prix" in warning for warning in track.warnings)


def test_one_unmatched_brand_name_is_reported_but_stays_ok(tmp_path: Path) -> None:
    line = "Le Brio garde le café chaud en trente jours"
    engine = _HeardEngine({line: "le breeoh garde le cafe chaud en 30 jours"})
    track = generate_voice(tmp_path, VoiceSpec(engine="fake", lines=[line]), "one", engine=engine)

    assert track.lines[0].unmatched_words == ["brio"]
    assert track.lines[0].status == "ok"
    assert track.lines[0].attempts == 1


def test_unmatched_words_are_surfaced_in_summaries(tmp_path: Path) -> None:
    from adanvil.jobs import _voice_summary
    from adanvil.server import _voice_summary as server_summary

    line = "Le Brio garde le café chaud en trente jours"
    track = generate_voice(
        tmp_path,
        VoiceSpec(engine="fake", lines=[line]),
        "view",
        engine=_HeardEngine({line: "le breeoh garde le cafe chaud en 30 jours"}),
    )
    assert _voice_summary(tmp_path, track)["lines"][0]["unmatched_words"] == ["brio"]
    assert server_summary(tmp_path, track)["lines"][0]["unmatched_words"] == ["brio"]


def test_a_longer_word_does_not_hide_a_short_brand_name() -> None:
    from adanvil.voice.generate import _unmatched_words

    expected = "En ce moment, le Brio est à prix réduit."
    heard = "En ce moment, le brioche est à pré-réduit."
    assert _unmatched_words(expected, heard) == ["brio", "prix"]
    # Elisions still count as heard.
    assert _unmatched_words("Ouvre application", "Ouvre l'application") == []
