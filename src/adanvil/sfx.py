"""Sound effects: the CC0 pack, deterministic synthesis, and the project's ``sfx/`` folder.

The pack is described in ``assets/sfx_pack.json``. No sound is stored in the repository: each
entry is either downloaded from a pinned CC0 source (and verified by the SHA-1 of the file it
comes from) or synthesised in code, and every sound goes through the same finishing: trimmed,
faded at both ends and set to one level (the loudest 400 ms at -16 LUFS, the peak never above
-1.5 dBFS). A sound whose download fails is synthesised instead and marked ``generated``.
Users can also drop their own files in ``sfx/``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import wave
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from adanvil import __version__, netguard
from adanvil.ffmpeg import probe, run
from adanvil.project import resolve_in_project
from adanvil.safeio import replace_bytes, replace_text

SFX_DIR = "sfx"
SAMPLE_RATE = 48_000
TARGET_LUFS = -16.0  # the loudest 400 ms of a finished sound
PEAK_CEILING_DB = -1.5  # the highest sample of a finished sound
SFX_SUFFIXES = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus")
MANIFEST_PATH = Path(__file__).with_name("assets") / "sfx_pack.json"
USER_AGENT = f"AdAnvil-sfx/{__version__} (https://github.com/maxbuildog/adanvil)"
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # the pinned archives are a few MB; the SHA-1 decides


def load_manifest() -> list[dict[str, Any]]:
    """The entries of the pack: ``name``, ``description``, ``generator`` and, for a sound that
    comes from a CC0 source, ``download`` (pinned URL, SHA-1, member, licence)."""

    return list(json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["sounds"])


# ---------------------------------------------------------------------------------------------
# Level: K-weighted loudness of a short sound, and the finishing every sound of the pack gets.
# ---------------------------------------------------------------------------------------------

_K_SHELF = ((1.53512485958697, -2.69169618940638, 1.19839281085285),
            (1.0, -1.69065929318241, 0.73248077421585))
_K_HIGHPASS = ((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621))


def _biquad(samples: np.ndarray, coefficients: tuple[tuple[float, ...], tuple[float, ...]]):
    (b0, b1, b2), (_, a1, a2) = coefficients
    out = [0.0] * len(samples)
    x1 = x2 = y1 = y2 = 0.0
    for index, x0 in enumerate(samples.tolist()):
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[index] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return np.asarray(out)


def momentary_max_lufs(samples: np.ndarray) -> float:
    """Loudest 400 ms window (K-weighted, ITU-R BS.1770 at 48 kHz) of a mono signal, in LUFS.

    A sound shorter than 400 ms is measured as a meter would hear it: alone in a silent
    window, so a 30 ms click reads far below its peak.
    """

    if len(samples) == 0 or float(np.max(np.abs(samples))) == 0.0:
        return -120.0
    weighted = _biquad(_biquad(np.asarray(samples, dtype=float), _K_SHELF), _K_HIGHPASS)
    window = int(0.4 * SAMPLE_RATE)
    hop = int(0.1 * SAMPLE_RATE)
    squared = weighted**2
    if len(squared) < window:
        squared = np.concatenate([squared, np.zeros(window - len(squared))])
    loudest = 0.0
    for start in range(0, len(squared) - window + 1, hop):
        loudest = max(loudest, float(squared[start : start + window].mean()))
    if loudest <= 0:
        return -120.0
    return -0.691 + 10.0 * math.log10(loudest)


def peak_db(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    return -120.0 if peak <= 0 else 20.0 * math.log10(peak)


def finish(
    samples: np.ndarray, *, trim_start: bool = True
) -> tuple[np.ndarray, dict[str, float]]:
    """Trim, fade and level a mono signal; return it with what was measured.

    ``trim_start=False`` keeps the quiet beginning (a riser must keep its nominal length: it
    is placed so that its end meets the moment it builds towards).
    """

    x = np.asarray(samples, dtype=float)
    x = x - float(np.mean(x[: min(len(x), 64)])) if len(x) else x
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= 0:
        raise ValueError("the sound is silent")
    loud = np.flatnonzero(np.abs(x) > peak * 10 ** (-50 / 20))
    start = max(0, int(loud[0]) - int(0.002 * SAMPLE_RATE)) if trim_start else 0
    tail = np.flatnonzero(np.abs(x) > peak * 10 ** (-60 / 20))
    end = min(len(x), int(tail[-1]) + int(0.005 * SAMPLE_RATE))
    x = x[start:end].copy()
    fade_in = max(1, int(0.001 * SAMPLE_RATE))
    fade_out = min(max(1, int(0.008 * SAMPLE_RATE)), max(1, len(x) // 5))
    x[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
    x[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    gain_db = min(TARGET_LUFS - momentary_max_lufs(x), PEAK_CEILING_DB - peak_db(x), 40.0)
    x = x * 10 ** (gain_db / 20)
    return x, {
        "duration": round(len(x) / SAMPLE_RATE, 3),
        "peak_dbfs": round(peak_db(x), 2),
        "momentary_max_lufs": round(momentary_max_lufs(x), 2),
    }


def write_wav(path: Path, samples: np.ndarray) -> None:
    """Mono 16-bit PCM at 48 kHz, written atomically."""

    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    replace_bytes(path, buffer.getvalue())


# ---------------------------------------------------------------------------------------------
# Synthesis: sine, noise and envelopes only. Same code, same bytes (the noise is a seeded PCG64).
# ---------------------------------------------------------------------------------------------


def _time(seconds: float) -> np.ndarray:
    return np.arange(int(round(seconds * SAMPLE_RATE))) / SAMPLE_RATE


def _noise(name: str, count: int) -> np.ndarray:
    return np.random.default_rng(zlib.crc32(name.encode("utf-8"))).standard_normal(count)


def _svf(samples: np.ndarray, centre: np.ndarray, q: float, mode: str = "band") -> np.ndarray:
    """State-variable filter with a centre frequency per sample (``band``, ``low``, ``high``)."""

    gains = np.tan(np.pi * np.clip(centre, 20.0, SAMPLE_RATE * 0.45) / SAMPLE_RATE).tolist()
    k = 1.0 / q
    out = [0.0] * len(samples)
    ic1 = ic2 = 0.0
    for index, (x, g) in enumerate(zip(samples.tolist(), gains, strict=True)):
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        v3 = x - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + g * a2 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == "band":
            out[index] = k * v1
        elif mode == "low":
            out[index] = v2
        else:
            out[index] = x - k * v1 - v2
    return np.asarray(out)


def _shape(count: int, peak: float, rise: float, fall: float) -> np.ndarray:
    """Envelope that climbs to 1 at ``peak`` (0 to 1 of the length) and falls back to 0."""

    u = np.linspace(0.0, 1.0, count)
    up = np.clip(u / max(peak, 1e-6), 0.0, 1.0) ** rise
    down = np.clip((1.0 - u) / max(1.0 - peak, 1e-6), 0.0, 1.0) ** fall
    return np.where(u < peak, up, down)


def _sweep(start: float, end: float, count: int) -> np.ndarray:
    return start * (end / start) ** np.linspace(0.0, 1.0, count)


def _damped(t: np.ndarray, frequency: float, tau: float, start: float = 0.0) -> np.ndarray:
    """A sine that dies away exponentially from ``start`` (silent before it)."""

    local = np.clip(t - start, 0.0, None)
    ramp = 1.0 - np.exp(-local / 0.0008)
    ring = np.sin(2 * np.pi * frequency * local) * np.exp(-local / tau) * ramp
    return np.where(t >= start, ring, 0.0)


def _burst(t: np.ndarray, name: str, start: float, tau: float, frequency: float, amount: float):
    """A mechanical tick: a burst of high-passed noise and a short ringing at ``frequency``."""

    hiss = np.diff(_noise(name + f"@{start:.4f}", len(t) + 1))
    local = np.clip(t - start, 0.0, None)
    envelope = np.where(t >= start, np.exp(-local / tau), 0.0)
    return amount * (0.7 * hiss * envelope + 0.6 * _damped(t, frequency, tau * 1.8, start))


def _bell(t: np.ndarray, frequency: float, start: float, tau: float, amount: float = 1.0):
    """A struck bell: three inharmonic partials, the high ones dying faster."""

    total = np.zeros(len(t))
    for ratio, weight, decay in ((1.0, 1.0, 1.0), (2.76, 0.4, 0.5), (5.4, 0.15, 0.25)):
        total += weight * _damped(t, frequency * ratio, tau * decay, start)
    return amount * total


def _whoosh(name: str, seconds: float, f0: float, f1: float, peak: float, q: float,
            rise: float = 1.7, fall: float = 2.2) -> np.ndarray:
    t = _time(seconds)
    body = _svf(_noise(name, len(t)), _sweep(f0, f1, len(t)), q)
    return body * _shape(len(t), peak, rise, fall)


def _riser(name: str, seconds: float) -> np.ndarray:
    t = _time(seconds)
    u = t / seconds
    hiss = _svf(_noise(name, len(t)), _sweep(250.0, 7000.0, len(t)), 2.5)
    phase = np.cumsum(2 * np.pi * _sweep(220.0, 1320.0, len(t)) / SAMPLE_RATE)
    tone = np.sin(phase) + 0.5 * np.sin(2 * phase)
    return (hiss + 0.35 * tone) * u**2.0


def _typing(name: str) -> np.ndarray:
    t = _time(1.3)
    rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")))
    total = np.zeros(len(t))
    moment = 0.02
    for number in range(9):
        amount = float(rng.uniform(0.6, 1.0))
        frequency = float(rng.uniform(1100.0, 2400.0))
        total += _burst(t, f"{name}{number}", moment, 0.005, frequency, amount)
        total += 0.4 * amount * _damped(t, 180.0, 0.012, moment)
        moment += float(rng.uniform(0.09, 0.2))
    return total


def _record_scratch(name: str) -> np.ndarray:
    t = _time(0.6)
    u = t / 0.6
    phase = np.cumsum(2 * np.pi * (1400.0 * np.exp(-3.0 * u) + 60.0) / SAMPLE_RATE)
    saw = 2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0
    grit = _svf(_noise(name, len(t)), np.full(len(t), 1800.0), 1.2)
    wobble = 0.55 + 0.45 * np.sin(2 * np.pi * 9.0 * t)
    envelope = np.minimum(t / 0.006, 1.0) * np.clip((0.6 - t) / 0.18, 0.0, 1.0)
    return (0.5 * saw + 0.9 * grit) * wobble * envelope


def _pop(name: str) -> np.ndarray:
    t = _time(0.16)
    phase = np.cumsum(2 * np.pi * (150.0 + 650.0 * np.exp(-t / 0.018)) / SAMPLE_RATE)
    body = np.sin(phase) * np.exp(-t / 0.03) * (1.0 - np.exp(-t / 0.0008))
    return body + 0.3 * _noise(name, len(t)) * np.exp(-t / 0.0015)


def _thud(name: str, seconds: float, top: float, bottom: float, drop: float, tau: float,
          tail: float) -> np.ndarray:
    t = _time(seconds)
    phase = np.cumsum(2 * np.pi * (bottom + (top - bottom) * np.exp(-t / drop)) / SAMPLE_RATE)
    body = np.sin(phase) * np.exp(-t / tau) * (1.0 - np.exp(-t / 0.002))
    rumble = _svf(_noise(name, len(t)), np.full(len(t), 140.0), 0.7) * np.exp(-t / tail)
    return body + 0.5 * rumble


def _blip(t: np.ndarray, frequency: float, start: float) -> np.ndarray:
    return _damped(t, frequency, 0.07, start) + 0.2 * _damped(t, frequency * 2, 0.05, start)


def _generate(name: str) -> np.ndarray:
    """The synthesised version of a sound of the pack (also the fallback of a download)."""

    if name in {"whoosh_short", "whoosh_long", "swoosh", "swipe"}:
        settings = {
            "whoosh_short": (0.45, 250.0, 1800.0, 0.40, 0.9, 1.7, 2.2),
            "whoosh_long": (1.1, 200.0, 1500.0, 0.55, 0.9, 1.9, 2.0),
            "swoosh": (0.65, 5000.0, 600.0, 0.35, 1.0, 1.5, 2.4),
            "swipe": (0.3, 1500.0, 6500.0, 0.30, 1.8, 1.2, 2.5),
        }[name]
        return _whoosh(name, *settings)
    if name == "riser_1s":
        return _riser(name, 1.0)
    if name == "riser_2s":
        return _riser(name, 2.0)
    if name == "pop":
        return _pop(name)
    if name == "click":
        t = _time(0.05)
        return _burst(t, name, 0.0, 0.0025, 2400.0, 1.0)
    if name == "tap":
        t = _time(0.12)
        return _damped(t, 420.0, 0.014) + 0.3 * _burst(t, name, 0.0, 0.002, 1800.0, 1.0)
    if name == "ding":
        return _bell(_time(1.1), 1568.0, 0.0, 0.4)
    if name == "chime":
        t = _time(1.6)
        return _bell(t, 1318.5, 0.0, 0.55) + _bell(t, 1760.0, 0.14, 0.7, 0.9)
    if name == "notification":
        t = _time(0.55)
        return _blip(t, 1174.7, 0.0) + _blip(t, 1568.0, 0.13)
    if name == "impact":
        return _thud(name, 0.7, 135.0, 45.0, 0.06, 0.16, 0.04)
    if name == "boom":
        return _thud(name, 1.4, 110.0, 35.0, 0.25, 0.45, 0.12)
    if name == "camera_shutter":
        t = _time(0.25)
        return _burst(t, name, 0.0, 0.004, 2400.0, 1.0) + _burst(t, name, 0.075, 0.006, 1500.0, 0.8)
    if name == "cash_register":
        t = _time(1.2)
        slide = _svf(_noise(name, len(t)), np.full(len(t), 2500.0), 0.8) * (
            np.where((t > 0.03) & (t < 0.12), 0.15, 0.0)
        )
        return (
            _burst(t, name, 0.0, 0.015, 900.0, 1.0)
            + slide
            + _bell(t, 2349.0, 0.14, 0.3, 0.8)
            + _bell(t, 3136.0, 0.19, 0.3, 0.6)
        )
    if name == "record_scratch":
        return _record_scratch(name)
    if name == "typing":
        return _typing(name)
    raise KeyError(f"no generator for the sound {name!r}")


def generate_sound(name: str) -> tuple[np.ndarray, dict[str, float]]:
    """The finished synthesised sound ``name`` (deterministic: the same call, the same samples)."""

    return finish(_generate(name), trim_start=not name.startswith("riser"))


# ---------------------------------------------------------------------------------------------
# Download of the CC0 sources, with the verification the demo pack uses (SHA-1 of the archive).
# ---------------------------------------------------------------------------------------------


class SfxError(RuntimeError):
    """An actionable error while building the sound pack."""


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, sha1: str, destination: Path, *, timeout: float = 60.0) -> Path:
    """Stream ``url`` to ``destination`` and keep it only when its SHA-1 is ``sha1``.

    The download goes through ``adanvil.netguard`` (public http(s) hosts, at most 200 MB)."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha1(destination) == sha1:
        return destination
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    try:
        netguard.fetch_to_file(
            url,
            partial,
            timeout=timeout,
            max_bytes=MAX_DOWNLOAD_BYTES,
            headers={"User-Agent": USER_AGENT},
        )
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise SfxError(f"could not download {url}: {exc}") from exc
    actual = _sha1(partial)
    if actual != sha1:
        partial.unlink(missing_ok=True)
        raise SfxError(f"SHA-1 mismatch for {url}: expected {sha1}, got {actual}; discarded")
    os.replace(partial, destination)
    return destination


def _decode_member(archive: Path, member: str) -> np.ndarray:
    """One audio file of a ZIP archive, decoded to mono 48 kHz floats."""

    with tempfile.TemporaryDirectory(prefix="adanvil-sfx-") as scratch:
        extracted = Path(scratch) / Path(member).name
        with zipfile.ZipFile(archive) as bundle:
            replace_bytes(extracted, bundle.read(member))
        decoded = run(
            ["-i", str(extracted), "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-f", "f32le", "-"],
            capture_stdout=True,
        )
    return np.frombuffer(decoded.stdout or b"", dtype="<f4").astype(float)


# ---------------------------------------------------------------------------------------------
# The pack in a project.
# ---------------------------------------------------------------------------------------------


def sfx_folder(root: Path) -> Path:
    return Path(root) / SFX_DIR


def _licence_text(record: dict[str, Any]) -> str:
    if record["source"] == "generated":
        note = record.get("note")
        return f"{record['name']}: generated by AdAnvil (synthesised, no licence needed)" + (
            f" [{note}]" if note else ""
        )
    return (
        f"{record['name']}: {record['author']}, {record['pack']} ({record['page']}), "
        f"{record['licence']} ({record['licence_url']}), file {record['member']}"
    )


def fetch_pack(
    root: Path,
    *,
    names: list[str] | None = None,
    offline: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Build the sound pack into ``sfx/``: download the CC0 sounds, synthesise the others.

    ``offline`` synthesises every sound. A sound already in ``sfx/`` is kept unless ``force``;
    ``names`` limits the work to some sounds. A download that fails is synthesised instead and
    reported in ``warnings``. Writes ``sfx/pack.json`` (what each sound is and where it comes
    from) and ``sfx/LICENCES.txt``.
    """

    root = Path(root)
    folder = sfx_folder(root)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    known = {entry["name"] for entry in manifest}
    if names is not None:
        unknown = sorted(set(names) - known)
        if unknown:
            raise SfxError(f"unknown sound(s): {', '.join(unknown)}; the pack has {sorted(known)}")
    wanted = [entry for entry in manifest if names is None or entry["name"] in names]

    pack_file = folder / "pack.json"
    records: dict[str, dict[str, Any]] = {}
    if pack_file.is_file():
        try:
            records = {
                item["name"]: item
                for item in json.loads(pack_file.read_text(encoding="utf-8"))["sounds"]
            }
        except (OSError, ValueError, KeyError, TypeError):
            records = {}

    warnings: list[str] = []
    built: list[str] = []
    kept: list[str] = []
    archives: dict[str, Path | None] = {}
    for entry in wanted:
        name = entry["name"]
        target = folder / f"{name}.wav"
        if target.is_file() and not force and name in records:
            kept.append(name)
            continue
        samples: np.ndarray | None = None
        record: dict[str, Any] = {"name": name, "file": f"{SFX_DIR}/{name}.wav"}
        download = entry.get("download")
        if download is not None and not offline:
            url = download["url"]
            if url not in archives:
                try:
                    archives[url] = download_file(
                        url, download["sha1"], root / "cache" / "sfx-source" / download["file"]
                    )
                except SfxError as exc:
                    archives[url] = None
                    warnings.append(f"{exc}; the sounds of this archive are synthesised instead")
            archive = archives[url]
            if archive is not None:
                try:
                    samples, measured = finish(_decode_member(archive, download["member"]))
                    record.update(
                        source="download",
                        author=download["author"],
                        pack=download["pack"],
                        page=download["page"],
                        member=download["member"],
                        licence=download["licence"],
                        licence_url=download["licence_url"],
                        sha1_archive=download["sha1"],
                    )
                except Exception as exc:
                    samples = None
                    warnings.append(f"{name}: could not use the download ({exc}); synthesised")
        if samples is None:
            samples, measured = generate_sound(name)
            record.update(source="generated", licence="none needed (synthesised by AdAnvil)")
            if download is not None:
                record["note"] = "offline" if offline else "download failed"
        write_wav(target, samples)
        record.update(description=entry["description"], **measured)
        records[name] = record
        built.append(name)

    ordered = [records[entry["name"]] for entry in manifest if entry["name"] in records]
    replace_text(pack_file, json.dumps({"sounds": ordered}, ensure_ascii=False, indent=2) + "\n")
    replace_text(
        folder / "LICENCES.txt",
        "Sound pack of AdAnvil. Only CC0 sources, or sounds synthesised in code.\n\n"
        + "\n".join(_licence_text(record) for record in ordered)
        + "\n",
    )
    return {
        "folder": SFX_DIR,
        "built": built,
        "kept": kept,
        "generated": sorted(r["name"] for r in ordered if r["source"] == "generated"),
        "downloaded": sorted(r["name"] for r in ordered if r["source"] == "download"),
        "warnings": warnings,
    }


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / handle.getframerate()
    except (wave.Error, EOFError, OSError):
        return None


def list_sfx(root: Path, *, fetch: bool = False, offline: bool = False) -> dict[str, Any]:
    """The sounds of ``sfx/``: the pack and the user's own files, with their duration and origin.

    ``fetch`` first builds the missing sounds of the pack (see ``fetch_pack``).
    """

    root = Path(root)
    fetched: dict[str, Any] | None = None
    if fetch:
        fetched = fetch_pack(root, offline=offline)
    folder = sfx_folder(root)
    records: dict[str, dict[str, Any]] = {}
    pack_file = folder / "pack.json"
    if pack_file.is_file():
        try:
            records = {
                item["name"]: item
                for item in json.loads(pack_file.read_text(encoding="utf-8"))["sounds"]
            }
        except (OSError, ValueError, KeyError, TypeError):
            records = {}
    descriptions = {entry["name"]: entry["description"] for entry in load_manifest()}
    sounds: list[dict[str, Any]] = []
    if folder.is_dir():
        for item in sorted(folder.iterdir()):
            if not item.is_file() or item.suffix.lower() not in SFX_SUFFIXES:
                continue
            duration = _wav_duration(item) if item.suffix.lower() == ".wav" else None
            if duration is None:
                try:
                    duration = probe(item).audio_duration
                except Exception:
                    duration = None
            record = records.get(item.stem) if item.suffix.lower() == ".wav" else None
            entry: dict[str, Any] = {
                "name": item.stem,
                "path": f"{SFX_DIR}/{item.name}",
                "duration": None if duration is None else round(duration, 3),
                "origin": "pack" if record else "user",
            }
            if record:
                entry["source"] = record["source"]
                entry["licence"] = record["licence"]
                entry["description"] = record.get("description") or descriptions.get(item.stem)
            sounds.append(entry)
    present = {item["name"] for item in sounds}
    result: dict[str, Any] = {
        "folder": SFX_DIR,
        "count": len(sounds),
        "sounds": sounds,
        "missing_from_pack": [name for name in descriptions if name not in present],
    }
    if result["missing_from_pack"]:
        result["hint"] = (
            "run `adanvil sfx-fetch <project_dir>` (or sfx_list with fetch=true) to build the "
            "pack; your own files can be dropped in sfx/ too"
        )
    if fetched is not None:
        result["fetch"] = fetched
    return result


def resolve_sfx(root: Path, value: str) -> Path:
    """The file an ``sfx`` source names: a project path, or a bare name of ``sfx/``.

    ``whoosh_short``, ``whoosh_short.wav`` and ``sfx/whoosh_short.wav`` all name the same file;
    a bare name finds the first of ``.wav``, ``.mp3``, ``.ogg``... that exists.
    """

    root = Path(root)
    text = value.replace("\\", "/")
    if "/" in text:
        return resolve_in_project(root, text)
    folder = sfx_folder(root)
    if Path(text).suffix.lower() in SFX_SUFFIXES:
        return resolve_in_project(root, f"{SFX_DIR}/{text}")
    for suffix in SFX_SUFFIXES:
        candidate = folder / f"{text}{suffix}"
        if candidate.is_file():
            return resolve_in_project(root, f"{SFX_DIR}/{text}{suffix}")
    return resolve_in_project(root, f"{SFX_DIR}/{text}.wav")


def sfx_source_path(root: Path, value: str) -> str:
    """The project-relative path of ``resolve_sfx`` (what an expanded overlay carries)."""

    return Path(os.path.relpath(resolve_sfx(root, value), Path(root))).as_posix()


__all__ = [
    "PEAK_CEILING_DB",
    "SAMPLE_RATE",
    "SFX_DIR",
    "SFX_SUFFIXES",
    "TARGET_LUFS",
    "SfxError",
    "download_file",
    "fetch_pack",
    "finish",
    "generate_sound",
    "list_sfx",
    "load_manifest",
    "momentary_max_lufs",
    "resolve_sfx",
    "sfx_source_path",
    "write_wav",
]
