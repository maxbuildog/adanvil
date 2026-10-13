"""Music: find a track with a licence that lets an ad use it, fetch it with its licence, and carry
the licence into the render.

Search sources are Openverse (no key) and, when ``FREESOUND_API_KEY`` is set, Freesound. Only the
licences that allow commercial use AND modification are kept: CC0, PDM, CC BY, CC BY-SA. NC and ND
licences are dropped, whatever the source says. A fetched track lands in ``music/`` with a sidecar
``<file>.licence.json`` and a line in ``music/LICENCES.txt``; the render reads that record.
Standard library only for the network (``urllib``).
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from adanvil import __version__
from adanvil.audio import AUDIO_SUFFIXES
from adanvil.project import resolve_in_project

MUSIC_DIR = "music"
LICENCE_SUFFIX = ".licence.json"
LICENCES_FILE = "LICENCES.txt"
USER_AGENT = f"AdAnvil-music/{__version__} (https://github.com/maxbuildog/adanvil)"
OPENVERSE_API = "https://api.openverse.org/v1/audio/"
FREESOUND_API = "https://freesound.org/apiv2/"
FREESOUND_KEY_ENV = "FREESOUND_API_KEY"
SOURCES = ("openverse", "freesound", "all")
MAX_DOWNLOAD_BYTES = 120 * 1024 * 1024
_TIMEOUT = 30.0

# Licence codes (Openverse spelling) that allow commercial use and modification.
ALLOWED_LICENCES = {
    "cc0": ("CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/", False),
    "pdm": ("Public Domain Mark 1.0", "https://creativecommons.org/publicdomain/mark/1.0/", False),
    "by": ("CC BY", "https://creativecommons.org/licenses/by/{version}/", True),
    "by-sa": ("CC BY-SA", "https://creativecommons.org/licenses/by-sa/{version}/", True),
}
FEELS = ("calm", "medium", "energetic")
# Words of a track's own title, tags and genres that hint at its feel (``guess_feel``). Only a
# hint: ``music_fetch`` measures the real feel (``music_profile``). "happy" and "birthday" are
# deliberately absent: they name an occasion or a mood, not a tempo.
_FEEL_WORDS = {
    "calm": {
        "calm",
        "ambient",
        "soft",
        "relaxing",
        "relax",
        "peaceful",
        "piano",
        "gentle",
        "slow",
        "chill",
        "chillout",
        "lofi",
        "lo-fi",
        "sleep",
        "meditation",
        "sad",
        "ballad",
        "dreamy",
        "warm",
        "emotional",
        "melancholic",
        "atmospheric",
    },
    "medium": {
        "pop",
        "indie",
        "groove",
        "groovy",
        "jazz",
        "folk",
        "soul",
        "corporate",
        "vlog",
        "midtempo",
        "mid-tempo",
        "downtempo",
    },
    "energetic": {
        "upbeat",
        "energetic",
        "energy",
        "fast",
        "dance",
        "edm",
        "rock",
        "party",
        "driving",
        "powerful",
        "workout",
        "epic",
        "hip-hop",
        "hiphop",
        "trap",
        "punchy",
        "funky",
        "motivational",
        "uplifting",
        "metal",
        "punk",
        "hardcore",
        "techno",
        "house",
        "drum",
        "bass",
        "speed",
        "aggressive",
    },
}
FEEL_GUESS_UNKNOWN = "unknown"


class MusicError(RuntimeError):
    """A music search or fetch failed in a way the caller should read."""


# --------------------------------------------------------------------------------- network


def _request(url: str, headers: dict[str, str] | None = None):
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})


def _http_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    """GET ``url`` as JSON. Retries twice on 429 and 5xx (waiting the ``Retry-After``, at most
    10 s); a rate limit that stays is reported as such."""

    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(_request(url, headers), timeout=_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                try:
                    wait = float(exc.headers.get("Retry-After", "") or 2.0)
                except ValueError:
                    wait = 2.0
                time.sleep(min(max(wait, 1.0), 10.0))
                continue
            if exc.code == 429:
                raise MusicError(
                    "the service refused the request (rate limit): wait a few minutes"
                ) from exc
            raise MusicError(
                f"the service answered HTTP {exc.code} for {url.split('?')[0]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last = exc
            if attempt < 2:
                time.sleep(1.5)
                continue
    raise MusicError(f"could not reach {url.split('?')[0]}: {last}")


def _http_download(url: str, destination: Path) -> int:
    """Stream ``url`` to ``destination`` (via a ``.part`` file); returns the size in bytes."""

    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    size = 0
    try:
        with (
            urllib.request.urlopen(_request(url), timeout=60.0) as response,
            partial.open("wb") as out,
        ):
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise MusicError("the file is larger than 120 MB: refused")
                out.write(chunk)
        if size == 0:
            raise MusicError("the download is empty")
        os.replace(partial, destination)
    except MusicError:
        partial.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise MusicError(f"could not download {url.split('?')[0]}: {exc}") from exc
    return size


# --------------------------------------------------------------------------------- licences


def _normalise_licence(code: str | None, version: str | None) -> dict[str, Any] | None:
    """``{"code", "name", "url", "attribution_required"}`` for an allowed licence, else None."""

    key = (code or "").strip().lower()
    if key not in ALLOWED_LICENCES:
        return None
    name, url, attribution = ALLOWED_LICENCES[key]
    version = (version or "").strip() or "4.0"
    if key in ("by", "by-sa"):
        name = f"{name} {version}"
    return {
        "code": key,
        "name": name,
        "url": url.format(version=version),
        "attribution_required": attribution,
    }


def _freesound_licence(value: str | None) -> dict[str, Any] | None:
    """Freesound spells its licences as URLs (or names): keep CC0 and Attribution only. Its
    "Attribution NonCommercial" and "Sampling+" are dropped."""

    text = (value or "").strip().lower()
    match = re.search(
        r"/(zero|by|by-sa|by-nc|by-nd|by-nc-sa|sampling\+?)/(?:1\.0|2\.\d|3\.0|4\.0)", text
    )
    if "publicdomain/zero" in text or text in ("creative commons 0", "cc0"):
        return _normalise_licence("cc0", "1.0")
    if match and match.group(1) == "by":
        version = re.search(r"by/(\d\.\d)", text)
        return _normalise_licence("by", version.group(1) if version else "4.0")
    if text == "attribution":
        return _normalise_licence("by", "4.0")
    return None


def _attribution_text(title: str, author: str, licence: dict[str, Any], page: str) -> str:
    if licence["code"] == "cc0":
        return f'"{title}" by {author} is marked with CC0 1.0 (no attribution required).'
    if licence["code"] == "pdm":
        return f'"{title}" by {author} is in the public domain (no attribution required).'
    text = f'"{title}" by {author}'
    if page:
        text += f" ({page})"
    return f"{text} is licensed under {licence['name']} ({licence['url']})."


# ---------------------------------------------------------------------------------- sources


def _tags(item: dict[str, Any]) -> list[str]:
    names = []
    for tag in item.get("tags") or []:
        name = tag.get("name") if isinstance(tag, dict) else tag
        if name:
            names.append(str(name).lower())
    for genre in item.get("genres") or []:
        names.append(str(genre).lower())
    return names


def _openverse_track(item: dict[str, Any]) -> dict[str, Any] | None:
    licence = _normalise_licence(item.get("license"), item.get("license_version"))
    if licence is None or not item.get("url"):
        return None
    title = html.unescape(str(item.get("title") or "untitled"))
    author = html.unescape(str(item.get("creator") or "unknown"))
    page = str(item.get("foreign_landing_url") or "")
    duration = item.get("duration")
    return {
        "source": "openverse",
        "id": str(item["id"]),
        "title": title,
        "author": author,
        "licence": licence["name"],
        "licence_code": licence["code"],
        "licence_url": licence["url"],
        "duration": None if duration is None else round(float(duration) / 1000.0, 1),
        "source_url": page,
        "preview_url": str(item["url"]),
        "attribution_required": licence["attribution_required"],
        "attribution_text": html.unescape(str(item.get("attribution") or ""))
        or _attribution_text(title, author, licence, page),
        "tags": _tags(item)[:10],
        "origin": str(item.get("source") or item.get("provider") or ""),
    }


OPENVERSE_PAGE = 20  # the most an anonymous request may ask for (more answers HTTP 401)


def search_openverse(query: str, *, limit: int) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for page in range(1, 4):
        params = {
            "q": query,
            "category": "music",
            "license": "cc0,pdm,by,by-sa",
            "page_size": str(OPENVERSE_PAGE),
            "page": str(page),
            "mature": "false",
        }
        data = _http_json(OPENVERSE_API + "?" + urllib.parse.urlencode(params))
        for item in data.get("results") or []:
            track = _openverse_track(item)
            if track is not None:
                tracks.append(track)
        if len(tracks) >= limit or page >= int(data.get("page_count") or 1):
            break
    return tracks


def _freesound_key() -> str | None:
    return os.environ.get(FREESOUND_KEY_ENV, "").strip() or None


def _freesound_track(item: dict[str, Any]) -> dict[str, Any] | None:
    licence = _freesound_licence(item.get("license"))
    previews = item.get("previews") or {}
    preview = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
    if licence is None or not preview:
        return None
    title = str(item.get("name") or "untitled")
    author = str(item.get("username") or "unknown")
    page = str(item.get("url") or f"https://freesound.org/s/{item['id']}/")
    duration = item.get("duration")
    return {
        "source": "freesound",
        "id": str(item["id"]),
        "title": title,
        "author": author,
        "licence": licence["name"],
        "licence_code": licence["code"],
        "licence_url": licence["url"],
        "duration": None if duration is None else round(float(duration), 1),
        "source_url": page,
        "preview_url": str(preview),
        "attribution_required": licence["attribution_required"],
        "attribution_text": _attribution_text(title, author, licence, page),
        "tags": [str(tag).lower() for tag in (item.get("tags") or [])][:10],
        "origin": "freesound",
    }


_FREESOUND_FIELDS = "id,name,username,license,duration,previews,url,tags"


def search_freesound(query: str, key: str, *, limit: int) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "filter": 'license:("Creative Commons 0" OR "Attribution") duration:[20 TO *]',
        "fields": _FREESOUND_FIELDS,
        "page_size": str(max(1, min(limit, 50))),
        "token": key,
    }
    data = _http_json(FREESOUND_API + "search/text/?" + urllib.parse.urlencode(params))
    tracks = []
    for item in data.get("results") or []:
        track = _freesound_track(item)
        if track is not None:
            tracks.append(track)
    return tracks


# ----------------------------------------------------------------------------------- ranking


def guess_feel(track: dict[str, Any], mood: str | None = None) -> str:
    """A rough feel (``calm``, ``medium`` or ``energetic``) from the track's OWN words: its title,
    tags and genres (the lists are ``_FEEL_WORDS``). ``unknown`` when no word matches or two
    feels match equally. The feel that was asked for is never read, and ``mood`` is ignored
    (kept in the signature for old callers). The measured feel of ``music_fetch`` is the real
    answer; this guess only orders the candidates."""

    words = set(
        re.findall(r"[a-z][a-z-]*", " ".join([track["title"], *track["tags"]]).lower())
    )
    scores = {feel: len(words & vocabulary) for feel, vocabulary in _FEEL_WORDS.items()}
    best = max(scores.values())
    if best == 0:
        return FEEL_GUESS_UNKNOWN
    winners = [feel for feel, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else FEEL_GUESS_UNKNOWN


def rank_tracks(
    tracks: list[dict[str, Any]], *, duration: float | None, feel: str | None
) -> list[dict[str, Any]]:
    """Sort ``tracks`` for an ad of ``duration`` seconds and a wanted ``feel``.

    Order: tracks at least as long as the ad first (the nearest length first, so a 10-minute
    file does not beat a 40 s one), then the tracks that are too short (the longest first;
    a short bed loops with a crossfade); inside each group, the tracks whose own words match the
    wanted feel first, then ``unknown`` ones. Each track gets ``fit``: ``covers_ad`` (bool or
    None) and ``feel_guess`` (from the track's title, tags and genres only, never from ``feel``).
    """

    ranked = []
    for track in tracks:
        track = dict(track)
        guess = guess_feel(track)
        covers = (
            None
            if duration is None or track["duration"] is None
            else (track["duration"] >= duration)
        )
        track["fit"] = {"covers_ad": covers, "feel_guess": guess}
        ranked.append(track)

    def sort_key(track: dict[str, Any]) -> tuple[Any, ...]:
        covers = track["fit"]["covers_ad"]
        length = track["duration"]
        guess = track["fit"]["feel_guess"]
        feel_rank = (
            0 if feel is None or guess == feel else (1 if guess == FEEL_GUESS_UNKNOWN else 2)
        )
        if covers is None:
            return (0, feel_rank, 0.0)
        if covers:
            return (0, feel_rank, length - (duration or 0.0))
        return (1, feel_rank, -(length or 0.0))

    return sorted(ranked, key=sort_key)


# -------------------------------------------------------------------------------- search API


def storyboard_target(root: Path, storyboard: str) -> dict[str, Any]:
    """``{"duration", "feel"}`` of a storyboard (a project file path, e.g.
    ``storyboards/summer.json``): its total duration and ``audio.music_feel``."""

    from adanvil.storyboard import load_storyboard

    path = resolve_in_project(root, storyboard)
    if not path.is_file():
        raise MusicError(f"storyboard file does not exist: {storyboard}")
    try:
        model = load_storyboard(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise MusicError(f"could not read the storyboard {storyboard}: {exc}") from exc
    return {"duration": round(model.total_duration(), 2), "feel": model.audio.music_feel}


def music_search(
    root: Path | None,
    query: str,
    *,
    mood: str | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    source: str = "openverse",
    limit: int = 12,
    fit_to: str | None = None,
    storyboard: str | None = None,
    feel: str | None = None,
) -> dict[str, Any]:
    """Search tracks whose licence allows commercial use and modification (see the module)."""

    if source not in SOURCES:
        raise MusicError(f"source must be one of {', '.join(SOURCES)} (got {source!r})")
    if not query or not query.strip():
        raise MusicError("query must not be empty")
    if feel is not None and feel not in FEELS:
        raise MusicError(f"feel must be one of {', '.join(FEELS)}")
    limit = max(1, min(int(limit), 50))
    notes: list[str] = []
    target: dict[str, Any] | None = None
    if fit_to is not None:
        if fit_to != "storyboard":
            raise MusicError('fit_to must be "storyboard" (or omitted)')
        if not storyboard or root is None:
            raise MusicError('fit_to="storyboard" needs storyboard: the project path of the file')
        target = storyboard_target(root, storyboard)
        feel = feel or target["feel"]
    wanted_feel = feel
    if wanted_feel is None and mood:
        mood_words = set(re.findall(r"[a-z][a-z-]*", mood.lower()))
        wanted_feel = next(
            (name for name, words in _FEEL_WORDS.items() if mood_words & words), None
        )
    text = f"{query.strip()} {mood.strip()}" if mood and mood.strip() else query.strip()
    pool = limit * 3 if (min_duration or max_duration or target) else limit
    tracks: list[dict[str, Any]] = []
    if source in ("openverse", "all"):
        tracks.extend(search_openverse(text, limit=pool))
    if source in ("freesound", "all"):
        key = _freesound_key()
        if key is None:
            notes.append(
                f"Freesound was not searched: {FREESOUND_KEY_ENV} is not set (a free key from "
                "https://freesound.org/apiv2/apply). Openverse results are listed"
                + (" instead." if source == "all" else " only if you search it.")
            )
        else:
            tracks.extend(search_freesound(text, key, limit=pool))
    seen: set[str] = set()
    unique = []
    for track in tracks:
        marker = track["source_url"] or f"{track['source']}:{track['id']}"
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(track)
    tracks = [
        t
        for t in unique
        if (t["duration"] is None or min_duration is None or t["duration"] >= min_duration)
        and (t["duration"] is None or max_duration is None or t["duration"] <= max_duration)
    ]
    wanted_duration = None if target is None else float(target["duration"])
    if wanted_duration is not None or wanted_feel is not None:
        tracks = rank_tracks(tracks, duration=wanted_duration, feel=wanted_feel)
    result: dict[str, Any] = {
        "query": text,
        "source": source,
        "licences_kept": ["CC0", "PDM", "CC BY", "CC BY-SA"],
        "count": min(len(tracks), limit),
        "tracks": tracks[:limit],
        "notes": notes,
    }
    if target is not None:
        result["fit_to"] = {
            "storyboard": storyboard,
            "duration": target["duration"],
            "feel": wanted_feel,
            "how": "tracks at least as long as the ad first, then the feel guessed from each "
            "track's own title, tags and genres (unknown when they say nothing); the measured "
            "feel that music_fetch returns is the real answer",
        }
    if not tracks:
        notes.append("no track matched: try other words, or drop min_duration / max_duration")
    return result


# ---------------------------------------------------------------------------------- fetching


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40].strip("-") or "track"


def _suffix(url: str, filetype: str | None) -> str:
    path_suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if path_suffix in AUDIO_SUFFIXES:
        return path_suffix
    if filetype and f".{filetype.lower()}" in AUDIO_SUFFIXES:
        return f".{filetype.lower()}"
    return ".mp3"


def _openverse_detail(identifier: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-fA-F-]{8,40}", identifier):
        raise MusicError(f"not an Openverse id: {identifier!r}")
    return _http_json(OPENVERSE_API + identifier + "/")


def _freesound_detail(identifier: str, key: str) -> dict[str, Any]:
    if not re.fullmatch(r"\d{1,12}", identifier):
        raise MusicError(f"not a Freesound id: {identifier!r}")
    params = urllib.parse.urlencode({"fields": _FREESOUND_FIELDS, "token": key})
    return _http_json(f"{FREESOUND_API}sounds/{identifier}/?{params}")


def strip_cover_art(path: Path) -> bool:
    """Remove the picture embedded in an audio file (``-map 0:a -c copy`` into the same
    container), so the stored file is audio only. True when a picture was removed; the file is
    left as it was when there is none or when ffmpeg fails."""

    from adanvil.ffmpeg import run, run_probe

    try:
        streams = json.loads(run_probe(["-show_streams", "-of", "json", str(path)])).get(
            "streams", []
        )
    except Exception:
        return False
    if not any(stream.get("codec_type") == "video" for stream in streams):
        return False
    clean = path.with_name(f"{path.stem}.clean{path.suffix}")
    try:
        run(
            [
                "-i",
                str(path),
                "-map",
                "0:a",
                "-map_metadata",
                "0",
                "-c",
                "copy",
                "-y",
                str(clean),
            ]
        )
        if not clean.is_file() or clean.stat().st_size == 0:
            return False
        os.replace(clean, path)
    except Exception:
        return False
    finally:
        clean.unlink(missing_ok=True)
    return True


def music_fetch(root: Path, source: str, identifier: str) -> dict[str, Any]:
    """Download one track into ``music/`` with its licence record, then profile it."""

    from adanvil.audio import music_profile

    root = Path(root)
    if source not in ("openverse", "freesound"):
        raise MusicError("source must be openverse or freesound")
    identifier = str(identifier).strip()
    if source == "openverse":
        item = _openverse_detail(identifier)
        track = _openverse_track(item)
        filetype = item.get("filetype")
    else:
        key = _freesound_key()
        if key is None:
            raise MusicError(
                f"{FREESOUND_KEY_ENV} is not set: Freesound needs a free API key "
                "(https://freesound.org/apiv2/apply)"
            )
        item = _freesound_detail(identifier, key)
        track = _freesound_track(item)
        filetype = "mp3"
    if track is None:
        raise MusicError(
            "this track's licence does not allow commercial use and modification "
            "(only CC0, PDM, CC BY and CC BY-SA are accepted), or it has no file to download"
        )
    folder = root / MUSIC_DIR
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{_slug(track['title'])}-{track['source'][:2]}{_slug(track['id'])[:8]}" + _suffix(
        track["preview_url"], filetype
    )
    target = folder / name
    size = _http_download(track["preview_url"], target)
    if strip_cover_art(target):
        size = target.stat().st_size
    record = {
        "source": track["source"],
        "id": track["id"],
        "url": track["source_url"],
        "download_url": track["preview_url"],
        "title": track["title"],
        "author": track["author"],
        "licence": track["licence"],
        "licence_code": track["licence_code"],
        "licence_url": track["licence_url"],
        "attribution_required": track["attribution_required"],
        "attribution_text": track["attribution_text"],
        "fetch_date": date.today().isoformat(),
        "file": f"{MUSIC_DIR}/{name}",
        "bytes": size,
    }
    write_licence_record(root, record["file"], record)
    profile = music_profile(root, record["file"])["tracks"][0]
    return {
        "path": record["file"],
        "licence": record["licence"],
        "licence_file": record["file"] + LICENCE_SUFFIX,
        "attribution_required": record["attribution_required"],
        "attribution_text": record["attribution_text"],
        "title": record["title"],
        "author": record["author"],
        "duration": profile.get("duration"),
        "tempo_bpm": profile.get("tempo_bpm"),
        "feel": profile.get("feel"),
        "loudness_lufs": profile.get("loudness_lufs"),
        "profile": profile,
    }


def write_licence_record(root: Path, relative: str, record: dict[str, Any]) -> None:
    """Write ``<file>.licence.json`` and append a line to ``music/LICENCES.txt``."""

    root = Path(root)
    path = root / (relative + LICENCE_SUFFIX)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log = root / MUSIC_DIR / LICENCES_FILE
    needs = (
        ' Attribution required: "' + record["attribution_text"] + '"'
        if record.get("attribution_required")
        else " No attribution required."
    )
    line = (
        f'{Path(relative).name}: "{record["title"]}" by {record["author"]}, '
        f"{record['licence']} ({record['licence_url']}), {record['source']} {record['id']} "
        f"({record['url']}), fetched {record['fetch_date']}.{needs}\n"
    )
    existing = log.read_text(encoding="utf-8") if log.is_file() else ""
    if not existing:
        existing = "Music tracks of this project: source and licence of each.\n"
    elif not existing.endswith("\n"):
        existing += "\n"
    log.write_text(existing + line, encoding="utf-8")


# --------------------------------------------------------------------- the licence of a render


def read_licence_record(root: Path, relative: str) -> dict[str, Any] | None:
    """The record next to a music file, or None (missing or unreadable)."""

    try:
        path = resolve_in_project(Path(root), relative + LICENCE_SUFFIX)
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not data.get("licence"):
        return None
    return data


# A line of ``music/LICENCES.txt`` written by hand or by an older version, e.g.
# ``night_street.mp3 : Night Street, Author, Pixabay 131746, licence Pixabay, telecharge ...``
# or the line ``music_fetch`` writes. The licence is found by name in either shape.
_KNOWN_LICENCE = re.compile(
    r"\b(CC0 1\.0|Public Domain Mark 1\.0|CC BY-SA \d\.\d|CC BY \d\.\d|CC BY-SA|CC BY)\b"
)
_LEGACY_LICENCE = re.compile(r"\blicen[cs]e\s+([^,;()]+)", re.IGNORECASE)
_ATTRIBUTION = re.compile(r'Attribution required:\s*"(.*)"\.?\s*$')


_URL = re.compile(r"\((https?://[^)\s]+)\)")
_FETCHED_LINE = re.compile(
    r'^\s*"(?P<title>.+?)"\s+by\s+(?P<author>.+?),\s+(?P<licence>CC0 1\.0|Public Domain Mark 1\.0'
    r"|CC BY-SA \d\.\d|CC BY \d\.\d|CC BY-SA|CC BY)\b(?P<tail>.*)$"
)


def _ledger_details(rest: str, licence: str) -> dict[str, Any]:
    """The title, author and source (and its URL) written on a ``LICENCES.txt`` line, those that
    are there: the line ``music_fetch`` writes (``"Title" by Author, CC BY 3.0 (licence url),
    source id (url), fetched ...``) or the hand-written ``Title, Author, Source id, licence X``."""

    rest = rest.strip()
    details: dict[str, Any] = {}
    fetched = _FETCHED_LINE.match(rest)
    if fetched:
        details["title"] = fetched.group("title").strip()
        details["author"] = fetched.group("author").strip()
        tail = fetched.group("tail")
        urls = _URL.findall(tail)
        if len(urls) >= 2:
            details["licence_url"] = urls[0]
            details["url"] = urls[1]
            between = tail.split(urls[0], 1)[1].split(urls[1], 1)[0]
            source = between.strip(" ,)(").rsplit("(", 1)[0].strip(" ,)(")
            if source:
                details["source"] = source
        elif urls:
            details["licence_url"] = urls[0]
        return details
    marker = re.search(r",?\s*licen[cs]e\s", rest, re.IGNORECASE)
    head = rest[: marker.start()] if marker else rest
    known = _KNOWN_LICENCE.search(head)
    if known:
        head = head[: known.start()]
    parts = [part.strip() for part in head.strip(" ,").split(", ") if part.strip()]
    if len(parts) >= 3:
        details["title"] = ", ".join(parts[:-2])
        details["author"] = parts[-2]
        details["source"] = parts[-1]
    elif len(parts) == 2:
        details["title"], details["author"] = parts
    elif len(parts) == 1:
        details["title"] = parts[0]
    link = _URL.search(rest)
    if link and not licence.startswith("CC"):
        details["url"] = link.group(1)
    return details


def read_ledger_record(root: Path, relative: str) -> dict[str, Any] | None:
    """The licence of ``relative`` read from its line in ``music/LICENCES.txt`` (matched by file
    name; the last line wins), or None. Used when the file has no ``.licence.json`` sidecar.
    The result carries ``ledger_line`` (the 1-based line number)."""

    name = Path(relative).name
    ledger = Path(root) / MUSIC_DIR / LICENCES_FILE
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    found: dict[str, Any] | None = None
    for number, line in enumerate(lines, start=1):
        head, separator, rest = line.partition(":")
        if not separator or head.strip() != name:
            continue
        attribution = _ATTRIBUTION.search(rest)
        known = _KNOWN_LICENCE.search(rest)
        legacy = _LEGACY_LICENCE.search(rest)
        if known:
            licence = known.group(1)
        elif legacy:
            licence = legacy.group(1).strip()
        else:
            continue
        required = attribution is not None or (
            known is not None and not licence.startswith(("CC0", "Public"))
        )
        found = {
            "licence": licence,
            "attribution_required": required,
            "attribution_text": attribution.group(1) if attribution else "",
            "title": None,
            "ledger_line": number,
            **_ledger_details(rest, licence),
        }
    return found


def music_licence(root: Path, relative: str | None) -> dict[str, Any] | None:
    """What the render and ``ad_score`` say about the licence of ``relative`` (``audio.music``):
    ``{"path", "status", "licence", "attribution_required", "attribution_text", "warnings"}``.
    The record is the sidecar, else the file's line in ``music/LICENCES.txt`` (``read_from`` says
    which). ``status`` is ``recorded`` or ``unknown``. None when the ad has no music file."""

    if not relative:
        return None
    record = read_licence_record(root, relative)
    origin = f"{relative}{LICENCE_SUFFIX}"
    if record is None:
        record = read_ledger_record(root, relative)
        if record is not None:
            origin = f"{MUSIC_DIR}/{LICENCES_FILE} line {record['ledger_line']} (no sidecar)"
    if record is None:
        return {
            "path": relative,
            "status": "unknown",
            "licence": None,
            "attribution_required": None,
            "attribution_text": None,
            "warnings": [
                f"music_licence_unknown: {relative} has no licence record ({relative}"
                f"{LICENCE_SUFFIX}, and no line for it in {MUSIC_DIR}/{LICENCES_FILE}); fetch it "
                "with music_fetch, or check by hand that the "
                "licence allows commercial use before the ad runs"
            ],
        }
    required = bool(record.get("attribution_required"))
    text = str(record.get("attribution_text") or "")
    warnings = []
    if required:
        warnings.append(f"music_attribution_required: put this in the ad's description: {text}")
    return {
        "path": relative,
        "status": "recorded",
        "licence": str(record["licence"]),
        "licence_url": record.get("licence_url"),
        "title": record.get("title"),
        "author": record.get("author"),
        "source": record.get("source"),
        "source_url": record.get("url"),
        "read_from": origin,
        "attribution_required": required,
        "attribution_text": text if required else None,
        "warnings": warnings,
    }


def licence_label(info: dict[str, Any] | None) -> str:
    """The text of the ``music licence`` column of the batch manifest."""

    if info is None:
        return ""
    if info["status"] == "unknown":
        return "unknown"
    return info["licence"] + ("; attribution required" if info["attribution_required"] else "")


__all__ = [
    "ALLOWED_LICENCES",
    "MusicError",
    "licence_label",
    "music_fetch",
    "music_licence",
    "music_search",
    "rank_tracks",
]
