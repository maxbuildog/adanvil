"""A playlist or stream description named like a media file is refused before ffmpeg opens it.

A 7.0 static ffmpeg fetched the URLs of a DASH manifest named ``clip.mp4`` even under
``-protocol_whitelist file`` (measured on 16/10/2026), so AdAnvil reads the first bytes of
every input itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adanvil.ffmpeg import run, run_probe

PLAYLISTS = [
    "#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:1,\nhttp://127.0.0.1:1/a.ts\n",
    "﻿ffconcat version 1.0\nfile 'http://127.0.0.1:1/a.mp4'\n",
    "v=0\no=- 0 0 IN IP4 127.0.0.1\nm=video 5004 RTP/AVP 96\n",
    "<?xml version='1.0'?><smil><body><video src='http://127.0.0.1:1/a.mp4'/></body></smil>",
    "<?xml version='1.0'?>\n<MPD xmlns='urn:mpeg:dash:schema:mpd:2011'></MPD>\n",
]


@pytest.mark.parametrize("content", PLAYLISTS)
def test_playlists_disguised_as_media_are_refused(tmp_path: Path, content: str) -> None:
    disguised = tmp_path / "clip.mov"
    disguised.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="playlist"):
        run(["-i", str(disguised), "-f", "null", "-"], timeout=60)
    with pytest.raises(ValueError, match="playlist"):
        run_probe(["-show_streams", str(disguised)])


def test_a_relative_input_is_read_from_the_working_folder(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_text(PLAYLISTS[0], encoding="utf-8")
    with pytest.raises(ValueError, match="playlist"):
        run(["-i", "clip.mp4", "-f", "null", "-"], timeout=60, cwd=tmp_path)


def test_own_concat_lists_and_real_media_still_work(tmp_path: Path) -> None:
    clip = tmp_path / "a.mp4"
    run(["-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=1", "-pix_fmt", "yuv420p", str(clip)])
    listing = tmp_path / "list.ffconcat"
    listing.write_text(f"ffconcat version 1.0\nfile '{clip.as_posix()}'\n", encoding="utf-8")
    joined = tmp_path / "joined.mp4"
    run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)])
    assert joined.is_file()
    assert "codec_type=video" in run_probe(["-show_streams", str(joined)])
