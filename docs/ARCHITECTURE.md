# AdAnvil architecture

AdAnvil is a deterministic video-ad renderer driven by an AI agent. The agent judges (what footage to use, what to write, whether the result looks right); AdAnvil executes (cuts, colour, text, audio, checks) and hands back files and contact sheets for the agent to look at.

## Layers

```
server.py / cli.py        MCP tools and CLI: thin wrappers, no logic
project.py                project folder: adanvil.json, brand.yaml, catalog.json, cache/, renders/
library/ reference/ brand/  inputs: footage catalog, reference-ad decoding, brand onboarding
speech/                   speech of the footage: transcripts, phrases, fillers, takes, captions
storyboard/               Storyboard schema (pydantic), validation, recipes/*.yaml
render/                   compose(storyboard) -> mp4, background jobs
qc/                       automatic checks + control images, creative score (score.py), pair comparison (pair.py)
safezones.py              platform interface boxes (TikTok, Reels, Stories, Feed) shared by the score and the validator
variants.py / formats.py / batch.py   variant matrix and slot replacement, 9:16 / 4:5 / 1:1 layout by bands, batch job with one shared cache and the manifest
media/  text/  textremoval/  audio.py   building blocks, pure and unit-tested
ffmpeg.py  _bin.py        subprocess wrapper, probe
```

Rules:
- Every building block is a plain Python function with typed arguments, usable without MCP.
- Filter-string builders are pure functions (string in, string out) so they can be unit-tested without running ffmpeg.
- All ffmpeg calls go through `adanvil.ffmpeg.run()`: list arguments, no shell, `-nostdin -y -loglevel error`, errors raised as `FFmpegError` carrying the command and the last 20 stderr lines.
- Output video contract, everywhere: H.264, `yuv420p` 8-bit, tagged `bt709` (colorspace, primaries, trc), 30 fps, SAR 1, 1080x1920 unless the storyboard says otherwise, AAC 48 kHz stereo, `+faststart`.
- Paths inside a storyboard are relative to the project folder. Tools refuse paths that escape it.

## Media pipeline (how a storyboard becomes a video)

1. **Normalize each segment once** into a mezzanine file in `cache/segments/<hash>.mp4` (1080x1920, 30 fps, yuv420p bt709, x264 `-preset veryfast -crf 16`, no audio). The hash covers the source path, size, mtime and every segment parameter, so re-renders after a text change reuse all segments. Order of operations inside one segment:
   1. trim (`-ss` before `-i`, `-t` duration),
   2. text removal if requested (on source pixels, before any crop),
   3. HDR to SDR tone-mapping when the source transfer is `arib-std-b67` (HLG) or `smpte2084` (PQ):
      `zscale=tin=<trc>:pin=bt2020:min=bt2020nc:t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv`,
   4. optional source crop (display pixel coordinates, after autorotation),
   5. cover-scale and crop to 1080x1920 around a focus point `(fx, fy)` in 0..1,
   6. speed change (`setpts=PTS/speed`),
   7. automatic gamma when mean luma YAVG (measured with `signalstats` on a downscaled 8-bit pass) is below 135: `gamma = min(1.6, log(Y/255)/log(0.55))`, applied with `eq=gamma=…:saturation=1.06`,
   8. motion: zoom or freeze-and-zoom (see below),
   9. `fps=30,setsar=1,format=yuv420p` and bt709 tags.
2. **Concatenate** mezzanines with the concat demuxer and `-c copy` (they share identical encoding parameters by construction). Check that the concatenated duration equals the sum of segment durations within one frame.
3. **Final pass**: overlays and audio on top of the concatenated video, one ffmpeg call. Each overlay is a full-frame 1080x1920 transparent PNG, fed with `-loop 1 -t <total> -i caption.png` and composited with `overlay=0:0:shortest=1:enable='between(t,a,b)'`. Without `shortest=1` a looped input never ends; without `-loop 1` the caption disappears after the first frame. Encoding: `-preset medium -crf 19` (final) or scaled to 540x960 with `-preset veryfast -crf 26` (preview). The filter graph is written to a file (`-/filter_complex` from ffmpeg 7, `-filter_complex_script` before) and the inputs are given relative to the directory they share, so a few hundred pictures stay under the 32767 characters of a Windows command line.

Animated captions (`render/animated.py`) are a run of overlays: one picture per state (the lit word of a `karaoke` or `word` chunk, a frame of the scale-in of a `pop`), each enabled on `gte(t,start)*lt(t,end)` so a state ends on the very number the next one starts on. The pictures are cropped to the ink of the whole chunk (same size and offset in every state, computed once by `text.render_states`) and placed with `overlay=x:y`, which keeps the overlay filter on a small area; they are cached in `cache/overlays/`. `text/animate.py` holds the timing (state plan, pop scales), `text/chunking.py` the punctuation-aware chunks.
Visual devices. A `split` segment (`media/split.py`) is a mezzanine like any other: each panel is rendered first by `render_segment` at the size of its half (so cover crop, speed, freeze, tone-mapping and auto gamma are the clip code, and the panels are cached on their own), then one ffmpeg call overlays the two panels on a `color` source in the gap colour (converted to BT.709) with the label picture on top, pinned to `duration * fps` frames. A `reveal` segment (`media/reveal.py`) renders each side first by `render_segment` at the size of the frame (a still image side gets the same `focus` crop as a clip), then one ffmpeg call pads and optionally zooms both sides with the same expression, lays each label on its own side, and builds the sweep from a mask: a white picture slid over a black frame by an `overlay` whose `x` or `y` is `round(boundary(t))` (in 4:4:4, because overlay snaps positions to the chroma grid), converted to gray, `alphamerge`d onto the after side and overlaid on the before side; the divider is one PNG overlaid at the same expression. `reveal_progress` and `reveal_position` are the same formula in Python (smoothstep, or three legs for `back_and_forth`), so the tests read the divider position from the pixels and compare. A picture card (`render/cards.py`, an `image` overlay with `animation`, `frame` or `rotate`) is drawn with Pillow (rounded corners and rotation supersampled) and every animation state (scale, opacity, offset, one per frame of the 250 ms entry or exit) is a cached picture on an exact span, like a sticker; a plain image overlay still takes the old full-canvas path. A `punch` is a second time-driven scale/crop stage after the clip's own motion, with `gte(t,at)*lt(t,at+hold)` gates so it has no ramp. A video inset (`render/inset.py`) is an extra `-ss in -t span -i` input of the final pass: `fps`, cover scale, `tpad` clone and `trim` to its span, `alphamerge` with a grey mask picture for the rounded and circle shapes, `setpts=PTS+start/TB`, then `overlay` on the same exact `enable` gate as an animated state; its border and shadow are one picture drawn under it, and with `audio: true` its sound replaces the bed under it in the clip-sound bed. Stickers (`render/stickers.py`) are drawn with Pillow at 3x and cached as one picture per state, like animated captions. The progress bar is a solid picture whose `overlay` x follows `W*progress-w` with `eval=frame`, so it costs two small inputs whatever the length of the ad.
4. **Contact sheet** of the result (16 frames, 4x4 or 8x2 grid) next to the video. Never rely on `tile` receiving fewer frames than cells: extract frames to PNG first, then tile them.

### Motion recipes (known traps)

- **Zoom on video**: never `zoompan` on video (frame explosion observed). Use a time-driven scale and crop:
  `scale=w='trunc(1080*Z/2)*2':h='trunc(1920*Z/2)*2':eval=frame,crop=1080:1920:'floor((1080*Z-1080)*fx)':'floor((1920*Z-1920)*fy)'` with `Z = z0+(z1-z0)*min(t/d,1)`. Crop x/y are computed from `t`, never from `iw`/`ih` (crop keeps its initial size when the input resolution changes).
- **Freeze and zoom on a point** (before/after inserts): `tpad=stop_mode=clone:stop_duration=<hold>` (clones the last frame), then the same scale/crop with Z ramping after the freeze starts and the crop centred on `(cx, cy)` clamped to the frame: `x = max(0, min(cx*Z-540, 1080*Z-1082))`, same for y with 960 / 1922.
- **Still image in the middle of a video**: the eval=frame zoom makes a still jitter by one pixel. Use a static image over a blurred copy of itself, or the end-card method below.
- **End card / CTA image with a slow zoom**: upscale the image to 4x the output (4320x7680) then `zoompan` with a single image input (no `-loop 1`), `d=<frames>`, `s=1080x1920`, `fps=30`. This is the only place `zoompan` is allowed.
- Product-only shots: slow zoom 1.00 to 1.04 over the shot, not more.

### Text removal (burnt-in text on reference footage)

Three methods, chosen per segment:
- `delogo` on a measured box (bright pixels > 235 on a sample frame), applied before crop. Any new caption placed there must be at least as large as the erased box.
- `mask`: pixel mask of pixels that stay near-white (min RGB channel > 200) on **every** sampled frame of the shot inside a band, plus explicit extra boxes (emoji), dilated 7 px, then `format=yuv420p,removelogo=mask.png`, then a gaussian blur (sigma 7) applied only inside a feathered copy of the mask (`alphamerge` + `overlay=shortest=1`). Only valid for text that does not move during the shot.
- `crop_out`: keep the larger band of the frame above or below the text, full width, and scale back up. Cleanest when the text sits near the top or bottom edge.

### Text rendering

Captions are rendered by Pillow into full-frame transparent PNGs. Fonts are bundled OFL fonts (see `THIRD_PARTY.md`), emojis come from a provider downloaded and cached on first use in the user cache directory (`twemoji` by default; `apple` only as an explicit user opt-in, never redistributed). Text never renders wider than 960 px: the size shrinks by 2 px steps down to a floor, then the renderer reports the overflow. Styles (defaults, all overridable by brand.yaml):

| style | look | typical use |
|---|---|---|
| `pill` | one white rounded pill per line, black text, bold sans | captions over calm shots |
| `outline` | white heavy text, black stroke, soft drop shadow, no box | native TikTok-style hook over moving shots |
| `outline_soft` | medium-weight white text, 80 % black stroke, strong blurred shadow | UGC diaries |
| `box` | dark rounded cartouche, white text | CTA text |
| `meme` | one long text, balanced short lines in a centred pyramid | single-text meme ads |
| `label` | white rounded card, dark text, soft shadow | a small tag, never the main text |
| `comment` | social comment bubble (`facebook` or `tiktok` look), blurred avatar, no invented author name | reply-to-comment hooks |

## Storyboard v1 (contract between the agent and the renderer)

```json
{
  "version": 1,
  "name": "demo_b",
  "recipe": "B",
  "format": {"width": 1080, "height": 1920, "fps": 30},
  "segments": [
    {"type": "clip", "source": "footage/before.mp4", "in": 1.0, "out": 3.2,
     "crop": null, "focus": [0.5, 0.4], "speed": 1.0, "gamma": "auto",
     "zoom": {"from": 1.0, "to": 1.04},
     "freeze": null,
     "remove_text": {"method": "mask", "box": [60, 1020, 1020, 1175], "extra_boxes": []},
     "label": "hook"},
    {"type": "image", "source": "brand/cta.png", "duration": 3.0, "motion": "zoom_in", "background": "blur"}
  ],
  "overlays": [
    {"type": "caption", "style": "pill", "lines": ["It's your sign", "to try it"], "start": 0.0, "end": 3.0, "y": 330, "size": 60},
    {"type": "comment", "look": "facebook", "text": "Does it really work?", "reactions": 12, "start": 0.0, "end": null, "y": 1300},
    {"type": "image", "source": "brand/badge.png", "start": 5.0, "end": 8.0, "x": 40, "y": 1500}
  ],
  "audio": {"music": "music/track.m4a", "from_reference": null, "volume": 0.8, "fade_in": 0.2, "fade_out": 1.0},
  "output": {"path": "renders/demo_b.mp4", "quality": "final"}
}
```

- Overlay times are seconds on the final timeline; `end: null` means the end of the video.
- `freeze`: `{"hold": 1.0, "zoom": 1.35, "center": [542, 1312]}` freezes the **last frame** of the clip for `hold` seconds while zooming towards `center` (output pixels). To freeze in the middle of a shot, end the segment there.
- `crop`: `{"x", "y", "w", "h"}` in source display pixels.
- `audio.clips`: the bed is the original sound of the clip segments (`audio.clips_bed`): one piece per segment, exactly as long as the segment (speed stretched with `atempo`, silence for images, clips without sound and a frozen hold), joined with a 10 ms `acrossfade` at every cut. Each piece reads 10 ms past its end so the crossfade overlaps the true continuation and the total stays the sum of the segments. The bed is normalised on its own (`loudnorm`, `voice_loudness`) and mixed with the music by `voice_music_mix`, like a voice-over. `audio_from` swaps the source of a piece (the L-cut). `speech/captions.py` maps the words of the played ranges to timeline time for `speech_captions` and for the `cut_mid_word` and `banned_word_spoken` checks.
- Speech sidecars: `library_index` writes `cache/speech/<file key>.json` (words with start, end, text, probability, plus the RMS level every 0.1 s) next to the catalogue, which keeps only `speech: {status, language, text, words_file, has_speech}`. Transcription batches all the clips of one job in a single model load (in-process faster-whisper, or one voice worker run) and the speech of each file is written to the catalogue as soon as that file is done: the worker saves its results after every file (atomically) and `WorkerEngine.align(on_result=...)` hands each one over while the next file runs. The indexing model is `ADANVIL_SPEECH_MODEL` (default `base`); the initial prompt is the brand vocabulary (`speech/transcribe.py`: brand name, product name, `speech.vocabulary`). Audio files (`.m4a`, `.wav`, `.mp3`, `.aac`) are `kind: audio` entries with the same speech summary; a video with silent sound gets `speech.status = silent_audio` and `separate_audio_candidates` (`library/separate_audio.py`, refreshed after every index run). `speech_refine` is a `refine` job (`library/index.py: refine_speech`): it transcribes chosen clips again with a stronger model and replaces their sidecar and summary in place; a failure keeps the old words. `speech_captions.corrections` (`speech/corrections.py`) fixes a misheard word in the captions only, within a character similarity of 0.5.
- Music shorter than the video loops with a 1.5 s `acrossfade`; longer music is trimmed; the audio always ends exactly with the video, with a fade-out. `audio.loudness` (default -14 LUFS, `null` to keep the source level) is applied in a separate pass on the finished bed: never put `loudnorm` in a filter graph that also trims with `-t` (it eats the last 1-2 s).
- Sound design: `sfx` overlays (`source`, `at`, `volume`) and `audio.sfx_auto` (`cuts`, `captions`, `every`, `min_gap`, `volume`, expanded by `storyboard.sfx.expand_sfx_auto`) are mixed by `audio.mix_sfx` over the finished bed as the last audio step: each sound is delayed to its time, a latency-free look-ahead limiter (`audio.peak_limit`) keeps the sample peak under -1.5 dBFS, the mix is encoded as the final video will be and limited again with a lower ceiling when the AAC overshoot takes its true peak above -1 dB, and the WAV has the exact length of the video. The sound pack (`sfx.py`, manifest `assets/sfx_pack.json`) is built into the project's `sfx/` on demand: pinned CC0 downloads verified by SHA-1, or deterministic synthesis, all trimmed and set to one level. `audio.beat_grid` (onset flux weighted towards the low end, tempo refined on a comb, phase aligned to the strongest onsets) feeds `music_beats`, the `beat_alignment` info of the validator and `storyboard_snap` (`storyboard/snap.py`), which moves cuts onto beats by changing the `out` or `duration` of the segments on each side.

## Vision (optional environment)

`vision/` holds everything that needs a model, kept out of the base package: `setup.py` builds an isolated venv (torch CPU, transformers, OpenCV) and downloads the pinned weights; `worker.py` is a standalone JSON-lines worker (`embed_images`, `embed_texts`, `detect_faces`) started by `client.py` as one shared persistent process; `engine.py` defines the engine protocol and a deterministic `FakeVisionEngine` for tests; `store.py` writes `cache/vision/<file key>.npz` (frame times, sizes, embeddings, face boxes); `labels.py` is the fixed zero-shot label list; `versions.py` matches frame embeddings to the reference photos of `brand.product_versions`; `framing.py` turns stored faces into a smoothed `FaceTrack` and the ffmpeg crop expression (`face_cover`), and `align_zooms` gives two faces the same size. `library/vision_index.py` is the indexing pass (tags and versions are derived from stored embeddings, so they refresh without decoding video) and `library/vision_search.py` ranks shots for a sentence. `media/facefocus.py` adds `focus: "face"` to segments: before rendering, `with_framing` attaches the framing found for the face (stored data, else a detection made on the spot) and the cache keys include it. `ADANVIL_VISION=off` disables the worker; tests inject engines with `vision.use_engine`.
