# AdAnvil

An MCP server that lets an AI agent edit vertical 9:16 video ads.

<!-- mcp-name: io.github.maxbuildog/adanvil -->

AdAnvil works with Claude Code, Codex, Cursor or any MCP client. It builds ads from a brand's own footage, from a reference ad, or from an idea. The agent judges (which clips, what the text says, whether the result looks right). AdAnvil renders deterministically and checks the result: the same storyboard always gives the same video.

It is not tied to a niche. The recipes and the editing guide speak in general terms (product, benefit, problem, target customer) with examples from several niches: coffee, fashion, fitness, pets, software, home, beauty.

![demo](https://raw.githubusercontent.com/maxbuildog/adanvil/main/docs/demo.gif)

The demo above was made from the CC0 demo pack in `examples/demo/` (four Wikimedia Commons clips, a product still and a fictional brand called Ember & Oak). No footage is stored in this repository: the pack is fetched on demand.

## What it does

- Reference decoding: shots, pacing, burnt-in text and speech, plus a shot-by-shot breakdown the agent writes and compares to your edit.
- Footage library: indexing, contact sheets and mosaics so the agent can look at every clip before using it, and the speech of every video and audio file (word timestamps, searchable as soon as each file is done) so a talking creator can be cut on the pauses.
- 9 recipes (YAML) for common ad shapes: text hook, hook plus product, before and after, customer diary, comment reply, meme text, voice-over ad, long remake, talking-head UGC.
- 7 caption styles (`pill`, `meme`, `outline`, `outline_soft`, `box`, `label`, `comment`) with emoji support.
- Removal of burnt-in text from reference footage.
- HDR (HLG and PQ) to SDR tone-mapping.
- Music matching and loudness control.
- Optional open-source voice-over (Chatterbox) with captions synced to the spoken words.
- Talking-head UGC: best take of each line across several recordings, cuts on the pauses without hesitations, the creator's own sound kept (with B-roll laid over the voice), captions built from the speech.
- Visual devices: a before/after reveal (a divider sweeps one picture away to show the other), split screens, picture cards (a photo or a screenshot that comes in over the video), video insets (picture-in-picture), stickers and arrows, a progress bar and punch-in zooms.
- Sound design: a pack of about 18 sound effects (CC0 or synthesised, never stored in the repository), `sfx` overlays and `audio.sfx_auto` at the cuts and captions, and cuts that snap to the beat of the music.
- End cards composed from a real product photo.
- Quality control with images: automatic checks, contact sheet of the result, full-resolution frames.
- Detached background jobs for indexing, voice and rendering, so long work survives a dropped connection.

## Install

Pick one path. The first is the simplest.

### 1. Claude Desktop extension (one click)

Download `adanvil-<version>.mcpb` from the [latest release](https://github.com/maxbuildog/adanvil/releases/latest), double-click it, and click Install in Claude Desktop. It asks for a projects folder and, optionally, a Freesound key (stored securely). You still need ffmpeg 6.1 or newer on your computer (`brew install ffmpeg`, `winget install Gyan.FFmpeg`, `sudo apt install ffmpeg`); if it is missing, the extension says so with the command for your system.

### 2. Docker (no Python, no ffmpeg to install)

```bash
git clone https://github.com/maxbuildog/adanvil && cd adanvil
docker build -t adanvil .
```

The image has ffmpeg 6.1, runs as a non-root user and speaks MCP over stdio. Mount the folder that holds your projects at `/projects`; the agent must give tool calls paths under `/projects` (for example `/projects/my-ad`): the image sets `ADANVIL_PROJECTS_DIR=/projects`, so any other folder, and `/projects` itself, is refused.

```bash
# Claude Code
claude mcp add adanvil -- docker run -i --rm -v "$PWD:/projects" adanvil
```

On Linux add `--user "$(id -u):$(id -g)"` so the files written to the folder belong to you.

Codex, in `~/.codex/config.toml`:

```toml
[mcp_servers.adanvil]
command = "docker"
args = ["run", "-i", "--rm", "-v", "/home/you/ads:/projects", "adanvil"]
tool_timeout_sec = 900
default_tools_approval_mode = "approve"  # needed with approval_policy = "never" (codex exec)
```

Cursor, in `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "adanvil": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-v", "/home/you/ads:/projects", "adanvil"]
    }
  }
}
```

The image holds the base package only. Transcription is available with `docker build --build-arg EXTRAS=transcribe -t adanvil:transcribe .`. The voice, vision and music-generation environments (`adanvil voice-setup`, `vision_setup`, `music-setup`) are not included: they need gigabytes of models, so use the `uvx` path for them.

### 3. uvx

Requirements: Python 3.11 or newer, [uv](https://docs.astral.sh/uv/) (it provides `uvx`), and ffmpeg 6.1 or newer, built with `libzimg`, on your `PATH` (check with `ffmpeg -version`; if it lives elsewhere, set `ADANVIL_FFMPEG` and `ADANVIL_FFPROBE` to the full paths).

```bash
# 1. Check the install (the first run downloads the package)
uvx --from adanvil adanvil --version

# 2. Register the server in Claude Code
claude mcp add adanvil -e ADANVIL_PROJECTS_DIR="$HOME/AdAnvil" -- uvx --from adanvil adanvil-mcp

# 3. Then ask your agent:
#    "Make a 9:16 ad from the footage in ./my-project"
```

`ADANVIL_PROJECTS_DIR` (recommended) is the folder that holds your projects: the server then refuses any project outside it. Projects are subfolders of it (for example `~/AdAnvil/summer-ad`). See [`SECURITY.md`](SECURITY.md).

The package is named `adanvil`. It installs two commands: `adanvil` (command line) and `adanvil-mcp` (the MCP server over stdio). `uvx adanvil-mcp` alone does not work, because `uvx` looks for a package with that name: always add `--from`.

The agent creates the project folder and tells you where to drop your files: footage in `footage/`, the reference ad in `reference/`, music in `music/`, logo and end card in `brand/`. It learns the method from the `editing_guide` tool and from [`AGENTS.md`](AGENTS.md).

Codex, in `~/.codex/config.toml`:

```toml
[mcp_servers.adanvil]
command = "uvx"
args = ["--from", "adanvil", "adanvil-mcp"]
env = { ADANVIL_PROJECTS_DIR = "/home/you/AdAnvil" }
tool_timeout_sec = 900                   # indexing long footage can take minutes
default_tools_approval_mode = "approve"  # needed with approval_policy = "never" (codex exec)
```

Without `default_tools_approval_mode`, a non-interactive `codex exec` run rejects every AdAnvil tool call with "MCP tool call requires approval".

Cursor, in `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "adanvil": {
      "command": "uvx",
      "args": ["--from", "adanvil", "adanvil-mcp"],
      "env": { "ADANVIL_PROJECTS_DIR": "/home/you/AdAnvil" }
    }
  }
}
```

Any other MCP client (generic stdio entry):

```json
{
  "mcpServers": {
    "adanvil": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "adanvil", "adanvil-mcp"],
      "env": { "ADANVIL_PROJECTS_DIR": "/home/you/AdAnvil" }
    }
  }
}
```

Restart the client after editing its config, then ask the agent to call `recipe_list`. Nine recipes means the server is connected.

Optional: reference transcription needs the `transcribe` extra (`faster-whisper`). Use `--from "adanvil[transcribe]"` in place of the plain `--from` value above.

## Try it with the demo pack

The demo needs a checkout, because the fetch script lives in the repository:

```bash
git clone https://github.com/maxbuildog/adanvil
cd adanvil
uv sync
uv run python examples/demo/fetch_demo.py ./demo-project
uv run adanvil index ./demo-project --source footage
```

Then register the MCP server and ask your agent to make an ad from `./demo-project` (give it the absolute path), or follow the command line walkthrough in [`examples/demo/README.md`](examples/demo/README.md).

## How it works

Everything lives in one project folder: `footage/`, `brand/`, `music/`, `reference/`, `renders/`, `cache/`, plus `adanvil.json`, `brand.yaml` and `catalog.json`. Every tool takes the absolute `project_dir`; every other path is relative to it and cannot escape it.

1. The agent indexes the footage, looks at the contact sheets, and optionally decodes a reference ad.
2. It picks a recipe and writes a storyboard (JSON): shots, captions, audio.
3. `storyboard_validate` checks the storyboard against the recipe rules, then `render_start` renders it in the background.
4. `qc_run` checks the file and returns control images that the agent looks at before delivering; `ad_score` scores its creative quality (aim for 80 or more).

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the pipeline, [`docs/EDITING_GUIDE.md`](docs/EDITING_GUIDE.md) for the editing doctrine, and [`AGENTS.md`](AGENTS.md) for the step-by-step guide an agent follows (same content as `skill/SKILL.md`, the Claude Code skill version).

## Choosing a model

The result depends on the model that drives AdAnvil. Reference analysis and shot choice are judgement work: use the strongest model you can (for example Claude Opus) for them. Smaller models execute the steps well, but they tend to copy the surface of a reference (same words, same shots) instead of rebuilding its logic for your brand.

## Tools (43)

| Tool | What it does |
| --- | --- |
| `project_init` | Create an AdAnvil project skeleton. |
| `project_read_text` | Read one of AdAnvil's own text files in the project (licence notes, JSON, YAML; at most 1 MB). |
| `brand_onboard` | Read a brand's site, then write the brand profile from the answers. |
| `brand_get` | Read the brand profile and brief. |
| `library_index` | Index or refresh footage (long runs become detached jobs). |
| `library_view` | One catalogue entry with its contact sheet. |
| `library_annotate` | Save what the agent saw in a clip (description, people, emotion, quality, exclusion). |
| `library_search` | Search the indexed footage by text (including what the clips say), tags, people, emotion, duration, or by sentence (`query`, with the vision environment). |
| `speech_view` | Compact phrases (with flags), fillers, restarts and clean cut segments of one clip's speech. |
| `speech_takes` | The best take of each line across several recordings of a script. |
| `speech_refine` | Transcribe chosen clips again with a stronger model and the brand vocabulary (detached job). |
| `library_overview` | Overview of the footage folders without indexing, plus a mosaic. |
| `frame_view` | The full-resolution frame ON SCREEN at `t` (`floor(t * fps)`, a `t` within 1 ms of a frame boundary is that frame) or at `frame`, with its exact time. |
| `reference_decode` | Decode a reference video into shots, pacing, text, speech and sheets. |
| `reference_breakdown` | Save the agent's shot-by-shot breakdown of the reference. |
| `recipe_list` | List the ad recipes with purpose and duration. |
| `recipe_get` | One full recipe with its rules and an example. |
| `editing_guide` | The editing doctrine for footage, copy, rendering and QC. |
| `storyboard_schema` | The JSON Schema of the storyboard. |
| `storyboard_save` | Save a storyboard as `storyboards/<name>.json` once; later tools take that path instead of the object. |
| `storyboard_patch` | Edit a saved storyboard with a list of `{path, value}` changes. |
| `caption_preview` | A caption or comment drawn over a real 9:16 frame. |
| `end_card` | An end card image composed from a real product photo. |
| `storyboard_validate` | Validate a storyboard against its schema and recipe rules. |
| `render_start` | Validate and queue a background render (preview or final). |
| `voice_generate` | Queue a checked voice-over generation job. |
| `voice_view` | Read a voice manifest and suggested shot boundaries. |
| `music_profile` | Tempo, loudness and energy of the music tracks. |
| `music_search` | Find music whose licence allows an ad (CC0, PDM, CC BY, CC BY-SA) on Openverse and Freesound; `fit_to="storyboard"` ranks for an ad. |
| `music_fetch` | Download a found track into `music/` with its licence record (`.licence.json`, `LICENCES.txt`), then profile it. |
| `music_generate` | Make a track locally (optional ACE-Step environment, `adanvil music-setup`), as a job. |
| `music_beats` | Beat times of a music track as it will play in the ad. |
| `sfx_list` | List the project's sound effects; `fetch=true` builds the CC0 sound pack in `sfx/`. |
| `storyboard_snap` | Move the cuts of a storyboard onto the beats of its music. |
| `job_status` | Status of a render, batch, voice or index job, with optional waiting. |
| `qc_run` | Technical QC on a rendered video, with control images (overlay crops follow the re-laid captions of a format; `caption_on_face`). Optional `storyboard`: the effective storyboard of that render. |
| `ad_score` | Creative score out of 100 of a rendered ad (hook, pacing, reading, story, platform safe zones, audio) with findings to fix and a safe zones preview. |
| `variants_build` | The matrix of variants of a storyboard (hooks x bodies x calls to action, up to 24): patches or slot replacements, validated and saved, with a contact sheet of the first and last 3 s of each. |
| `render_batch` | One detached job that renders many storyboards (and formats) with a shared segment cache and writes `renders/<batch>/manifest.csv`. |
| `timeline_export` | Export a storyboard as OpenTimelineIO, Final Cut XML, EDL and SRT plus the overlay PNGs (named by storyboard index), the sound effects and the music levels, with the real media descriptors and a `not_exported` list, to finish it in DaVinci Resolve or Premiere Pro (CLI: `adanvil export`). |
| `render_formats` | One storyboard in 9:16, 4:5 and 1:1 (re-cropped footage, captions re-laid out by bands, safe zones of the placement). |
| `compare_pair` | How different a before and an after look (0 to 100), on the face or the whole frame, with a side-by-side preview, to pick the pair. |
| `vision_setup` | Install the optional vision environment (search by sentence, automatic tags, product versions, faces). |

CLI commands (`adanvil <command> --help`): `mcp`, `voice-setup`, `vision-setup`, `init`, `cat`, `onboard`, `index`, `decode`, `recipes`, `validate`, `render`, `voice`, `qc`, `preview-caption`, `end-card`, `sfx-fetch`. Add `--json` for machine-readable output.

## Smart library (optional)

Finding the right shot is the slowest part of an edit. `adanvil vision-setup` (or the `vision_setup` tool) installs, once, a separate environment (about 2.5 GB, CPU only, no GPU needed) with a small multilingual image and text model ([SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224), Apache-2.0) and a face detector (YuNet, MIT); the licences are in `THIRD_PARTY.md`. Nothing else changes without it, and `library_index` then says `vision: unavailable`.

- **Index with vision.** `library_index` also embeds the frames of each contact sheet and looks for faces (`cache/vision/<file key>.npz`), and gives every entry `auto_tags` (a fixed label list such as `person`, `bathroom`, `hands`, `smiling`, `text on screen`), kept apart from your own `tags`.
- **Search by sentence.** `library_search(query="a woman with a sad face touching her jaw")`, in any language, ranks shots across the whole library, with the usual filters applied first, and returns the best ones with one numbered mosaic: no sheet to open.
- **Product versions from photos.** `brand.yaml` lists `product_versions: [{name: current, photos: [...]}, {name: old, photos: [...]}]`; indexing guesses which version each shot shows, and `library_search(product_version="current")` uses the guess unless a human annotation exists (a human annotation always wins). Give photos, not words.
- **Face-aware framing.** `focus: "face"` keeps the main face centred at `face_y` (0.38 of the frame by default) on clips, images, split panels and reveal sides; a `reveal` also puts both faces at the same place and size by default (`align_faces`). `compare_pair` uses the face detector too.

## Voice-over (optional)

AdAnvil can add a generated voice-over with [Chatterbox](https://github.com/resemble-ai/chatterbox), an open-source text-to-speech model, with captions synced to the spoken words.

- Run `adanvil voice-setup` once. It creates a separate Python environment (in the AdAnvil cache folder) and downloads about 3 GB of models. `adanvil voice-setup --engine kokoro` adds the fast engine (below) to the same environment.
- Generation runs on the CPU. One worker process per job loads each model once and serves every step (speech, alignment, check). Measured on an idle 8-core server (French, 3 to 9 lines of about 5 seconds): the Chatterbox voice takes about 25 to 35 seconds per line, so 9 lines take about 5.5 minutes (it took 10.5 before V2.6); a busy machine is slower, count up to 5 minutes per sentence.
- Two engines. `chatterbox` is the default and the recommended voice for ads (the owner listened to both and kept it): it clones the `reference` voice, so it also fits a brand that has a voice to match. `kokoro` (Apache-2.0) has fixed voices and no cloning (`{"engine": "kokoro", "voice": "ff_siwis"}`, French `ff_siwis`, English `af_heart` by default, `bf_emma` and others for British English): a fast option for drafts and previews (check the script, the timing and the captions before paying for the real voice), about 25 seconds for 9 lines plus 50 seconds of alignment and check on the same server. It speaks French and English only.
- The pronunciation check reuses the alignment transcript when every word is confident (`check_confidence`, default 0.6) and the line reads as written; only doubtful lines are transcribed again by the `check_model`. Set `check_confidence` to `null` (or `ADANVIL_VOICE_CHECK_CONFIDENCE=off`) to check every line with the `check_model`.
- The voice speaks for the brand, as a narrator. It must not pose as a fake customer.
- The voice language is the `language` of the voice spec. When the spec does not set it, AdAnvil uses the `language` of the project's `brand.yaml` (`fr-FR` counts as `fr`), and `en` when the brand has none.
- Chatterbox embeds a Perth watermark in its audio. AdAnvil keeps it. Do not remove it.
- Use only a reference voice you have the right to use.

## Talking-head UGC (optional)

A creator talking to camera is cut from the speech in the footage. `library_index` transcribes the clips and audio files that have sound (word timestamps, in the same job, once per file, each file stored as soon as it is done), with faster-whisper when it is installed (`pip install 'adanvil[transcribe]'`) or else the voice environment (`adanvil voice-setup`); without either, indexing still succeeds and the entries say so. Indexing uses a fast model to make the speech searchable; `speech_refine` transcribes the clips you keep again with a stronger one, and the brand name, the product name and `speech.vocabulary` of `brand.yaml` are given to the transcriber. A voice recorded in a separate audio file is found by name (`separate_audio_candidates` on a video with silent sound) and laid with `audio_from`; syncing is up to the agent, on a preview. Then `speech_takes` picks the best take of each line, `speech_view` gives the cuts that keep the speech and drop pauses and hesitations, and a storyboard with `audio.clips`, `audio_from` (B-roll over the voice) and a `speech_captions` overlay renders the ad. The `ugc_talking` recipe puts it together; the guides explain each step.

- `ADANVIL_SPEECH_MODEL` sets the indexing model (default `base`: fast, good enough to search). `speech_refine` uses `medium` unless told otherwise. The language is the `language` of `brand.yaml`, else auto-detected.
- `ADANVIL_SPEECH=off` stops `library_index` from transcribing on its own (`transcribe="auto"`); `transcribe=true` still does.
- The creator's words are claims: the validator warns `banned_word_spoken` when a kept word is on the brand's banned list, and `cut_mid_word` when a cut falls inside a word (not when the voice goes on across the cut). The fillers that are cut (euh, um, uh, er, genre...) can be replaced in `brand.yaml` under `speech`, which also takes a `vocabulary` list of words to spell right. A misheard word can be fixed in the captions with `speech_captions.corrections`, close spellings only, and every correction is listed for review (`speech_corrections`).

## Visual devices (optional)

A `reveal` segment (a divider line sweeps a before picture away to show the after picture, from two real pictures or two clips, with labels that follow their side) for a before/after the viewer watches happen, a `split` segment (side by side or top and bottom, with labels such as FROM THIS and TO THIS) for comparisons, an `image` overlay with `animation`, `frame` and `rotate` (a photo or a review screenshot that comes in over the video as a card), a `video` overlay (picture-in-picture, with a rect, rounded or circle shape, border and shadow), `sticker` overlays (arrow, hand-drawn circle and underline, badge, emoji, image, with a pop or draw animation), a `progress` bar and a `punch` (instant zoom) on a clip. Use one or two per ad. Fields and rules: [AGENTS.md](AGENTS.md) and [the editing guide](docs/EDITING_GUIDE.md).

## Sound design (optional)

Sounds go over the audio bed of an ad: a whoosh on a cut, a pop on a caption, a ding on the price, a riser before the reveal.

- `adanvil sfx-fetch <project_dir>` (or `sfx_list` with `fetch=true`) builds a pack of about 18 short sounds in the project's `sfx/` folder. Each one is downloaded from a pinned CC0 source (checked by SHA-1) or synthesised in code, then trimmed and set to one level. Nothing is stored in this repository; sources and licences are listed in `examples/sfx/README.md` and written to `sfx/LICENCES.txt`. Your own files in `sfx/` work too.
- `{"type": "sfx", "source": "sfx/pop.wav", "at": 3.2, "volume": 0.8}` places a sound; `audio.sfx_auto` places them at the cuts and at the captions. Sounds never clip and never run past the end.
- `music_beats` gives the beats of a track as it will play, and `storyboard_snap` moves the cuts onto them (the validator reports how many cuts are already on a beat).

## Performance settings

- `ADANVIL_WORKERS`: how many files, thumbnails or segments are processed at once. Default `max(1, min(4, cpus // 2))`, and at most 2 when the machine has under 12 GB of RAM.
- `ADANVIL_FFMPEG_THREADS`: threads of each ffmpeg decoder or encoder. Default `max(1, cpus // workers)`.
- Voice: `ADANVIL_VOICE_CHECK_MODEL` sets the Whisper model of the pronunciation check (unless the voice spec sets `check_model`). `ADANVIL_VOICE_THREADS` sets the torch and Whisper threads (default: usable cores). `ADANVIL_VOICE_CHECK_CONFIDENCE` is the word confidence from 0 to 1 above which the alignment transcript replaces the check model (`off` to always run the check model; the spec's `check_confidence` wins). `ADANVIL_VOICE_PERSISTENT=0` goes back to one process per step instead of one worker per job.
- Speech: `ADANVIL_SPEECH_MODEL` (default `base`), `ADANVIL_SPEECH=off` (no automatic transcription), `ADANVIL_SPEECH_PROMPT` (an optional text that biases the transcriber, for example towards keeping hesitations).
- Other: `ADANVIL_CACHE` moves the cache folder; `ADANVIL_FFMPEG` and `ADANVIL_FFPROBE` point to the binaries (absolute paths; otherwise they are looked up on `PATH`, never in the current folder).
- Security: `ADANVIL_PROJECTS_DIR` keeps every project under one folder; `ADANVIL_ALLOW_PRIVATE_URLS=1` lets `brand_onboard` and the downloads reach `localhost` and the local network (development only). See [`SECURITY.md`](SECURITY.md).
- Benchmark: `uv run python scripts/bench.py --files 12 --seconds 4`.

## Responsible use

AdAnvil edits ads; you answer for what they say and show.

- Use real comments and real customer words only. The tool never invents an offer, a price, a guarantee or a claim: the human confirms them.
- Get the consent of the people who appear in your footage.
- Check the licences of the music and the footage you use (the demo pack is CC0, your own files may not be).
- Make sure every claim in the ad is true and allowed in your market.
- Do not clone a voice without the right to do so, and do not present a generated voice as a real customer.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licence and status

MIT, see [`LICENSE`](LICENSE). Fonts, emoji and other third-party material are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).

Status: alpha, version `0.1.0`. Interfaces can still change. See [`CHANGELOG.md`](CHANGELOG.md).
