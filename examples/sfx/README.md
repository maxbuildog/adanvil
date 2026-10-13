# Sound-effects pack

`adanvil sfx-fetch <project_dir>` (or the `sfx_list` tool with `fetch=true`) builds a small pack of sound effects in the project's `sfx/` folder. **No sound is stored in this repository.** Every sound is either

- **downloaded** from a pinned source whose licence is **CC0** (the file is taken from a ZIP archive whose SHA-1 is checked; a mismatch discards the download), or
- **generated** by AdAnvil: synthesised in code from sines, noise and envelopes (a seeded generator, so the same bytes every time). A sound whose download fails is generated instead, and `--offline` generates all of them.

Every sound, downloaded or generated, goes through the same finishing: trimmed, faded in and out over a few milliseconds, then set to one level so that its loudest 400 ms is at -16 LUFS and its peak never goes above -1.5 dBFS (short sounds are limited by the peak). The result is a mono 16-bit WAV at 48 kHz in `sfx/<name>.wav`, with `sfx/pack.json` (what each sound is and where it comes from) and `sfx/LICENCES.txt`.

The manifest is `src/adanvil/assets/sfx_pack.json`. Your own files dropped in `sfx/` work the same way.

## The sounds

| Sound | Origin | What it is for |
| --- | --- | --- |
| `whoosh_short` | generated | a cut or a quick move (0.4 s) |
| `whoosh_long` | generated | a slow transition or a pan (1 s) |
| `swoosh` | generated | an element sliding in (0.6 s, falling sweep) |
| `swipe` | generated | a flick or a screen swipe (0.3 s) |
| `pop` | generated | a caption or an element appearing |
| `click` | CC0 download | a tiny interface click |
| `tap` | CC0 download | a finger or a button |
| `ding` | CC0 download | a price, a result, a point scored |
| `chime` | CC0 download | a success or a reveal |
| `notification` | CC0 download | a notification blip |
| `cash_register` | generated | a price or a sale (drawer, then a double bell) |
| `camera_shutter` | generated | a photo, a before/after |
| `record_scratch` | generated | a stop, a twist, a mistake |
| `typing` | generated | keys under typed text |
| `riser_1s` | generated | a one-second build-up that ends at its peak, before a reveal |
| `riser_2s` | generated | a two-second build-up |
| `impact` | CC0 download | a hit on a reveal or a hard cut |
| `boom` | generated | a sub boom with a long tail for a big reveal |

A downloaded sound has a generated fallback with the same name, so the pack is always complete.

## Sources of the downloaded sounds

All of them come from [Kenney](https://kenney.nl) (Kenney Vleugels), who releases these packs under **Creative Commons CC0 1.0** (public domain dedication): the licence is stated on each pack's page ("License: Creative Commons CC0", linking to <https://creativecommons.org/publicdomain/zero/1.0/>) and in the `License.txt` inside each archive. Credit is not required; it is given anyway in `sfx/LICENCES.txt`.

| Sound | Pack and page | File in the archive | Archive (pinned URL, SHA-1) | Licence |
| --- | --- | --- | --- | --- |
| `click` | Interface Sounds, <https://kenney.nl/assets/interface-sounds> | `Audio/click_003.ogg` | `https://kenney.nl/media/pages/assets/interface-sounds/fa43c1dd4d-1677589452/kenney_interface-sounds.zip`, `374a0ba3ffa61f62fd58102507065c417eef5174` | CC0-1.0 |
| `tap` | Interface Sounds | `Audio/tick_004.ogg` | same archive | CC0-1.0 |
| `ding` | Interface Sounds | `Audio/glass_001.ogg` | same archive | CC0-1.0 |
| `chime` | Interface Sounds | `Audio/confirmation_002.ogg` | same archive | CC0-1.0 |
| `notification` | Interface Sounds | `Audio/confirmation_003.ogg` | same archive | CC0-1.0 |
| `impact` | Impact Sounds, <https://kenney.nl/assets/impact-sounds> | `Audio/impactPunch_heavy_000.ogg` | `https://kenney.nl/media/pages/assets/impact-sounds/87b4ddecda-1677589768/kenney_impact-sounds.zip`, `a2c54979c9d8f43e804eea667c1bca0407bf4022` | CC0-1.0 |

The other twelve sounds (`whoosh_short`, `whoosh_long`, `swoosh`, `swipe`, `pop`, `cash_register`, `camera_shutter`, `record_scratch`, `typing`, `riser_1s`, `riser_2s`, `boom`) are generated: they have no source and no licence to carry.

Licence rule of this pack: **CC0 only.** A sound whose licence cannot be verified on its source page is not added: it is synthesised instead. When adding a sound, pin the archive URL and its SHA-1 in the manifest, name the file inside it, and list it in the table above and in `THIRD_PARTY.md`.
