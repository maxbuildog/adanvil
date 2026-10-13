# Changelog

## 0.3.0 (2026-10-13)

Install in one step (Claude Desktop extension, Docker image) and hand an ad to a human editor (`timeline_export`: OpenTimelineIO, Final Cut Pro 7 XML for Premiere Pro and DaVinci Resolve, CMX 3600 EDL, SRT and the captions as transparent PNGs, with the sound effects, music levels, licences and an honest list of what an editor format cannot carry). Each step below was checked by an agent editing a real ad. Not yet proven: an actual import into Premiere Pro or DaVinci Resolve.

V2.10e: last small fixes of the export.

- A clip or split/reveal panel that needs more source than its file has is clamped to the file followed by a freeze of the last frame (what the render does), with a warning; `storyboard_validate` warns `panel_source_short`.
- Sound effect files are declared at their real length (OTIO exact; XML whole frames with the trim named in the warnings).
- Variable-frame-rate sources are listed with their measured average. The licences and attribution texts of the music and sound effects are in `README.txt` and the result (`licences`).
- Tests: `test_v210e_export_last.py`.

V2.10d: the follow-up to the r22 check of the export.

- `qc_run`: a render report written before entries carried `overlay` and `text` is named from the storyboard, so crops are `overlay_<storyboard index>_<few words>.jpg` through the MCP tool in every path.
- `timeline_export`: `export_manifest.json` lists the files an export owns; a re-export cleans them, and a folder without the manifest is refused unless `overwrite=true` (which clears only export files). The relative and portable EDL uses relative paths and `relink.txt` states the Premiere Pro finding. One nominal frame-rate rule (the catalog's), every differing source listed, all timecodes start at 01:00:00:00, segments counted from 0 everywhere, sfx tracks named with their storyboard overlay index, sfx clips floored to their file length.
- Tests: `test_v210d_export_followup.py`.

V2.10c: the fixes from the r21 check of the export.

- `timeline_export`: captions are markers on `V1` (no more tracks of offline clips); speed follows the xmeml and CMX 3600 conventions (`out - in = end - start` with a Time Remap filter, source out = source in + record duration with an `M2` line); every `<file>` and OTIO `available_range` has the real rate, size, duration and start timecode of its source, in/out are counted in the file's own frames and EDL source times start at the file's timecode; absolute paths keep the project path (no symlink resolution), OTIO `target_url` is a `file://` URL, a relative export writes `file://localhost/RELINK_ROOT/...` plus `relink.txt` (or absolute paths with `media_root`), `copy_media` prefixes colliding names with a hash and does not copy the overlay PNGs, and each media mode has its own folder (`-relative`, `-portable`; `overwrite=true` to replace another mode); sound effects (one audio track each) and the music level and fades (Audio Levels keyframes in the XML) are exported; the result and a `README.txt` carry a `not_exported` list; overlays are named by storyboard index everywhere.
- QC: crops are named `overlay_<index>_<few words>.jpg`, never cut a neighbouring caption in half, `endcard.jpg` is no longer saved when `end.jpg` is the same frame, and `overlay_visible` values carry their unit.
- `frame_view` takes `frame=` as well as `t`, and a `t` within 1 ms of a frame boundary is that frame (10.9333 s at 30 fps is frame 328).
- A badge `pop` draws one state at a time (the last pop frame no longer runs to the end of the sticker over the settled state).
- Tests: `test_v210c_export_fixes.py`.

V2.10b: export to an editing program, and the fixes from the r20 check.

- `timeline_export` (CLI `adanvil export`): writes `exports/<name>/` with an OpenTimelineIO file (hand-written, no dependency: video tracks, speed as a time warp, stills, captions as clips with markers, overlays, audio), a Final Cut Pro 7 XML (xmeml 4), a CMX 3600 EDL (video cuts), an SRT of every caption at its final time, and each caption, sticker and card as a full-canvas transparent PNG listed in `overlays.json`. Absolute media paths by default, `relative` and `copy_media` for a portable folder.
- `frame_view` returns the frame on screen at `t` (`floor(t * fps)`) and states its time (`frame`, `frame_t`); the QC and score samples use the same rule, which removes the false `caption_on_face` at a cut.
- A `LICENCES.txt` line also gives the title, author and source of the track.
- QC: a caption crop holds the whole caption box (translucent boxes and shadows included) and is named after its storyboard overlay (the render report tags each entry with `overlay`); split screen and reveal images are returned; the variants contact sheet is `<base>_<first variant>_<n>v_sheet.jpg`.
- `render.compose.build_overlays` is the overlay half of the render, shared with the export.
- Tests: `test_v210b_export.py`.

V2.10a: install in one step.

- Claude Desktop extension (MCPB, manifest 0.4, server type `uv`): `packaging/mcpb/` and `scripts/build_mcpb.py` build `dist/adanvil-<version>.mcpb` (`--check` verifies the manifest). Version and tools list are generated from the package and the server; settings for the projects folder and an optional sensitive `FREESOUND_API_KEY`. The launcher stops with the install command for the system when ffmpeg or ffprobe is missing.
- `Dockerfile` and `.dockerignore`: Ubuntu 24.04 (ffmpeg 6.1), non-root user, `adanvil-mcp` over stdio, `/projects` volume, `EXTRAS` build argument (for example `transcribe`). The voice, vision and music-generation environments are not in the image.
- CI builds the Docker image and the `.mcpb` file (nothing is pushed).
- README "Install" has three paths (Claude Desktop extension, Docker, `uvx`); `AGENTS.md` and `skill/SKILL.md` mention the other two.
- Tests: `test_packaging.py` (manifest, bundle, launcher message, Docker end to end when Docker is present).

## 0.2.0 (2026-10-13)

V2.9b: fixes from the Opus check of V2.8e and V2.9.

- A picture embedded in an audio file (`attached_pic`, or one still mjpeg/png frame in a file with audio) no longer makes it "video": validation, `music_profile`, `music_beats` and the audio bed accept fetched mp3 files; `music_fetch` strips the picture (`-map 0:a -c copy`).
- The licence of a track with no `.licence.json` is read from its line in `music/LICENCES.txt` (matched by file name); the report says where it came from (`read_from`).
- `fit.feel_guess` comes from the track's own title, tags and genres (`calm`, `medium`, `energetic`, `unknown`), never from the requested feel; `fit_to` ranks by it.
- `variants_build` refuses to overwrite ANY existing storyboard unless `overwrite=true`; the error lists every file.
- Batch reports give the real size of the file written; the manifest has a `quality` column (`draft` stays an alias of `preview`).
- Overlay starts and ends snap to the frame grid (nearest frame, the cut's frame when within half a frame of a cut): no caption flashes one frame after a cut.
- Tests: `test_v29b_music_fixes.py`.

V2.8e: placement by format, self-checked.

- A move is checked, not predicted: after choosing a spot the caption is moved there, measured, and `caption_on_face` is run on that final position against the faces of every sampled moment of its time range (every 0.2 s); a spot that fails gives way to the next (other side, then smaller), then to the spot with the fewest covered looks, and only then the note says `cannot avoid a face`. The faces under a caption are detected on the footage at each moment with the detector and frame size `qc_run` uses (the stored index is too sparse for a head that turns).
- Notes and QC agree: one final check on the final position (after the per-format overrides) gives the render notes; `qc_run` looks at four frames per caption and, when the source face data and the rendered-frame detection disagree, reports both.
- A badge or sticker placed by `x` shrinks around its centre in another format.
- Manifest `CTA text` lists each text with its role, most prominent first (`caption: ... | badge: ...`), so A and B differ even when they share phrases in other roles.
- `variants_build` refuses to overwrite a storyboard file an earlier `variants_build` of the same base did not make (`overwrite=true` to force); the error names the file.
- `draft` is an alias of `preview` for every quality argument and `output.quality`; the manifest score is filled for preview renders too (scored on the preview file).
- Tests: `test_v28e_layout_selfcheck.py` (one per item; the face boxes of a real ad segment).

V2.9: music, with its licence.

- `music_search(project_dir, query, mood, min_duration, max_duration, source, limit, fit_to, storyboard, feel)`: Openverse (no key) and, with `FREESOUND_API_KEY`, Freesound. Only CC0, PDM, CC BY and CC BY-SA are listed (NC and ND are dropped, also Freesound's NonCommercial and Sampling+). Each track: title, author, licence, duration, source URL, preview URL, `attribution_required` and the attribution text. No key: a note, never a failure. `fit_to="storyboard"` ranks the tracks at least as long as the ad first, then the feel guessed from title and tags (`audio.music_feel`, new optional storyboard field).
- `music_fetch(project_dir, source, id)`: downloads into `music/`, writes `<file>.licence.json` and appends `music/LICENCES.txt`, then returns tempo, feel and loudness (`music_profile`). Refuses any other licence.
- The render report has `music` (status, licence, attribution text) and the warnings `music_licence_unknown` (no record) and `music_attribution_required` (with the exact text); `ad_score` has `facts.audio.music_licence` and the same two findings (the first costs 3 points, the second none); the batch manifest has a `music licence` column (older manifests are migrated).
- Local generation: ACE-Step v1 3.5B qualifies (code and weights Apache-2.0). Optional `adanvil music-setup` + `music_generate(project_dir, prompt, duration)` (a job, kind `music`, `job_status.music`); measured 567 s for 30 s on 8 CPU cores, 18 GB of memory. MusicGen (NC), AudioLDM 2 (NC), Stable Audio Open (gated, revenue-limited licence) were rejected: THIRD_PARTY.md section 1.8.
- Docs: music bullet in guide section 7, AGENTS.md and SKILL.md "Music"; tools 39 to 42.
- Tests: `test_v29_music.py` (recorded-shape Openverse and Freesound responses, licence filter, sidecar and LICENCES.txt, warnings, manifest, ranking, generation with a fake worker).

V2.8d: caption placement by format and review sheet, fixes from the V2.8c test.

- Formats: a caption keeps the position the re-layout gave it when it is free of faces; otherwise it moves the least (nearest free spot, same side of the face first, clear of the busy content of a still and, when possible, of the other overlays on screen), and each move is reported with its distance. A hook at the top of 4:5 no longer jumps 969 px to the bathrobe.
- One face test: `captionfaces.caption_on_face` is used by the format layout, `qc_run` and `ad_score`, measured against the face core (eye line to chin, small margin; hair and forehead do not count). Detections under confidence 0.88 (drawings, diagrams) or under 6% of the picture width are ignored.
- `variants_build` contact sheet: the second row is the 3 s before the end card when the ad ends on an image card (the CTA part), so CTA variants can be told apart.
- Manifest: new `CTA id` column; `CTA text` lists only the texts that differ between the CTA variants of the batch (whole text with no CTA axis). Older manifests are migrated when merged.
- Tests: `test_v28d_placement_faces.py`.

V2.8c: variants and formats, fixes from a real test.

- `variants_build`: `changes: []` (or no `changes`) is the control; the overlap tests use the shared 0.001 s tolerance (a slot range built from summed floats counts as inside); `sfx` overlays follow the slot (shifted after it, dropped inside it when the new slot brings its own, listed under `moved`); every error of every variant comes back at once (`errors`: variant, path, message); the contact sheet shows the first 3 s and the last 3 s of each variant; a base that is itself a variant keeps its ids.
- Manifest: `CTA text` is the text of the last CTA part, so CTA A and CTA B differ; rendering into an existing batch folder merges the rows by file name.
- Formats: a caption never sits on a face (stored vision faces, else the vision worker, else skipped with a note; moves to the nearest free band, `caption_on_face` when it cannot); `ad_score` and `qc_run` flag `caption_on_face` on the rendered frames of every format; an overlay may carry `formats: {"1:1": {x, y, size, anchor, hidden}}`, applied after the re-layout; a 9:16 storyboard renders as is in 9:16; `qc_run(storyboard=...)` crops the overlays where the effective storyboard of the format has them.
- Stickers: `x: "center"` or `anchor: "center"` centres a badge (or any sticker) across the frame.
- Tests: `test_v28c_variants_usability.py` (one test per item, fake vision faces for the caption checks).

V2.8: variants and formats.

- `variants_build(project_dir, base, variants, name_pattern="{base}_{hook}_{cta}")`: the cartesian product of up to 3 axes (24 variants at most) of a base storyboard. An option is a list of `storyboard_patch` changes and/or a slot replacement `{replace_label, segments, overlays}`: the segments of that label (one run) are swapped, everything after moves by the difference of duration (overlays, word times, the end of overlays that span the slot), the overlays timed on the old segments go away and the new ones are timed from the slot start. Refused with a clear message: a voice-over (`audio.voice`) that would no longer match, an overlay half inside the slot, a label that is not one run, an invalid segment, two variants with one name, more than 24 or 3 axes. Each variant is validated and saved as `storyboards/<name>.json` (+ `<name>.variant.json` with its ids); the result has durations, warnings and a contact sheet of the first 3 s of every variant (built from whole segments, so it already fills the cache).
- `render_batch(project_dir, storyboards, quality, batch, formats, score)`: one detached job (`kind: "batch"`) renders the storyboards in order with one `Cache`, so shared segments are built once; progress per render, `job_status` lists each output as it finishes and reports `total_seconds` and `cache_builds`; a render that fails is listed and the batch goes on.
- Formats: new storyboard field `formats: ["9:16", "4:5", "1:1"]` and tool `render_formats`. `reformat` lays the same storyboard out for another ratio: segments are re-cropped by the existing cover (face-aware with `focus: "face"`), an end card that was `cover` becomes `contain` over its blur, vertical positions of captions, stickers, images and insets are mapped band by band between the free area of the two placements (top bar to bottom band), then texts, stickers and images that still leave it are pushed back inside using their measured boxes. Files `<name>_9x16.mp4`, `<name>_4x5.mp4`, `<name>_1x1.mp4`.
- New safe zone placement `feed` (4:5 and 1:1): `ad_score` and `qc_run` pick it by the size of the video (explicit `platforms` still wins). The preview render scales to half size by even numbers for any ratio.
- `renders/<batch>/manifest.csv`: file, variant ids, format, duration, score, hook text, CTA text, ready for an ads manager.
- Docs: "Testing variants and formats" (guide section 11, skill step 13).
- Tests: `test_v28_variants_formats.py` (matrix expansion and shifting, refusals, voice, batch with shared cache counted on the segment builds, manifest, 4:5 and 1:1 layouts inside frame and Feed zones, face kept in frame, still shown whole, feed score and QC, the tools through the server).

V2.7: smart library (optional vision environment).

- `adanvil vision-setup` / tool `vision_setup`: an isolated environment (CPU) with SigLIP 2 Base (Apache-2.0, multilingual image and text embeddings, pinned revision) and the YuNet face detector (MIT, SHA-256 checked), behind a persistent worker (`vision/worker.py --serve`, kept alive ten minutes between calls). Licences read at the source pages and recorded in `THIRD_PARTY.md` (section 1.7); OpenAI CLIP weights rejected (deployed use out of scope on the model card). No new dependency in the base package.
- `library_index(vision="auto" | true | false)`: frames of the contact sheet embedded and searched for faces (`cache/vision/<file key>.npz`), `auto_tags` with scores from a fixed label list (never mixed with human `tags`), `vision` status per entry (`unavailable: ...` without the environment; indexing never fails for it), re-tagging from stored embeddings when the labels or the reference photos change. Index jobs report a `vision` stage.
- `library_search(query=..., limit=12, auto_tags=[...])`: shots ranked by text-image similarity (entry, shot, time range, best frame, probability score), the usual filters applied first, plus one numbered mosaic image.
- `brand.yaml` `product_versions` can be a list of `{name, photos}` (the old text answer still works). Indexing guesses a product version and confidence per shot (`shots[].product_version`), `auto_product` on the entry; `library_search(product_version=...)` uses the guess when there is no human annotation, a human annotation always wins.
- `focus: "face"` and `face_y` (default 0.38) on clip, image, split panel and reveal side; the crop follows the smoothed main face of a clip. `reveal` puts both faces at the same place and size when each side shows one face (`align_faces`, default true, enlargement capped at 1.6x). `compare_pair` finds faces with the detector when installed (`face_source`), else the skin heuristic.
- Measured on 22 real rushes (8 idle-ish CPU cores): the vision pass over the whole set took 2 min 10 s (about 6 s per entry, models loaded once); indexing again with nothing changed takes 2 s; a sentence search takes under 1 s once the worker is up (the first call loads the model, about 10 s). The probability of a match is centred on the similarity measured on this footage (label on screen: cosine 0.11 to 0.14, not on screen: 0.03 to 0.10), because SigLIP's own bias gives zeros for plain labels.
- Honest limit of the product version guess: on two views of ONE device (the only "versions" that footage had) the guess is right on 52 % of the frames where it sees the product (35 of 67), the product is seen on 61 % of the frames that show it (67 of 109) with 4 false alarms on 74 frames without it, 57 % exact over the three answers; matching whole frames against whole photos separates different packagings far better than a small change on one product. Give 3 to 5 photos per version and confirm the guess on the sheet.
- Docs: Chatterbox is the default and recommended voice for ads (owner verdict after listening to both), Kokoro a fast option for drafts and previews.
- Tests: `test_v27_vision.py` with a deterministic fake engine (storage, ranking, tags, version matching, face maths, rendered face positions) and one real-worker test skipped when the environment is absent; the suite sets `ADANVIL_VISION=off`.

V2.6: faster voice-over.

- One persistent worker per voice job (`worker.py --serve`, JSON lines on stdin and stdout): the models load once and serve speech, alignment and check, instead of one process per step that reloaded them. The job falls back to the old one-process-per-batch mode if the worker cannot start or dies (the lines already done are kept); `ADANVIL_VOICE_PERSISTENT=0` forces the old mode. Speech transcription uses the same protocol.
- The reference voice is analysed once per job instead of once per line.
- Cheaper pronunciation check: the transcript of the alignment is the check when `align_model` and `check_model` are the same, or when every word is at least `check_confidence` (default 0.6, `ADANVIL_VOICE_CHECK_CONFIDENCE`) and the line reads as written. Doubtful lines still go to `check_model`. Word times come from the alignment, moved onto the trimmed file.
- New optional `kokoro` engine (Apache-2.0 code and weights, French `ff_siwis`, English `af_heart`, fixed voices, no cloning): `adanvil voice-setup --engine kokoro`, `VoiceSpec.engine: "kokoro"`, `VoiceSpec.voice`. One attempt per line (a retry would sound the same).
- Threads: `ADANVIL_VOICE_THREADS` also sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` of the worker. Measured: two TTS processes at once do not make a job faster (the TTS stage took 249 s against 253 s sequential for 9 lines), so lines stay sequential.
- Measured on an idle 8-core server, French, Chatterbox: 3 lines 395 s then 166 s, 9 lines 621 s then 332 s. Kokoro: 9 lines 74 s.
- Tests: `test_voice_speed.py` (protocol, one load per model, crash fallback, timeout, streaming, shared and skipped checks, kokoro).

Stills no longer tremble.

- Every slow zoom (image `zoom_in`, `reveal` `motion`, clip `zoom` and the ramp of `freeze`) is drawn with the `perspective` filter driven by the frame number, which resamples the picture at 1/256 of a pixel with a cubic kernel, instead of a scale to an even integer size followed by an integer crop (the picture stood still for a few frames, then jumped by a pixel). The same on ffmpeg 6.1 and 9, exact frame counts. Cache keys of zoomed pictures change.
- Picture cards (`image` overlay): the slide offsets and the `pop` scale are placed at sub-pixel precision (the card is resampled onto the pixel grid), not rounded to whole pixels.
- Docs: stills stay `static` unless a smooth zoom adds something (1.00 to 1.06), and a before/after prefers the brand's real creator photos over an anonymous composite from the website.
- Tests: `test_still_smoothness.py` tracks a soft line of a fine grid frame by frame through each path: it follows the curve of the zoom within half a pixel and never stands still or goes back.

V2.5: the creative score of an ad, and the choice of a before/after pair.

- `ad_score(project_dir, video, storyboard=None, platforms=None)`: a score out of 100 for a rendered ad, with six sub-scores (hook 20, pacing 15, reading 20, story 20, safe zones 10, audio 15) and findings sorted by impact, each with a time, a severity and a fix. Hook: a cut or motion in the first second, a face or the product, a written text within 0.5 s, audio at once. Pacing: cuts per 10 s by quarter, the longest shot, a shot over 4 s in the middle of a fast ad. Reading: 0.25 s per word + 0.6 s on screen (at least 1.2 s), at most 7 words per line, a WCAG-like contrast of 3:1 or more against the pixels around the letters on the real frames. Story: product after the problem, product shown, call to action written or spoken, an end card or call to action in the last 4 s. Audio: loudness, voice 6 dB over the music, no silence of 0.7 s or more. It uses the storyboard for exact timings and words, and runs on the pixels and the sound without it (the skipped checks are listed). Images: `safe_zones_preview.jpg` and the frame of each top finding. Thresholds are guidelines, documented in `qc/score.py` and the guides.
- Platform safe zones (`safezones.py`): the top bar, the right-side button column and the bottom band of `tiktok`, `reels` and `stories` as pixel boxes on a 1080x1920 frame, with findings when a caption or a key sticker falls inside.
- `compare_pair(project_dir, before, after, before_t, after_t, before_crop, after_crop)`: `visible_difference` from 0 to 100 between two pictures (images, or clip frames at given times, optionally cut with a `crop`), computed on the face when a skin-region heuristic finds one in both (no new dependency), else on the whole frame after aligning sizes, with a side-by-side preview. 35 tools.
- `reveal` and `split` image sides accept `crop` (`{x, y, w, h}` in source pixels) so one half of a brand's side-by-side composite is a side with no file prepared; an out-of-frame crop is a `panel_crop` error.
- `storyboard_validate(score_check=true)` adds the pixel-free checks of the score as `score_*` warnings (reading time, line length, product too early or never shown, call to action, safe zones) and `summary.score_precheck`.
- Docs: "Before delivery: score the ad", the safe zones per platform and the rule for choosing a before/after pair (compare every candidate with `compare_pair`, look at the previews, use the obvious one; a still pair of the same pose beats moving clips for a reveal).

V2.4c: before/after reveal and picture cards.

- `reveal` segment: a full-frame before picture that a divider line sweeps away (`left_to_right`, `right_to_left` or `top_to_bottom`) to show the after picture. Each side is a clip (`in`, `focus`, `crop`, `speed`, `gamma`, played together from their own `in`) or a still image, cover-cropped around one shared `focus` so faces line up. `hold_before` (0.8 s), `sweep` (0.9 s, eased), `labels` that follow their side (the before label goes as the divider passes it), `label_style`, `divider` (`width`, `color`, `shadow`, or `null` for a plain wipe), `motion: zoom_in` (both sides zoom in sync) and `back_and_forth` (across, back to the middle, across again). Exact frame count, same colour handling as clips; the divider position is a closed formula checked to the pixel by the tests. `qc_run` returns a frame mid-sweep and one on the after picture.
- `image` overlays become picture cards: `animation` (`pop`, `slide_left`, `slide_right`, `slide_up`, `fade`; 250 ms entry, optional `exit`), `frame` (`radius`, `border_width`, `border_color`, `shadow`) and `rotate`. Every animation state is a deterministic cached picture. A plain image overlay renders exactly as before.
- `image` segments accept `focus` for a `cover` crop (centred by default, so existing caches stay valid).
- Validation: `split_same_source` (the two panels of a split or the two sides of a reveal are one file at overlapping moments: the difference will not be visible), `reveal_orientation` (sides of different orientation families), `reveal_after_short` (under 0.6 s on the after picture); a reveal side shares `source_duration`, `panel_crop`, `digital_zoom`, `split_crop` and `text_overflow` with split panels. `storyboard_snap` never shortens a reveal below `hold_before + sweep`.
- The `before_after` recipe prefers a reveal with two real pictures, then a split; the docs gain a "Before and after" section (what makes a pair credible and visible, when a card is better, compliance reminders).

V2.4b fixes.

- Emphasis on numbers and prices: `emphasis` matches `65,90 €`, `65,90`, `65.90€` and `-20%` however they are spaced, accepts a multi-word entry, and `*...*` markers work around any text (numbers, currency signs, `%`, spaces) without printing the stars. `storyboard_validate` and `caption_preview` warn `emphasis_unmatched` when an entry matches nothing.
- `storyboard_snap` gives the new version its own `name` and `output.path` (`save_as`, else `<name>_snapped`), so rendering it never overwrites the original; `storyboard_patch` moves a default `output.path` along with a renamed `name`.
- A `badge` sticker with no `color` is the brand `visual.accent_color`, else neutral dark `#111827` with white text.
- One emotion vocabulary (`vocab.EMOTION_VALUES`) for the breakdown, library annotate, `library_search` (an unknown filter value is now refused) and segments; the docs list all 13 values.
- Text box detection ignores boxes over 60 % of the frame area (or over 90 % of both its sides) and boxes seen on fewer than 2 sampled frames, so a white wall is no longer reported as text.

Visual grammar.

- `split` segment: two panels (clips or images) side by side or top and bottom, cover-cropped around `focus`, with `gap`, `gap_color` and optional `labels` (`label`, `pill` or `outline` style). A clip that ends before the segment holds its last frame. Rendered as an ordinary mezzanine with an exact frame count.
- `video` overlay (picture-in-picture): `rect`, `rounded` or `circle` shape, `border`, `shadow`, exact timing, optional `audio` (needs `audio.clips`, refused next to a voice-over or another voice).
- `sticker` overlay: `arrow`, `circle`, `underline`, `badge`, `emoji` and `image`, with `pop` and `draw` (300 ms) animations rendered as deterministic picture sequences.
- `progress` overlay: a bar filling across the ad (or `start` to `end`), full on the last frame.
- `punch` on clip segments: instant zooms (no ramp) at `at` for `hold`, centred on `center` or `focus`; only above 1.25 counts for `digital_zoom`.
- Validation: `split_layout`, `panel_crop`, `inset_out_of_frame`, `inset_audio`, `sticker_out_of_frame`, `sticker_render` errors; `split_crop`, `caption_covered_by_inset`, `inset_source_short`, `inset_audio_replaces`, `inset_audio_silent` warnings. `qc_run` crops every inset and shows a frame of each split screen. The `before_after` recipe prefers a split when both pictures are true.

Sound design.

- A pack of about 18 sound effects, built on demand into the project's `sfx/` folder by `adanvil sfx-fetch` or the `sfx_list` tool (`fetch=true`): whooshes, swoosh, swipe, pop, click, tap, ding, chime, notification, cash register, camera shutter, record scratch, typing, risers of 1 s and 2 s, impact, boom. Each is downloaded from a pinned CC0 source (SHA-1 verified) or synthesised in code (deterministic), trimmed and set to one level (loudest 400 ms at -16 LUFS, peak under -1.5 dBFS). Sources and licences: `examples/sfx/README.md`, `THIRD_PARTY.md` and `sfx/LICENCES.txt`. No sound is committed.
- `sfx` overlay (`source`, `at`, `volume`): mixed over the finished bed (voice, clips or music), with a limiter that keeps the true peak under -1 dB, never past the end of the ad.
- `audio.sfx_auto` (`cuts`, `captions`, `every`, `min_gap`, `volume`): sounds at each cut and at each written caption, expanded to explicit sounds listed in `summary.sfx` of the validation and in the `storyboard_save` result.
- `music_beats` (beat grid of a track as it plays, within a frame on a steady beat) and `storyboard_snap` (cuts moved onto the beats within `max_shift`, keeping `in` points, order, total duration and a 0.5 s minimum; captions and sounds on a moved cut follow). Validation reports an info `beat_alignment`. 33 tools.
- Validation: errors `sfx_missing`, `sfx_not_audio`, `sfx_after_end`, warning `sfx_auto_empty`; the report gains an `info` list.

Animated captions.

- `animation` on `caption`, `voice_captions` and `speech_captions`: `karaoke` (the whole chunk, the spoken word lit), `word` (one or two big words at a time) and `pop` (a 120 ms scale-in, optional `bounce`); default `none`, so old storyboards render as before. The lit word takes `highlight` (default the brand accent, else `#FFD400`), `highlight_scale` and a rounded `highlight_box`. Layout is computed once per chunk, so the text never moves between states; states are cropped pictures placed by the overlay filter, so the render costs about the same as static captions.
- `emphasis` (case and accent insensitive) and `*word*` markers in typed captions and in the `show` text of voice lines draw keywords in the highlight colour, with or without an animation.
- Generated chunks respect punctuation: never across a sentence end, stop at a comma once they hold two words, never start with a mark.
- `caption_preview` draws any state of an animated caption (`chunk_index` and `word_index` in the overlay, or the time `t`) and takes `storyboard` (path or object).
- New tools `storyboard_save` (writes `storyboards/<name>.json`) and `storyboard_patch` (a list of `{path, value}` edits) so a remote agent sends a storyboard once; `storyboard_validate`, `render_start` and `caption_preview` take its path. 30 tools.
- Validation reports every bad `shot_type`, `scale`, `emotion` and `role` at once, each with the allowed values.
- The final encode reads its filter graph from a file and its inputs relative to the project, so hundreds of caption pictures no longer overflow a Windows command line.

Talking-head UGC.

- Speech in the library: `library_index` transcribes clips with sound (word timestamps, one model load per job, `transcribe` is `"auto"`, `true` or `false`); the catalogue keeps a short `speech` summary and the words live in `cache/speech/`. No transcriber is a status, never an error.
- New tools `speech_view` (compact phrases with flags, fillers, restarts, clean cut segments; per-word data with `words=true`, pages of 60 phrases), `speech_takes` (the best take of each line across recordings) and `speech_refine` (transcribe the chosen clips again with a stronger model and the brand vocabulary, as a detached job). 28 tools.
- `audio.clips` keeps the original sound of the clip segments, aligned with the cuts and normalised like a voice, with music under it; `audio_from` lays B-roll over the creator's voice.
- `speech_captions` overlay: captions built from the speech on the timeline (jump cuts and speed aware); `caption_preview` accepts it.
- Validation: `cut_mid_word` and `banned_word_spoken` warnings, `speech_missing` warning, `speech_captions` and `audio_from` errors.
- Recipe `ugc_talking` (9 recipes). The brand profile gains an optional `speech` section (filler lists, `vocabulary`).
- Better words: the brand name, the product name and `speech.vocabulary` are the transcriber's initial prompt; `speech_refine` redoes chosen clips with a stronger model (default `medium`); `speech_captions.corrections` shows a misheard word right (close spellings only, listed as a `speech_corrections` warning for review).
- Faster, progressive indexing: the speech of each file is saved as soon as that file is transcribed (so `library_search(text=...)` works during the job), the indexing model defaults to `base`, and audio is extracted in parallel.
- Audio files (`.m4a`, `.wav`, `.mp3`, `.aac`) are indexed as `audio` entries with their speech and work as `audio_from` sources. A video with silent sound gets the speech status `silent_audio` (renamed from `not attempted: audio is near-silent`) and `separate_audio_candidates`.
- Compact outputs: `speech_view` returns phrases and clean segments by default (`words`, `max_phrases`, `first_phrase`); `brand_onboard` returns a short site summary and saves the full facts in `brand/site_facts.json`.
- Recipe slots have a `role` (a segment `role` value) and a `purpose` (the old prose, renamed): the `result` slot is a `payoff`.
- Validation: `cut_mid_word` no longer fires when the sound goes on across the cut (same source, within one frame); new warnings `caption_on_end_card` and `cta_text_missing`.

## 0.1.0 (2026-10-02, not published)

First public version (alpha).

- MCP server with 25 tools and a matching command line (`adanvil`).
- Reference decoding (shots, pacing, burnt-in text, speech) and a shot-by-shot breakdown saved next to the reference.
- Footage library: indexing with parallel workers, contact sheets, mosaics, annotations (emotion, people, product version), search.
- 8 recipes: text hook, hook plus product, before and after, customer diary, comment reply, meme text, voice-over ad, long remake.
- 7 caption styles with emoji, caption preview over a real frame, end cards from a product photo.
- Text removal on reference footage, HDR to SDR tone-mapping, music profile and loudness control.
- Optional open-source voice-over (Chatterbox) with a pronunciation check and captions synced to the voice.
- Deterministic rendering in detached background jobs, with automatic QC and control images.
- Demo pack (CC0 Wikimedia Commons clips, fictional brand) in `examples/demo/`.
