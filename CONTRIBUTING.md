# Contributing to AdAnvil

Thanks for helping. AdAnvil is alpha software: small, focused pull requests are easiest to review.

## Dev setup

You need Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and ffmpeg 6.1 or newer (built with `libzimg`) on your `PATH`.

```bash
git clone https://github.com/maxbuildog/adanvil
cd adanvil
uv sync
ffmpeg -version          # must print 6.1 or newer
uv run pytest            # the whole suite
uv run ruff check .      # lint, must be clean
```

The full test suite renders real video and takes several minutes (10 to 15 on a small machine). While you work, run only the files you touched, for example `uv run pytest tests/test_text.py -q`. CI runs the whole suite with ruff on every push and pull request.

## Tests

- Test media is generated, never committed: `tests/fixtures/make.py` builds clips and images with ffmpeg and Pillow into `tests/fixtures/out/` (ignored by Git). Do not add binary fixtures to the repository.
- Keep tests brand-neutral and niche-neutral. Use fictional brands (the demo uses Ember & Oak) from different niches (coffee, running shoes, pet food, software...) and neutral prices, never a real brand or real figures.
- Some tests depend on exact names (for example a short brand name that must be reported as unmatched when a longer word contains it). If you rename a fixture word, check that the test still tests the same thing.
- The voice tests use a fake engine. Nothing in the suite downloads models or reaches the network.

## Files that must stay in sync

- `docs/EDITING_GUIDE.md` and its byte-for-byte copy `src/adanvil/assets/EDITING_GUIDE.md` (served by the `editing_guide` tool). A test fails if they differ. Edit the first, then copy it over the second.
- `AGENTS.md` and `skill/SKILL.md` carry the same agent guide. `SKILL.md` only adds a front-matter block and drops the connection section. Change both in the same pull request.
- The README tools table and the tool list in `src/adanvil/server.py`.

## How to add a recipe

A recipe is a YAML file in `src/adanvil/storyboard/recipes/`. Copy a similar one (`text_hook.yaml` is the simplest) and keep the same keys: `id` (the file name), `name`, `summary`, `when_to_use`, `duration`, `audio`, `captions`, `structure` (slots), `rules`, and an `example` storyboard.

- `rules` may only use checks the validator knows (see the `allowed` set in `tests/test_storyboard_rules.py` and `src/adanvil/storyboard/validate.py`).
- Each slot of `structure` has a `role` (one of the segment `role` values: `hook`, `problem`, `payoff`, `cta`...; a slot named like a role gets it by default, any other slot must say which role it plays) and a `purpose` (what it is for, in words). `tests/test_ugc_iteration.py` checks every recipe.
- The `example` storyboard must validate against the recipe.
- Add or update a test. `tests/test_storyboard_rules.py` checks the recipe count and that every example validates: update the count, and add a test for any behaviour that is specific to your recipe.
- Mention the recipe in `AGENTS.md`, `skill/SKILL.md` and the README (recipe list and the "choose a recipe" guidance).

## How to add a caption style

Caption styles live in `src/adanvil/text/styles.py`.

1. Add the name to the `StyleName` literal.
2. Add its defaults to the `_DEFAULTS` table and its drawing branch to the renderer (see how `box` or `label` are drawn).
3. Check the fitting rules (`_fit_for_style`): text must never render wider than 960 px.
4. Add the style to the parametrized `test_all_caption_styles_render_inside_canvas` in `tests/test_text.py`, plus a test for anything specific (padding, colours, emoji).
5. Document it in `docs/EDITING_GUIDE.md` (and the packaged copy), `AGENTS.md`, `skill/SKILL.md` and the README.

## Style

- Code: ruff with the settings in `pyproject.toml` (line length 100). Typed arguments, small pure functions, ffmpeg only through `adanvil.ffmpeg.run()` (see `docs/ARCHITECTURE.md`).
- No new dependency without a discussion in an issue first.
- Docs: plain English, concrete, no hype. Do not use em dashes or en dashes.

## Pull request checklist

- [ ] `uv run ruff check .` is clean.
- [ ] `uv run pytest` passes (or you ran the relevant files and say so).
- [ ] New behaviour has a test, and the tests are brand-neutral.
- [ ] `docs/EDITING_GUIDE.md` and `src/adanvil/assets/EDITING_GUIDE.md` are identical.
- [ ] `AGENTS.md` and `skill/SKILL.md` are in sync.
- [ ] README and `CHANGELOG.md` are updated when behaviour, tools or settings change.
- [ ] No media, secrets, local scratch folders or third-party material without a licence were added (see `THIRD_PARTY.md`).

## Reporting bugs

Use the bug report template. Include your OS, `ffmpeg -version`, the MCP client and its model, the storyboard excerpt that fails and the `error` returned by `job_status`.
