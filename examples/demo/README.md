# Reproducible demo pack

The demo creates a ready-to-index Ember & Oak project: four vertical or crop-ready Commons clips, a product still, a fictional brand profile, a portrait end card, and a deterministic 24-second synthetic music loop. The repository contains no media; the five pack files are fetched when the command runs.

## Run

From an AdAnvil checkout (after `uv sync`). Prefix each command with `uv run`, or activate `.venv` first:

```text
uv run python examples/demo/fetch_demo.py DEMO_DIR
```

Use `--music none` to omit the generated loop, or `--force` to fetch the verified pack files again. The generated `music/demo_loop.m4a` is only a starting point: it may be replaced with any track for which you have the necessary licence.

| Role | Project file | Licence | Commons page |
| --- | --- | --- | --- |
| Hands | `footage/hands_toast.webm` | CC0 | [Buena Vista Winery](https://commons.wikimedia.org/wiki/File:Buena_Vista_Winery_-_September_2024_-_Sarah_Stierch_02.webm) |
| Product | `footage/candle_flameless.webm` | CC0 | [Flameless candle 1](https://commons.wikimedia.org/wiki/File:Flameless_candle_1.webm) |
| Product | `footage/candle_pink.webm` | CC0 | [Pink candle burning](https://commons.wikimedia.org/wiki/File:Pink_candle_burning.webm) |
| Lifestyle | `footage/waterfall.webm` | CC0 | [Watagataki Falls](https://commons.wikimedia.org/wiki/File:Watagataki_Falls_(Video).webm) |
| Product image | `brand/jar.jpg` | CC0 | [Cosmetic jar](https://commons.wikimedia.org/wiki/File:Cosmetic_jar_MET_26.7.1442_EGDP012736.jpg) |

Everything fetched for this pack is marked CC0 (or public domain), so an advertisement made solely from these assets has no attribution duty. Attribution is still appreciated. The synthetic music is generated locally and is not a downloaded Commons asset.

## Render a first ad

Index the footage, inspect the `text_hook` recipe, then write `storyboard.json` using its slots and the files above:

```text
uv run adanvil index DEMO_DIR --source footage
uv run adanvil recipes text_hook
uv run adanvil validate DEMO_DIR storyboard.json --recipe text_hook
uv run adanvil render DEMO_DIR storyboard.json
```

The CLI writes the rendered video under `DEMO_DIR/renders/`. Use `adanvil recipes text_hook` as the source of truth for the current recipe checklist before filling the storyboard.
