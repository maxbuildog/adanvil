# Claude Desktop extension (MCPB)

`manifest.json` describes the extension (server type `uv`: Claude Desktop runs `server/main.py`
with `uv` and installs the bundled package). `server/main.py` checks ffmpeg, then starts
`adanvil-mcp`. The version, the server block, the user settings and the tools list in
`manifest.json` are generated:

    uv run python scripts/build_mcpb.py            # writes dist/adanvil-<version>.mcpb
    uv run python scripts/build_mcpb.py --check    # fails when manifest.json is out of date

The build copies `src/adanvil`, `pyproject.toml`, `README.md` and `LICENSE` into the bundle.
