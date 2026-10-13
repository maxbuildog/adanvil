"""Build the Claude Desktop extension: dist/adanvil-<version>.mcpb.

    python scripts/build_mcpb.py           # sync manifest.json, then write the .mcpb
    python scripts/build_mcpb.py --check   # exit 1 when manifest.json is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "packaging" / "mcpb"
MANIFEST = PKG / "manifest.json"

USER_CONFIG = {
    "projects_folder": {
        "type": "directory",
        "title": "Projects folder",
        "description": "Folder where AdAnvil keeps your ad projects (footage, brand, renders).",
        "required": False,
        "default": "${DOCUMENTS}/AdAnvil",
    },
    "freesound_api_key": {
        "type": "string",
        "title": "Freesound API key (optional)",
        "description": "Adds Freesound to the music search. Leave empty to use Openverse only.",
        "sensitive": True,
        "required": False,
    },
}
SERVER = {
    "type": "uv",
    "entry_point": "server/main.py",
    "mcp_config": {
        "command": "uv",
        "args": ["run", "--directory", "${__dirname}", "python", "server/main.py"],
        "env": {
            "ADANVIL_PROJECTS_DIR": "${user_config.projects_folder}",
            "FREESOUND_API_KEY": "${user_config.freesound_api_key}",
        },
    },
}


def package_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def _first_line(text: str) -> str:
    line = " ".join(text.strip().split("\n\n")[0].split()).replace("``", "")
    return line if len(line) <= 200 else line[:197].rstrip() + "..."


def server_tools() -> list[dict[str, str]]:
    import anyio
    from mcp import Client

    sys.path.insert(0, str(ROOT / "src"))
    from adanvil.server import build_server

    async def listing() -> list[dict[str, str]]:
        async with Client(build_server()) as client:
            result = await client.list_tools()
        tools = [
            {"name": tool.name, "description": _first_line(tool.description or "")}
            for tool in result.tools
        ]
        return sorted(tools, key=lambda tool: tool["name"])

    return anyio.run(listing)


def expected_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["version"] = package_version()
    manifest["description"] = (
        "Agentic video ad editor: turns your brand's footage into ready-to-run 9:16 ads. "
        "Needs ffmpeg 6.1 or newer installed on your computer."
    )
    manifest["server"] = SERVER
    manifest["tools"] = server_tools()
    manifest["tools_generated"] = False
    manifest["user_config"] = USER_CONFIG
    return manifest


def render(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def build(out_dir: Path) -> Path:
    MANIFEST.write_text(render(expected_manifest()), encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"adanvil-{manifest['version']}.mcpb"
    files: list[tuple[Path, str]] = [
        (MANIFEST, "manifest.json"),
        (PKG / "icon.png", "icon.png"),
        (PKG / "server" / "main.py", "server/main.py"),
        (PKG / ".mcpbignore", ".mcpbignore"),
        (ROOT / "pyproject.toml", "pyproject.toml"),
        (ROOT / "README.md", "README.md"),
        (ROOT / "LICENSE", "LICENSE"),
    ]
    for path in sorted((ROOT / "src" / "adanvil").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            files.append((path, path.relative_to(ROOT).as_posix()))
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        for source, name in files:
            bundle.write(source, name)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only verify manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    if args.check:
        if MANIFEST.read_text(encoding="utf-8") != render(expected_manifest()):
            print("packaging/mcpb/manifest.json is out of date: run scripts/build_mcpb.py")
            return 1
        return 0
    print(build(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
