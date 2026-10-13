"""V2.10a: the Claude Desktop extension manifest and the Docker image."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import anyio
import pytest
from mcp import Client

import adanvil
from adanvil.server import build_server

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "packaging" / "mcpb" / "manifest.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _server_tools() -> set[str]:
    async def listing() -> set[str]:
        async with Client(build_server()) as client:
            return {tool.name for tool in (await client.list_tools()).tools}

    return anyio.run(listing)


def test_manifest_matches_package_and_server() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert manifest["version"] == project["version"] == adanvil.__version__
    assert manifest["name"] == "adanvil"
    assert manifest["author"]["name"] == "maxbuildog"
    assert manifest["license"] == "MIT"
    assert manifest["repository"]["url"].startswith("https://github.com/maxbuildog/adanvil")
    assert (MANIFEST.parent / manifest["icon"]).is_file()
    assert (MANIFEST.parent / "server" / "main.py").is_file()
    assert {tool["name"] for tool in manifest["tools"]} == _server_tools()
    assert all(tool["description"] for tool in manifest["tools"])
    config = manifest["user_config"]
    assert config["projects_folder"]["type"] == "directory"
    assert config["freesound_api_key"]["sensitive"] is True
    assert config["freesound_api_key"]["required"] is False


def test_manifest_is_in_sync_with_build_script() -> None:
    build = _load(ROOT / "scripts" / "build_mcpb.py", "build_mcpb")
    assert MANIFEST.read_text(encoding="utf-8") == build.render(build.expected_manifest())


def test_mcpb_file_is_built(tmp_path: Path) -> None:
    build = _load(ROOT / "scripts" / "build_mcpb.py", "build_mcpb")
    original = MANIFEST.read_text(encoding="utf-8")
    target = build.build(tmp_path)
    assert MANIFEST.read_text(encoding="utf-8") == original
    assert target.name == f"adanvil-{adanvil.__version__}.mcpb"
    with zipfile.ZipFile(target) as bundle:
        names = set(bundle.namelist())
        assert {"manifest.json", "icon.png", "server/main.py", "pyproject.toml"} <= names
        assert "src/adanvil/server.py" in names
        assert not any("__pycache__" in name for name in names)
        assert json.loads(bundle.read("manifest.json"))["version"] == adanvil.__version__


def test_launcher_explains_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load(ROOT / "packaging" / "mcpb" / "server" / "main.py", "mcpb_launcher")
    monkeypatch.setenv("ADANVIL_FFMPEG", str(ROOT / "nope" / "ffmpeg"))
    monkeypatch.setenv("ADANVIL_FFPROBE", str(ROOT / "nope" / "ffprobe"))
    message = launcher.check_ffmpeg()
    assert message and "ffmpeg" in message and "Install it with" in message
    monkeypatch.delenv("ADANVIL_FFMPEG")
    monkeypatch.delenv("ADANVIL_FFPROBE")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/" + name)
    assert launcher.check_ffmpeg() is None


def test_dockerfile_basics() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM ubuntu:24.04" in text
    assert "USER adanvil" in text
    assert 'ENTRYPOINT ["adanvil-mcp"]' in text
    assert "VOLUME /projects" in text
    assert (ROOT / ".dockerignore").is_file()


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")
def test_docker_image_lists_tools(tmp_path: Path) -> None:
    probe = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
    if probe.returncode != 0:
        pytest.skip("docker daemon is not usable")
    subprocess.run(
        ["docker", "build", "-t", "adanvil-test", str(ROOT)], check=True, timeout=1500
    )

    async def listing() -> set[str]:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command="docker",
            args=["run", "-i", "--rm", "-v", f"{tmp_path}:/projects", "adanvil-test"],
        )
        async with stdio_client(params) as streams:
            from mcp import ClientSession

            async with ClientSession(*streams) as session:
                await session.initialize()
                return {tool.name for tool in (await session.list_tools()).tools}

    assert anyio.run(listing) == _server_tools()
