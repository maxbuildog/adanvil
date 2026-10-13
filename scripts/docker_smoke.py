"""Smoke test of the Docker image: tools/list, then one tiny render of the demo pack.

    python scripts/docker_smoke.py IMAGE DEMO_DIR

DEMO_DIR is made with ``python examples/demo/fetch_demo.py DEMO_DIR``. It is mounted at
/projects, so the project path seen by the server is /projects.
"""

from __future__ import annotations

import json
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


async def main(image: str, demo: str) -> int:
    params = StdioServerParameters(
        command="docker", args=["run", "-i", "--rm", "-v", f"{demo}:/projects", image]
    )
    async with stdio_client(params) as streams, ClientSession(*streams) as session:
        await session.initialize()
        tools = [tool.name for tool in (await session.list_tools()).tools]
        print("tools/list:", len(tools), "tools")
        board = {
            "version": 1,
            "name": "docker_smoke",
            "segments": [
                {"type": "clip", "source": "footage/waterfall.webm", "in": 0, "out": 1.0}
            ],
        }
        args = {"project_dir": "/projects", "storyboard": board}
        valid = json.loads(_text(await session.call_tool("storyboard_validate", args)))
        print("validate ok:", valid["ok"])
        started = await session.call_tool("render_start", {**args, "quality": "preview"})
        job = json.loads(_text(started))
        done = await session.call_tool(
            "job_status", {"job_id": job["job_id"], "wait_seconds": 100, "project_dir": "/projects"}
        )
        status = json.loads(_text(done))
        print("render:", status["status"], status.get("report", {}).get("path"))
        if status["status"] != "done":
            print(status)
            return 1
    print("tools:", len(tools))
    return 0


if __name__ == "__main__":
    sys.exit(anyio.run(main, sys.argv[1], sys.argv[2]))
