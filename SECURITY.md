# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub: **Security** tab of [maxbuildog/adanvil](https://github.com/maxbuildog/adanvil/security), then **Report a vulnerability**. Do not open a public issue. You will get an answer within 7 days.

## Supported versions

Only the latest release receives fixes. Versions before 0.3.1 have known issues (see `CHANGELOG.md`): please upgrade.

## What AdAnvil protects against

AdAnvil is a local MCP server that runs with your rights. The agent that drives it reads content you do not control (reference ads, brand websites, transcripts, project folders shared by others), so AdAnvil treats every tool argument and every project file as untrusted:

- **Projects stay projects.** Tools only read and write inside a project folder, never through `..` or hidden names (`.ssh`, `.config`), and a write never follows a link planted in a project (links you make on purpose, such as `footage/` on another drive, are still read). Your home folder, a drive root and network paths cannot be used as a project. Set `ADANVIL_PROJECTS_DIR` to keep every project under one folder (the Claude Desktop extension and the Docker image do it for you).
- **No local network.** Website onboarding and downloads only reach public `http`/`https` addresses, with size and time limits. `ADANVIL_ALLOW_PRIVATE_URLS=1` lifts the address rule, for example to onboard a site you run on `localhost`.
- **No foreign code.** Tools never run a program or load a model chosen by the agent: the helper environments install pinned versions, and Whisper model and voice names come from fixed lists.
- **ffmpeg reads local media files only** (`-protocol_whitelist file`, and a playlist disguised as a video is refused), never your terminal or the MCP stream.

Out of scope: an agent you allowed to run shell commands or edit files with its own tools can do so without AdAnvil.
