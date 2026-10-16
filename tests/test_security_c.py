"""Security 0.3.1, agent C: downloads, helper environments, model names and planted links.

Every test reproduces an attack from the audit and shows it is refused, next to a test that the
normal use still works. The local servers listen on the loopback address, which the download
guard refuses by default: the tests that need to reach one opt in with
``ADANVIL_ALLOW_PRIVATE_URLS=1`` (the documented development switch), the others prove the
refusal by checking that the server saw no request at all.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import wave
from collections.abc import Callable, Iterator
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from adanvil import music, musicgen, netguard, sfx
from adanvil.brand import default_brand, save_brand
from adanvil.onboard import analyze_site, onboard
from adanvil.onboard import site as site_module
from adanvil.project import init_project
from adanvil.speech import transcribe
from adanvil.text import emoji as emoji_module
from adanvil.voice import engines as voice_engines
from adanvil.voice import generate as voice_generate
from adanvil.voice import setup as voice_setup_module
from adanvil.voice import worker as voice_worker
from adanvil.voice.engines import FakeEngine, LineJob, WorkerEngine
from adanvil.voice.generate import VoiceSpec, generate_voice

ALLOW = "ADANVIL_ALLOW_PRIVATE_URLS"


@pytest.fixture(autouse=True)
def _guard_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard is on unless a test switches it off (a developer machine may set the variable)."""

    monkeypatch.delenv(ALLOW, raising=False)
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def allow_private(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW, "1")


# ------------------------------------------------------------------------------ test servers


@contextlib.contextmanager
def serve(
    respond: Callable[[BaseHTTPRequestHandler], None], *, protocol: str = "HTTP/1.0"
) -> Iterator[tuple[str, list[tuple[str, str | None]]]]:
    """A local HTTP server; yields its base URL and the list of ``(path, Authorization)`` seen."""

    seen: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = protocol

        def do_GET(self) -> None:  # noqa: N802
            seen.append((self.path, self.headers.get("Authorization")))
            respond(self)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def reply(
    handler: BaseHTTPRequestHandler, body: bytes = b"hello", kind: str = "text/plain"
) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", kind)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def hello(handler: BaseHTTPRequestHandler) -> None:
    reply(handler)


# --------------------------------------------------------------------------------------- C1


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file:///C:/Windows/win.ini",
        "ftp://example.com/x",
        "data:text/plain;base64,aGVsbG8=",
        "gopher://example.com/",
        "/etc/passwd",
        "//example.com/x",
        "",
    ],
)
def test_only_http_and_https_are_fetched(url: str) -> None:
    with pytest.raises(netguard.NetguardError, match="http"):
        netguard.fetch(url, timeout=2, max_bytes=100)


def test_a_url_with_credentials_is_refused() -> None:
    with pytest.raises(netguard.NetguardError, match="user name"):
        netguard.fetch("http://user:secret@example.com/", timeout=2, max_bytes=100)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "127.255.255.254",
        "10.0.0.1",
        "172.16.5.4",
        "172.31.255.255",
        "192.168.1.10",
        "169.254.169.254",  # the cloud metadata service
        "100.64.0.1",  # carrier-grade NAT
        "0.0.0.0",
        "224.0.0.251",
        "240.0.0.1",
        "255.255.255.255",
        "192.0.2.7",
        "::1",
        "::",
        "fe80::1",
        "fe80::1%eth0",
        "fc00::1",
        "fd12:3456::1",
        "ff02::1",
        "::ffff:127.0.0.1",  # IPv4-mapped forms of the above
        "::ffff:10.1.2.3",
        "::ffff:169.254.169.254",
        "64:ff9b::7f00:1",  # NAT64 of 127.0.0.1
        "2002:7f00:1::",  # 6to4 of 127.0.0.1
        "not an address",
    ],
)
def test_non_global_addresses_are_refused(address: str) -> None:
    assert netguard.is_public_address(address) is False


@pytest.mark.parametrize(
    "address", ["8.8.8.8", "93.184.216.34", "1.1.1.1", "2606:4700:4700::1111", "::ffff:8.8.8.8"]
)
def test_global_addresses_are_allowed(address: str) -> None:
    assert netguard.is_public_address(address) is True


def test_a_loopback_server_receives_nothing() -> None:
    with serve(hello) as (base, seen):
        with pytest.raises(netguard.NetguardError, match="not a public internet address"):
            netguard.fetch(base + "/admin?action=delete", timeout=2, max_bytes=100)
        assert seen == []


def test_other_spellings_of_loopback_are_refused_too() -> None:
    with serve(hello) as (base, seen):
        port = base.rsplit(":", 1)[1]
        for host in ("localhost", "2130706433", "0x7f.0.0.1", "127.1", "[::1]", "[::ffff:7f00:1]"):
            with pytest.raises(netguard.NetguardError):
                netguard.fetch(f"http://{host}:{port}/", timeout=2, max_bytes=100)
        assert seen == []


def test_the_development_switch_lifts_only_the_address_rule(allow_private: None) -> None:
    with serve(hello) as (base, seen):
        response = netguard.fetch(base + "/page", timeout=2, max_bytes=100)
        assert (response.status, response.body, response.url) == (200, b"hello", base + "/page")
        assert response.headers["Content-Type"] == "text/plain" and seen == [("/page", None)]
    with pytest.raises(netguard.NetguardError, match="http"):
        netguard.fetch("file:///etc/hostname", timeout=2, max_bytes=100)  # still refused


def test_the_name_is_checked_again_when_the_socket_connects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS rebinding: the name passed the check, then resolves to the loopback address."""

    with serve(hello) as (base, seen):
        port = base.rsplit(":", 1)[1]
        monkeypatch.setattr(netguard, "_check_url", lambda url, allow: urllib.parse.urlsplit(url))
        with pytest.raises(netguard.NetguardError, match="not a public internet address"):
            netguard.fetch(f"http://localhost:{port}/", timeout=2, max_bytes=100)
        assert seen == []


def test_the_peer_address_is_checked_after_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with serve(hello) as (base, seen):
        port = int(base.rsplit(":", 1)[1])
        monkeypatch.setattr(netguard, "_check_url", lambda url, allow: urllib.parse.urlsplit(url))
        monkeypatch.setattr(
            netguard,
            "_resolve",
            lambda host, number, allow: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, ("127.0.0.1", port))
            ],
        )
        with pytest.raises(netguard.NetguardError, match="connected to 127.0.0.1"):
            netguard.fetch(f"http://some.test:{port}/", timeout=2, max_bytes=100)
        assert seen == []  # connected, but not a byte of request was sent


def _fake_dns(monkeypatch: pytest.MonkeyPatch, table: dict[str, str]) -> None:
    real = socket.getaddrinfo

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any):
        if host in table:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (table[host], int(port or 0)))]
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


def test_every_redirect_hop_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public page that redirects to an internal address does not get there."""

    def redirect(handler: BaseHTTPRequestHandler) -> None:
        port = handler.server.server_address[1]  # type: ignore[attr-defined]
        handler.send_response(302)
        handler.send_header("Location", f"http://internal.test:{port}/secret")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    with serve(redirect) as (base, seen):
        port = int(base.rsplit(":", 1)[1])
        _fake_dns(monkeypatch, {"public.test": "93.184.216.34", "internal.test": "169.254.169.254"})
        # The "public" name reaches the local test server, whatever its address.
        monkeypatch.setattr(
            netguard,
            "_connect_checked",
            lambda host, number, timeout, allow: socket.create_connection(
                ("127.0.0.1", port), timeout
            ),
        )
        with pytest.raises(netguard.NetguardError, match="internal.test"):
            netguard.fetch(f"http://public.test:{port}/start", timeout=2, max_bytes=100)
        assert [path for path, _ in seen] == ["/start"]


def test_a_redirect_to_a_file_url_is_refused(allow_private: None) -> None:
    def redirect(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(302)
        handler.send_header("Location", "file:///etc/passwd")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    with serve(redirect) as (base, _seen), pytest.raises(urllib.error.HTTPError):
        netguard.fetch(base + "/", timeout=2, max_bytes=100)


def test_at_most_five_redirects(allow_private: None) -> None:
    def chain(handler: BaseHTTPRequestHandler) -> None:
        step = int(handler.path.rsplit("/", 1)[1])
        if step >= int(handler.path.split("/")[1]):
            reply(handler)
            return
        handler.send_response(302)
        handler.send_header("Location", f"/{handler.path.split('/')[1]}/{step + 1}")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    with serve(chain) as (base, seen):
        assert netguard.fetch(base + "/5/0", timeout=2, max_bytes=100).body == b"hello"  # 5 hops
        assert len(seen) == 6
    with serve(chain) as (base, seen):
        with pytest.raises(netguard.NetguardError, match="more than 5 redirects"):
            netguard.fetch(base + "/6/0", timeout=2, max_bytes=100)
        assert len(seen) == 6  # the sixth redirect was never followed


def test_the_final_url_follows_a_redirect(allow_private: None) -> None:
    def route(handler: BaseHTTPRequestHandler) -> None:
        if handler.path == "/old":
            handler.send_response(301)
            handler.send_header("Location", "/new")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
        else:
            reply(handler, b"moved")

    with serve(route) as (base, _seen):
        response = netguard.fetch(base + "/old", timeout=2, max_bytes=100)
        assert response.body == b"moved" and response.url == base + "/new"


def test_credentials_are_dropped_when_a_redirect_changes_host(allow_private: None) -> None:
    with serve(hello) as (other, other_seen):
        other_port = other.rsplit(":", 1)[1]

        def redirect(handler: BaseHTTPRequestHandler) -> None:
            if handler.path == "/b":
                reply(handler)
                return
            target = f"http://localhost:{other_port}/landing" if handler.path == "/away" else "/b"
            handler.send_response(302)
            handler.send_header("Location", target)
            handler.send_header("Content-Length", "0")
            handler.end_headers()

        with serve(redirect) as (base, seen):
            headers = {"Authorization": "Token secret"}
            netguard.fetch(base + "/away", timeout=2, max_bytes=100, headers=headers)
            assert seen == [("/away", "Token secret")]
            assert other_seen == [("/landing", None)]  # another host: no key
            seen.clear()
            netguard.fetch(base + "/same", timeout=2, max_bytes=100, headers=headers)
            assert seen == [("/same", "Token secret"), ("/b", "Token secret")]  # same host: kept


def test_a_proxy_of_the_environment_is_used_without_the_peer_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a proxy the peer is the proxy (here on the loopback address); the name still counts."""

    def proxy(handler: BaseHTTPRequestHandler) -> None:
        reply(handler, b"via proxy")

    with serve(proxy) as (base, seen):
        _fake_dns(monkeypatch, {"public.test": "93.184.216.34", "internal.test": "10.0.0.5"})
        monkeypatch.setenv("HTTP_PROXY", base)
        monkeypatch.setenv("http_proxy", base)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        response = netguard.fetch("http://public.test/page", timeout=2, max_bytes=100)
        assert response.body == b"via proxy"
        assert seen == [("http://public.test/page", None)]  # the proxy got the absolute URL
        with pytest.raises(netguard.NetguardError, match="internal.test"):
            netguard.fetch("http://internal.test/", timeout=2, max_bytes=100)
        assert len(seen) == 1


def test_a_body_over_the_limit_is_refused(allow_private: None) -> None:
    def declared(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Length", str(50 * 1024 * 1024))
        handler.end_headers()

    with serve(declared) as (base, _seen):
        with pytest.raises(netguard.NetguardError, match="more than the 1000 allowed"):
            netguard.fetch(base + "/", timeout=2, max_bytes=1000)

    def endless(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        chunk = b"x" * 4096
        with contextlib.suppress(OSError):
            for _ in range(10_000):
                handler.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")

    with serve(endless, protocol="HTTP/1.1") as (base, _seen):
        with pytest.raises(netguard.NetguardError, match="larger than 100000 bytes"):
            netguard.fetch(base + "/", timeout=2, max_bytes=100_000)


def test_a_download_cut_short_is_an_error(allow_private: None) -> None:
    def short(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Length", "1000")
        handler.end_headers()
        handler.wfile.write(b"only a little")

    with serve(short) as (base, _seen):
        with pytest.raises(netguard.NetguardError, match="stopped at 13 of 1000"):
            netguard.fetch(base + "/", timeout=2, max_bytes=10_000)


def test_a_server_that_drips_is_cut_at_the_deadline(allow_private: None) -> None:
    def drip(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Length", "100000")
        handler.end_headers()
        with contextlib.suppress(OSError):
            for _ in range(600):
                handler.wfile.write(b"x")
                handler.wfile.flush()
                time.sleep(0.1)

    with serve(drip) as (base, _seen):
        started = time.monotonic()
        with pytest.raises(netguard.NetguardError, match="took longer than 1 s"):
            netguard.fetch(base + "/", timeout=5, max_bytes=10_000_000, deadline=1.0)
        assert time.monotonic() - started < 4  # the socket timeout (5 s) never had to fire


def test_a_server_that_drips_its_headers_is_cut_too(allow_private: None) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    stop = threading.Event()

    def slowloris() -> None:
        connection, _ = listener.accept()
        with connection, contextlib.suppress(OSError):
            connection.recv(4096)
            for byte in b"HTTP/1.1 200 OK\r\nX-Padding: " + b"a" * 5000:
                if stop.is_set():
                    return
                connection.sendall(bytes([byte]))
                time.sleep(0.1)

    thread = threading.Thread(target=slowloris, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        with pytest.raises(netguard.NetguardError, match="took longer"):
            netguard.fetch(
                f"http://127.0.0.1:{listener.getsockname()[1]}/",
                timeout=5,
                max_bytes=1000,
                deadline=1.0,
            )
        assert time.monotonic() - started < 4
    finally:
        stop.set()
        listener.close()
        thread.join(timeout=3)


def test_fetch_to_file_streams_and_cleans_up(tmp_path: Path, allow_private: None) -> None:
    with serve(lambda handler: reply(handler, b"x" * 5000)) as (base, _seen):
        target = tmp_path / "a.bin"
        response = netguard.fetch_to_file(base + "/", target, timeout=2, max_bytes=10_000)
        assert response.size == 5000 and response.body == b"" and target.read_bytes() == b"x" * 5000
        refused = tmp_path / "b.bin"
        with pytest.raises(netguard.NetguardError):
            netguard.fetch_to_file(base + "/", refused, timeout=2, max_bytes=100)
        assert not refused.exists()  # no partial file left


def test_fetch_to_file_never_writes_through_an_existing_path(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    planted = tmp_path / "download.part"
    try:
        planted.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    with pytest.raises(FileExistsError):
        netguard.fetch_to_file("http://example.com/x", planted, timeout=2, max_bytes=10)
    assert victim.read_text(encoding="utf-8") == "precious" and planted.is_symlink()


# --------------------------------------------------------------------------------------- C2


def test_brand_onboard_does_not_call_a_local_service() -> None:
    """The audit's proof of concept: a server on 127.0.0.1 got ``GET /admin?action=delete``."""

    def admin(handler: BaseHTTPRequestHandler) -> None:
        reply(handler, b"<html><title>Admin panel</title></html>", "text/html")

    with serve(admin) as (base, seen):
        facts = analyze_site(base + "/admin?action=delete")
        assert seen == []
        assert facts.title == ""  # nothing of the service came back to the agent
        assert any("not a public internet address" in warning for warning in facts.warnings)


def test_brand_onboard_does_not_open_a_file_url(tmp_path: Path) -> None:
    secret = tmp_path / "secret.html"
    secret.write_text("<html><title>TOP SECRET MARKER</title></html>", encoding="utf-8")
    for url in (secret.as_uri(), "file:///definitely/not/there.html"):
        facts = analyze_site(url)
        assert facts.title == ""
        assert "TOP SECRET MARKER" not in facts.model_dump_json()
        assert any("only http and https" in warning for warning in facts.warnings)
        assert not any(
            "TypeError" in warning or "not there" in warning for warning in facts.warnings
        )


def test_brand_onboard_still_reads_a_site(tmp_path: Path, allow_private: None) -> None:
    page = (
        b"<html lang='fr'><head><title>Fernwood</title><style>a{color:#aa2244}</style></head>"
        b"<body><h1>Small batch coffee</h1><p>Only 12,50 EUR today</p></body></html>"
    )
    with serve(lambda handler: reply(handler, page, "text/html; charset=utf-8")) as (base, _s):
        result = onboard(tmp_path, url=base + "/", answers=None)
        assert result.facts is not None and result.facts.title == "Fernwood"
        assert "12,50 EUR" in result.facts.prices and result.facts.colors[0].hex == "#AA2244"
        assert (tmp_path / "brand" / "site_facts.json").is_file()


def test_a_page_over_three_megabytes_is_refused(allow_private: None) -> None:
    big = b"<html><title>big</title><body>" + b"x" * (3 * 1024 * 1024 + 10) + b"</body></html>"
    with serve(lambda handler: reply(handler, big, "text/html")) as (base, _seen):
        facts = analyze_site(base + "/")
        assert facts.title == "" and any(
            "more than the 3145728 allowed" in w for w in facts.warnings
        )


# The fixed patterns take about a tenth of a second on these inputs; the ones they replace took
# 49 s (price) and 2.5 s (Shopify currency) on 50 000 characters. The limit leaves room for a
# loaded machine and still tells the two apart.
LIMIT = 2.0


def _seconds(function: Callable[[], object]) -> float:
    started = time.perf_counter()
    function()
    return time.perf_counter() - started


@pytest.mark.parametrize(
    "hostile",
    [
        "1" + " 111" * 12_500,  # a long run of thousands groups, no currency
        "1" + ".111" * 12_500 + "x",
        "$1" + " 111" * 12_500 + "x",
        "9" * 50_000,
        "1 " * 25_000,
    ],
    ids=["groups", "dots", "dollar", "digits", "single-digits"],
)
def test_the_price_pattern_is_linear(hostile: str) -> None:
    assert _seconds(lambda: site_module._prices(hostile)) < LIMIT
    assert _seconds(lambda: site_module._currency_counts(hostile)) < LIMIT


def test_the_price_pattern_still_reads_real_prices() -> None:
    text = "From 1 234,50 EUR, was €1.299, or $19.99 / 5 USD. Pack 2 15,90 €. 12 345 678 €"
    found = site_module._prices(text)
    assert found == ["1 234,50 EUR", "€1.299", "$19.99", "5 USD", "15,90 €", "12 345 678 €"]


@pytest.mark.parametrize(
    "script",
    [
        "Shopify.currency = {" * 2500,
        "Shopify" + " " * 50_000,
        "Shopify.currency = {" + "x" * 50_000,
    ],
    ids=["repeated", "spaces", "no-active"],
)
def test_the_shopify_currency_pattern_is_linear(script: str) -> None:
    assert _seconds(lambda: site_module._shopify_currency([script])) < LIMIT
    real = 'Shopify.currency = {"active":"EUR","rate":"1.0"};'
    assert site_module._shopify_currency([real]) == "EUR"


def test_the_css_helpers_are_linear_on_hostile_text() -> None:
    hostile = [
        "font-family: " + "var(--a," * 6000,
        "font-family: a" + " " * 50_000 + "b",
        "font-family: " + "'a' " * 12_000,
        ";--a:" * 10_000,
    ]
    assert _seconds(lambda: site_module._fonts_from_css(hostile)) < LIMIT
    assert site_module._fonts_from_css(["a { font-family: var(--f, 'Fira Sans'), serif }"]) == [
        "Fira Sans"
    ]


def test_a_hostile_page_is_analysed_within_bounds() -> None:
    junk = "1" + " 111" * 12_500
    page = (
        f"<html><head><title>t</title><style>{'var(--a,' * 7000}</style>"
        f"<script>{'Shopify.currency = {' * 2500}</script></head>"
        f"<body><p>{junk}</p><h1>{junk}</h1></body></html>"
    )

    def fetch(url: str) -> tuple[int, str, str]:
        return 200, "text/html", page

    assert _seconds(lambda: analyze_site("https://example.test/", fetch=fetch)) < 5 * LIMIT


# --------------------------------------------------------------------------------------- C3


def _record(**changes: Any) -> dict[str, Any]:
    record = {
        "source": "openverse",
        "id": "abc",
        "url": "https://example.org/abc",
        "title": "Night Street",
        "author": "Ashot",
        "licence": "CC BY 4.0",
        "licence_code": "by",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution_required": True,
        "attribution_text": '"Night Street" by Ashot is licensed under CC BY 4.0.',
        "fetch_date": "2026-10-15",
        "file": "music/real.mp3",
    }
    record.update(changes)
    return record


FORGED_LINE = (
    'victim.mp3: "Fake" by Nobody, CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/), '
    "openverse 1 (https://example.org/1), fetched 2026-10-15. No attribution required."
)


@pytest.mark.parametrize(
    "separator", ["\n", "\r\n", "\r", "\x0b", "\x0c", "\x1c", "\x85", "\u2028", "\u2029", "\x00"]
)
def test_a_title_cannot_forge_the_line_of_another_file(tmp_path: Path, separator: str) -> None:
    (tmp_path / "music").mkdir()
    music.write_licence_record(
        tmp_path, "music/real.mp3", _record(title="Song" + separator + FORGED_LINE)
    )
    text = (tmp_path / "music" / "LICENCES.txt").read_text(encoding="utf-8")
    assert len(text.split("\n")) == 3  # header, one line, and the end of the file
    assert len(text.splitlines()) == 2
    assert music.read_ledger_record(tmp_path, "music/victim.mp3") is None
    real = music.read_ledger_record(tmp_path, "music/real.mp3")
    assert real is not None and real["title"].startswith("Song")


def test_author_and_attribution_are_cleaned_too(tmp_path: Path) -> None:
    (tmp_path / "music").mkdir()
    music.write_licence_record(
        tmp_path,
        "music/real.mp3",
        _record(author="Bo\n" + FORGED_LINE, attribution_text="credit\r\n" + FORGED_LINE),
    )
    text = (tmp_path / "music" / "LICENCES.txt").read_text(encoding="utf-8")
    assert len(text.splitlines()) == 2
    assert music.read_ledger_record(tmp_path, "music/victim.mp3") is None


def test_the_ledger_is_read_line_by_line_even_with_a_separator_in_it(tmp_path: Path) -> None:
    (tmp_path / "music").mkdir()
    ledger = tmp_path / "music" / "LICENCES.txt"
    ledger.write_text("real.mp3: Song\u2028" + FORGED_LINE + "\n", encoding="utf-8")
    assert music.read_ledger_record(tmp_path, "music/victim.mp3") is None


def test_the_licence_files_still_read_back(tmp_path: Path) -> None:
    (tmp_path / "music").mkdir()
    music.write_licence_record(tmp_path, "music/real.mp3", _record())
    sidecar = json.loads((tmp_path / "music" / "real.mp3.licence.json").read_text("utf-8"))
    assert sidecar["title"] == "Night Street"
    line = music.read_ledger_record(tmp_path, "music/real.mp3")
    assert line is not None and line["attribution_required"] is True
    assert line["attribution_text"] == '"Night Street" by Ashot is licensed under CC BY 4.0.'


def test_licence_files_are_not_written_through_planted_links(tmp_path: Path) -> None:
    (tmp_path / "music").mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    try:
        (tmp_path / "music" / "real.mp3.licence.json").symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    music.write_licence_record(tmp_path, "music/real.mp3", _record())
    assert victim.read_text(encoding="utf-8") == "precious"
    assert not (tmp_path / "music" / "real.mp3.licence.json").is_symlink()


def test_a_ledger_that_is_a_link_is_refused_and_never_read(tmp_path: Path) -> None:
    (tmp_path / "music").mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    try:
        (tmp_path / "music" / "LICENCES.txt").symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    with pytest.raises(music.MusicError, match="symbolic link"):
        music.write_licence_record(tmp_path, "music/real.mp3", _record())
    assert victim.read_text(encoding="utf-8") == "precious"


def _answer(body: Any) -> netguard.Response:
    return netguard.Response(200, Message(), json.dumps(body).encode(), "https://x.test/", 0)


def test_the_freesound_key_travels_in_a_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, dict[str, str]]] = []

    def fetch(url: str, **kwargs: Any) -> netguard.Response:
        sent.append((url, kwargs["headers"]))
        return _answer({"results": []})

    monkeypatch.setattr(music.netguard, "fetch", fetch)
    music.search_freesound("calm piano", "SECRETKEY", limit=5)
    music._freesound_detail("123", "SECRETKEY")
    assert len(sent) == 2
    for url, headers in sent:
        assert "SECRETKEY" not in url and "token" not in url
        assert headers["Authorization"] == "Token SECRETKEY"
        assert headers["User-Agent"].startswith("AdAnvil-music/")


def test_a_refused_music_url_is_a_music_error_without_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(url: str, **kwargs: Any) -> netguard.Response:
        raise netguard.NetguardError("nope")

    monkeypatch.setattr(music.netguard, "fetch", refuse)
    with pytest.raises(music.MusicError, match="refused https://freesound.org/apiv2/x/: nope"):
        music._http_json("https://freesound.org/apiv2/x/?query=a", {"Authorization": "Token k"})


def test_a_track_preview_url_cannot_point_at_a_local_service(tmp_path: Path) -> None:
    with serve(lambda handler: reply(handler, b"ID3" + b"x" * 100)) as (base, seen):
        target = tmp_path / "track.mp3"
        with pytest.raises(music.MusicError, match="refused"):
            music._http_download(base + "/track.mp3", target)
        assert seen == [] and not target.exists()
        assert not (tmp_path / "track.mp3.part").exists()


def test_music_downloads_keep_their_cap_and_their_part_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allow_private: None
) -> None:
    with serve(lambda handler: reply(handler, b"a" * 4000)) as (base, _seen):
        target = tmp_path / "track.mp3"
        assert music._http_download(base + "/t.mp3", target) == 4000
        assert target.read_bytes() == b"a" * 4000 and not (tmp_path / "track.mp3.part").exists()
        monkeypatch.setattr(music, "MAX_DOWNLOAD_BYTES", 1000)
        other = tmp_path / "other.mp3"
        with pytest.raises(music.MusicError, match="1000"):
            music._http_download(base + "/t.mp3", other)
        assert not other.exists() and not (tmp_path / "other.mp3.part").exists()


def test_a_planted_part_file_link_is_not_followed(tmp_path: Path, allow_private: None) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")
    try:
        (tmp_path / "track.mp3.part").symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    with serve(lambda handler: reply(handler, b"a" * 100)) as (base, _seen):
        music._http_download(base + "/t.mp3", tmp_path / "track.mp3")
    assert victim.read_text(encoding="utf-8") == "precious"


def test_the_cover_art_copy_has_a_name_nobody_can_plant(tmp_path: Path) -> None:
    """``strip_cover_art`` wrote ``<name>.clean.<ext>`` next to the track, a predictable name that
    ffmpeg opened through any link planted there."""

    from adanvil._bin import ffmpeg_path

    track = tmp_path / "art.mp3"
    cover = tmp_path / "cover.png"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(cover)
    subprocess.run(
        [ffmpeg_path(), "-nostdin", "-y", "-loglevel", "error",
         "-f", "lavfi", "-t", "2", "-i", "sine=frequency=330:sample_rate=44100",
         "-i", str(cover), "-map", "0:a", "-map", "1:v", "-c:a", "libmp3lame", "-c:v", "mjpeg",
         "-id3v2_version", "3", "-disposition:v", "attached_pic", str(track)],
        check=True,
    )  # fmt: skip
    victim = tmp_path / "victim.mp3"
    victim.write_bytes(b"precious")
    try:
        (tmp_path / "art.clean.mp3").symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    assert music.strip_cover_art(track) is True
    assert victim.read_bytes() == b"precious"


# --------------------------------------------------------------------------------------- C4


def test_the_sound_pack_download_refuses_a_local_service(tmp_path: Path) -> None:
    with serve(lambda handler: reply(handler, b"payload")) as (base, seen):
        with pytest.raises(sfx.SfxError, match="not a public internet address"):
            sfx.download_file(base + "/x.zip", "0" * 40, tmp_path / "x.zip")
        assert seen == [] and list(tmp_path.iterdir()) == []


def test_the_sound_pack_download_still_verifies_and_keeps_the_file(
    tmp_path: Path, allow_private: None
) -> None:
    import hashlib

    with serve(lambda handler: reply(handler, b"payload")) as (base, _seen):
        good = hashlib.sha1(b"payload").hexdigest()
        target = tmp_path / "x.zip"
        assert sfx.download_file(base + "/x.zip", good, target) == target
        assert target.read_bytes() == b"payload"
        with pytest.raises(sfx.SfxError, match="SHA-1 mismatch"):
            sfx.download_file(base + "/x.zip", "0" * 40, tmp_path / "y.zip")
        assert not (tmp_path / "y.zip").exists() and not (tmp_path / "y.zip.part").exists()


def test_the_sound_pack_is_not_written_through_planted_links(tmp_path: Path) -> None:
    folder = tmp_path / "sfx"
    folder.mkdir()
    victims = {}
    for name in ("pack.json", "LICENCES.txt", "pop.wav", ".pop.wav.part"):
        victim = tmp_path / f"victim-{name.strip('.')}"
        victim.write_text("precious", encoding="utf-8")
        victims[name] = victim
        try:
            (folder / name).symlink_to(victim)
        except (OSError, NotImplementedError):
            pytest.skip("symbolic links are not available")
    result = sfx.fetch_pack(tmp_path, offline=True, names=["pop"])
    assert result["built"] == ["pop"]
    assert all(v.read_text(encoding="utf-8") == "precious" for v in victims.values())
    assert (folder / "pop.wav").is_file() and not (folder / "pop.wav").is_symlink()
    assert not (folder / "pack.json").is_symlink() and not (folder / "LICENCES.txt").is_symlink()


def _png(path: Path, size: tuple[int, int] = (72, 72)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", size, (255, 0, 0, 255)).save(buffer, "PNG")
    return buffer.getvalue()


def test_an_emoji_download_that_is_not_a_png_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {"html": b"<html>captive portal</html>", "cut": _png(tmp_path)[:60], "big": b""}
    big = Image.new("RGB", (2000, 2000), (0, 0, 0))
    buffer = io.BytesIO()
    big.save(buffer, "PNG")
    payloads["big"] = buffer.getvalue()
    for name, data in payloads.items():

        def fake(url: str, destination: Path, _data: bytes = data, **_kw: Any) -> None:
            Path(destination).write_bytes(_data)

        monkeypatch.setattr(netguard, "fetch_to_file", fake)
        cache = tmp_path / name
        cache.mkdir()
        with pytest.raises(Exception):  # noqa: B017 (ValueError or an image error)
            emoji_module._download_emoji("https://x.test/1f600.png", cache / "1f600.png")
        assert list(cache.iterdir()) == [], name


def test_an_emoji_png_is_cached_after_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake(url: str, destination: Path, **kwargs: Any) -> None:
        calls.append(url)
        assert kwargs["max_bytes"] <= 1024 * 1024
        Path(destination).write_bytes(_png(tmp_path))

    monkeypatch.setattr(netguard, "fetch_to_file", fake)
    provider = emoji_module.TwemojiProvider(tmp_path / "cache")
    image = provider.get("\U0001f600", 36)
    assert image is not None and image.size == (36, 36)
    assert [p.name for p in provider.cache_dir.iterdir()] == ["1f600.png"]
    assert "jdecked/twemoji@" in calls[0]
    assert provider.get("\U0001f600", 36) is not None and len(calls) == 1  # cached


# A worker that writes a real file and then prints whatever result line the test wants.
_WORKER = """
import json, os, sys, wave
import numpy as np
request = json.load(open(sys.argv[1]))
t = np.arange(int(48000 * request["duration"])) / 48000
data = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
with wave.open(request["out"], "wb") as handle:
    handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(48000)
    handle.writeframes(data.tobytes())
mode = os.environ["ADANVIL_TEST_MODE"]
out = request["out"]
victim = os.environ.get("ADANVIL_TEST_VICTIM", "")
if mode == "outside":
    out = victim
elif mode == "link":
    os.remove(out)
    os.symlink(victim, out)
elif mode == "forged_first":
    print("ADANVIL_RESULT " + json.dumps({"ok": True, "out": victim, "seconds": 0}))
elif mode == "garbage_first":
    print("ADANVIL_RESULT {not json")
print("ADANVIL_RESULT " + json.dumps({"ok": True, "out": out, "seconds": 1.5}))
"""


def _music_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> tuple[Path, Path]:
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    (cache / "music_config.json").write_text(json.dumps({"python": sys.executable}), "utf-8")
    worker = tmp_path / "fake_worker.py"
    worker.write_text(_WORKER, encoding="utf-8")
    victim = tmp_path / "victim.wav"
    victim.write_bytes(b"precious")
    monkeypatch.setenv("ADANVIL_CACHE", str(cache))
    monkeypatch.setenv("ADANVIL_MUSIC_MIN_FREE_GB", "0")
    monkeypatch.setenv("ADANVIL_TEST_MODE", mode)
    monkeypatch.setenv("ADANVIL_TEST_VICTIM", str(victim))
    monkeypatch.setattr(musicgen, "WORKER", worker)
    root = tmp_path / "p"
    init_project(root, "kettles")
    return root, victim


@pytest.mark.parametrize("mode", ["outside", "link"])
def test_music_generation_never_reads_or_deletes_a_file_outside_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    root, victim = _music_project(tmp_path, monkeypatch, mode)
    if mode == "link":
        try:
            os.symlink(victim, tmp_path / "probe")
        except (OSError, NotImplementedError):
            pytest.skip("symbolic links are not available")
    with pytest.raises(musicgen.MusicUnavailableError, match="its own output folder"):
        musicgen.generate_music(root, "calm piano", 10, seed=3)
    assert victim.read_bytes() == b"precious"
    assert not list((root / "music").glob("gen-*"))  # no track, no half record
    assert not list((root / "cache").rglob("request.json"))  # the request folder is gone


def test_music_generation_ignores_a_result_line_forged_in_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for mode in ("forged_first", "garbage_first", "plain"):
        root, victim = _music_project(tmp_path / mode, monkeypatch, mode)
        result = musicgen.generate_music(root, "calm piano", 10, seed=3)
        assert (root / result["path"]).is_file() and victim.read_bytes() == b"precious"
        assert result["generation_seconds"] == 1.5


def test_a_planted_link_at_the_track_name_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, victim = _music_project(tmp_path, monkeypatch, "plain")
    track = root / "music" / "gen-calm-piano-3.flac"
    try:
        track.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    musicgen.generate_music(root, "calm piano", 10, seed=3)
    assert (
        victim.read_bytes() == b"precious"
        and not track.is_symlink()
        and track.stat().st_size > 1000
    )


# --------------------------------------------------------------------------------------- C5


@pytest.mark.parametrize(
    "voice",
    [
        "../../voices/evil.pt",
        "/home/user/.cache/evil.pt",
        "C:\\Users\\victim\\evil.pt",
        "af_heart.pt",
        "af_heart,af_bella",
        "af_heart,/tmp/x.pt",
        "AF_heart",
        "a_heart",
        "afheart",
        "af_",
        "af_heart ",
        "af-heart",
        "",
    ],
)
def test_a_kokoro_voice_that_is_not_a_plain_name_is_refused(voice: str) -> None:
    with pytest.raises(ValidationError, match="Kokoro voice"):
        VoiceSpec(engine="kokoro", voice=voice, lines=["Bonjour"])


@pytest.mark.parametrize(
    "voice", ["af_heart", "ff_siwis", "bf_emma", "am_adam", "zf_xiaobei", "hf_alpha"]
)
def test_the_kokoro_voices_still_work(voice: str) -> None:
    assert VoiceSpec(engine="kokoro", voice=voice, lines=["Bonjour"]).voice == voice
    assert VoiceSpec(engine="kokoro", lines=["Bonjour"]).voice is None


def test_the_engine_refuses_a_voice_that_is_not_a_name(tmp_path: Path) -> None:
    engine = WorkerEngine(engine="kokoro", python=sys.executable, persistent=False)
    job = LineJob(index=0, text="Hi", language="en", out=tmp_path / "a.wav", voice="x/../y.pt")
    with pytest.raises(ValueError, match="Kokoro voice name"):
        engine.synthesize([job])


def _fake_kokoro_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Fake ``torch``, ``huggingface_hub`` and ``kokoro`` that record how the model is loaded."""

    world = SimpleNamespace(downloads=[], voices=[], pipelines=[], models=[], loaded=[])

    class FakeModel:
        def __init__(self, repo_id: str, config: str, model: str) -> None:
            world.models.append((repo_id, Path(config).name, Path(model).name))

        def eval(self) -> FakeModel:
            return self

    class FakePipeline:
        def __init__(self, lang_code: str, repo_id: str, model: Any = None) -> None:
            world.pipelines.append((lang_code, repo_id, type(model).__name__))

        def __call__(self, text: str, voice: Any):
            world.voices.append(voice)
            yield "g", "p", np.full(2400, 0.25, dtype="float32")

    def snapshot_download(repo: str, revision: str | None = None, **kwargs: Any) -> str:
        world.downloads.append((repo, revision, tuple(kwargs.get("allow_patterns") or ())))
        folder = tmp_path / "snapshot"
        (folder / "voices").mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "voices/ff_siwis.pt"):
            (folder / name).write_bytes(b"x")
        return str(folder)

    def load(path: Any, weights_only: bool = False) -> str:
        world.loaded.append((Path(path).name, weights_only))
        return f"tensor:{Path(path).stem}"

    torch = ModuleType("torch")
    torch.set_num_threads = lambda _n: None  # type: ignore[attr-defined]
    torch.load = load  # type: ignore[attr-defined]
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    kokoro = ModuleType("kokoro")
    kokoro.KPipeline = FakePipeline  # type: ignore[attr-defined]
    kokoro.KModel = FakeModel  # type: ignore[attr-defined]
    for name, module in {"torch": torch, "huggingface_hub": hub, "kokoro": kokoro}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return world


def test_the_worker_refuses_a_kokoro_voice_that_is_not_a_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = _fake_kokoro_world(monkeypatch, tmp_path)
    for voice in ("../../evil.pt", "af_heart,af_bella", "AF_heart", "x.pt"):
        result = voice_worker._tts(
            {"engine": "kokoro", "text": "a", "language": "en", "voice": voice,
             "out": str(tmp_path / "o.wav")},
            {},
        )  # fmt: skip
        assert result["ok"] is False and "Kokoro voice name" in result["error"]
    assert world.pipelines == [] and world.downloads == [] and world.loaded == []


def test_kokoro_is_loaded_from_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = _fake_kokoro_world(monkeypatch, tmp_path)
    models: dict = {}
    out = tmp_path / "k.wav"
    for voice in ("ff_siwis", "af_heart", "ff_siwis"):
        language = "fr" if voice.startswith("f") else "en"
        op = {"engine": "kokoro", "text": "Bonjour", "language": language, "voice": voice,
              "out": str(out)}  # fmt: skip
        assert voice_worker._tts(op, models)["ok"] is True
    assert world.downloads and all(
        repo == "hexgrad/Kokoro-82M" and revision == "f3ff3571791e39611d31c381e3a41a3af07b4987"
        for repo, revision, _patterns in world.downloads
    )
    assert world.models == [("hexgrad/Kokoro-82M", "config.json", "kokoro-v1_0.pth")]  # once
    assert world.pipelines == [("f", "hexgrad/Kokoro-82M", "FakeModel"),
                               ("a", "hexgrad/Kokoro-82M", "FakeModel")]  # fmt: skip
    # The voice reaches Kokoro as a tensor read from the pinned snapshot, never as a string
    # Kokoro could take for a file or a blend.
    assert world.voices == ["tensor:ff_siwis", "tensor:af_heart", "tensor:ff_siwis"]
    assert world.loaded == [("ff_siwis.pt", True), ("af_heart.pt", True)]


def test_kokoro_reports_a_voice_the_pinned_repository_does_not_have(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_kokoro_world(monkeypatch, tmp_path)
    result = voice_worker._tts(
        {"engine": "kokoro", "text": "a", "language": "en", "voice": "af_nope",
         "out": str(tmp_path / "o.wav")},
        {},
    )  # fmt: skip
    assert result["ok"] is False and "no voice named 'af_nope'" in result["error"]


def _fake_chatterbox_world(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: str | None
) -> SimpleNamespace:
    world = SimpleNamespace(downloads=[], loads=[])

    class FakeTTS:
        sr = 24000

        @classmethod
        def from_local(cls, folder: str, device: str) -> FakeTTS:
            world.loads.append(("local", folder, device))
            return cls()

        @classmethod
        def from_pretrained(cls, device: str) -> FakeTTS:
            world.loads.append(("pretrained", None, device))
            return cls()

        def prepare_conditionals(self, path: str, exaggeration: float = 0.5) -> None:
            pass

        def generate(self, text: str, **kwargs: object) -> str:
            return "wav"

    def snapshot_download(**kwargs: Any) -> str:
        world.downloads.append(kwargs)
        return str(tmp_path / "chatterbox")

    torch = ModuleType("torch")
    torch.set_num_threads = lambda _n: None  # type: ignore[attr-defined]
    torch.manual_seed = lambda _n: None  # type: ignore[attr-defined]
    torchaudio = ModuleType("torchaudio")
    torchaudio.save = lambda path, _wav, _sr: Path(path).write_bytes(b"x")  # type: ignore[attr-defined]
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    mtl = ModuleType("chatterbox.mtl_tts")
    mtl.ChatterboxMultilingualTTS = FakeTTS  # type: ignore[attr-defined]
    modules = {
        "torch": torch,
        "torchaudio": torchaudio,
        "huggingface_hub": hub,
        "chatterbox": ModuleType("chatterbox"),
        "chatterbox.mtl_tts": mtl,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(voice_worker, "_library_version", lambda _name: version)
    return world


def test_chatterbox_is_loaded_from_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = _fake_chatterbox_world(monkeypatch, tmp_path, "0.1.7")
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"ref")
    op = {"op": "tts", "text": "un", "language": "fr", "reference": str(reference),
          "out": str(tmp_path / "o.wav")}  # fmt: skip
    assert voice_worker._tts(op, {})["ok"] is True
    assert len(world.downloads) == 1
    download = world.downloads[0]
    assert download["repo_id"] == "ResembleAI/chatterbox"
    assert download["revision"] == "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
    assert "t3_mtl23ls_v2.safetensors" in download["allow_patterns"]
    assert world.loads == [("local", str(tmp_path / "chatterbox"), "cpu")]


def test_chatterbox_keeps_its_own_loader_for_another_library_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = _fake_chatterbox_world(monkeypatch, tmp_path, "0.1.4")
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"ref")
    op = {"op": "tts", "text": "un", "language": "fr", "reference": str(reference),
          "out": str(tmp_path / "o.wav")}  # fmt: skip
    assert voice_worker._tts(op, {})["ok"] is True
    assert world.downloads == [] and world.loads == [("pretrained", None, "cpu")]


class _WhisperSpy:
    """A ``WhisperModel`` that records how it was built."""

    built: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, name: str, **kwargs: Any) -> None:
        _WhisperSpy.built.append((name, kwargs))

    def transcribe(self, audio: Any, **kwargs: Any):
        return [SimpleNamespace(text="", words=[])], None


@pytest.fixture
def whisper_spy(monkeypatch: pytest.MonkeyPatch) -> type[_WhisperSpy]:
    _WhisperSpy.built = []
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=_WhisperSpy))
    monkeypatch.setattr(voice_worker, "_load_16k", lambda path: path)
    for name in ("ADANVIL_SPEECH_MODEL", "ADANVIL_VOICE_CHECK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return _WhisperSpy


@pytest.mark.parametrize(
    "name", ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"]
)
def test_the_worker_loads_the_allowed_whisper_models(
    whisper_spy: type[_WhisperSpy], name: str
) -> None:
    assert voice_worker._align({"audio": "a.wav", "model": name}, {})["ok"] is True
    ((loaded, options),) = whisper_spy.built
    assert loaded == name and options["device"] == "cpu"
    pinned = voice_worker.WHISPER_REVISIONS.get(name)
    assert options.get("revision") == pinned  # base, small and medium only


@pytest.mark.parametrize(
    "name",
    [
        "../../../tmp/evil",
        "/models/evil",
        "C:\\evil",
        "someone/evil-whisper",
        "base.en",
        "turbo",
        "",
        "BASE",
    ],
)
def test_the_worker_refuses_other_whisper_models(whisper_spy: type[_WhisperSpy], name: str) -> None:
    result = voice_worker._align({"audio": "a.wav", "model": name}, {})
    assert result["ok"] is False and "not allowed" in result["error"] and whisper_spy.built == []


def test_the_operators_own_whisper_model_stays_free(
    whisper_spy: type[_WhisperSpy], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADANVIL_SPEECH_MODEL", "/models/my-distilled-whisper")
    assert voice_worker._align({"audio": "a.wav", "model": "/models/my-distilled-whisper"}, {})[
        "ok"
    ]
    assert whisper_spy.built[0][0] == "/models/my-distilled-whisper"
    assert "revision" not in whisper_spy.built[0][1]
    assert voice_worker._align({"audio": "a.wav", "model": "/models/other"}, {})["ok"] is False
    monkeypatch.setenv("ADANVIL_VOICE_CHECK_MODEL", "my-check-model")
    assert voice_worker._align({"audio": "a.wav", "model": "my-check-model"}, {})["ok"] is True


def test_no_revision_is_sent_to_a_library_that_does_not_take_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    class Old:
        def __init__(self, name: str, device: str, compute_type: str, cpu_threads: int) -> None:
            seen.append({"name": name})

        def transcribe(self, audio: Any, **kwargs: Any):
            return [SimpleNamespace(text="", words=[])], None

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Old))
    monkeypatch.setattr(voice_worker, "_load_16k", lambda path: path)
    assert voice_worker._align({"audio": "a.wav", "model": "base"}, {})["ok"] is True
    assert seen == [{"name": "base"}]


def test_the_pinned_revisions_are_the_same_in_the_worker_and_in_adanvil() -> None:
    assert voice_worker.WHISPER_MODELS == transcribe.WHISPER_MODELS
    assert voice_worker.WHISPER_REVISIONS == transcribe.WHISPER_REVISIONS
    assert set(transcribe.WHISPER_REVISIONS) == {"base", "small", "medium"}


def test_transcribe_refuses_a_model_name_from_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ADANVIL_SPEECH_MODEL", raising=False)
    for name in ("someone/evil-whisper", "/tmp/model", "..\\model", "base.en"):
        with pytest.raises(ValueError, match="not allowed"):
            transcribe.transcribe_clips([], project_root=tmp_path, model=name)
    for name in transcribe.WHISPER_MODELS:
        assert transcribe.check_model_name(name) == name
    monkeypatch.setenv("ADANVIL_SPEECH_MODEL", "my-local-model")
    assert transcribe.check_model_name("my-local-model") == "my-local-model"
    assert transcribe.transcribe_clips([], project_root=tmp_path) == []  # the env name is free


def _tone(path: Path) -> None:
    samples = (0.3 * np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())


def test_the_in_process_transcriber_passes_the_pinned_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[tuple[str, dict[str, Any]]] = []

    class Model:
        def __init__(
            self, name: str, *, device: str, compute_type: str, revision: str | None = None
        ):
            built.append((name, {"revision": revision}))

        def transcribe(self, audio: Any, **kwargs: Any):
            return [SimpleNamespace(words=[])], SimpleNamespace(language="en")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    monkeypatch.delenv("ADANVIL_SPEECH_MODEL", raising=False)
    clip = tmp_path / "a.wav"
    _tone(clip)
    for name in ("small", "large-v3"):
        transcribe.transcribe_clips([clip], project_root=tmp_path, model=name)
    assert built == [
        ("small", {"revision": transcribe.WHISPER_REVISIONS["small"]}),
        ("large-v3", {"revision": None}),
    ]


# --------------------------------------------------------------------------------------- C6


def _plan(commands: list[list[str]]) -> str:
    return "\n".join(" ".join(command) for command in commands)


def test_the_voice_environment_pins_the_packages_that_load_models() -> None:
    chatterbox = _plan(voice_setup_module.voice_setup(plan=True))
    assert "chatterbox-tts==0.1.7" in chatterbox and "faster-whisper==1.2.1" in chatterbox
    kokoro = _plan(voice_setup_module.voice_setup(plan=True, engine="kokoro"))
    assert "kokoro==0.9.4" in kokoro and "faster-whisper==1.2.1" in kokoro
    assert "soundfile" in kokoro and "librosa" in kokoro  # free: their wheels vary by platform
    for text in (chatterbox, kokoro):
        assert "chatterbox-tts " not in text and "kokoro>=" not in text


def test_the_spacy_model_is_installed_against_its_hash() -> None:
    kokoro = _plan(voice_setup_module.voice_setup(plan=True, engine="kokoro"))
    url = next(word for word in kokoro.split() if "en_core_web_sm" in word)
    assert url.startswith("https://github.com/explosion/spacy-models/releases/download/")
    assert url.endswith("#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85")


def test_the_vision_environment_pins_transformers_and_the_hub() -> None:
    from adanvil.vision.setup import vision_setup

    plan = _plan(vision_setup(plan=True))
    assert "transformers==5.19.0" in plan and "huggingface_hub==1.33.0" in plan
    # torch, numpy, pillow and opencv stay free: their wheels vary by platform.
    assert "torch>=" not in plan and "numpy>=" not in plan and "opencv-python-headless" in plan


# --------------------------------------------------------------------------------------- C7


def _linked(tmp_path: Path, link: Path) -> Path:
    victim = tmp_path / f"victim-{link.name}"
    victim.write_text("precious", encoding="utf-8")
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    return victim


def test_brand_yaml_is_not_written_through_a_link(tmp_path: Path) -> None:
    victim = _linked(tmp_path, tmp_path / "project" / "brand.yaml")
    save_brand(default_brand("Kettles"), tmp_path / "project" / "brand.yaml")
    assert victim.read_text(encoding="utf-8") == "precious"
    assert "Kettles" in (tmp_path / "project" / "brand.yaml").read_text(encoding="utf-8")


def test_onboarding_files_are_not_written_through_links(
    tmp_path: Path, allow_private: None
) -> None:
    root = tmp_path / "project"
    victims = [_linked(tmp_path, root / "brand" / "site_facts.json")]
    page = b"<html><head><title>Fernwood</title></head><body><h1>Coffee</h1></body></html>"
    with serve(lambda handler: reply(handler, page, "text/html")) as (base, _seen):
        onboard(root, url=base + "/", answers=None)
        victims.append(_linked(tmp_path, root / "brand_brief.md"))
        victims.append(_linked(tmp_path, root / "brand.yaml"))
        answers = {"brand_name": "Fernwood", "website": base, "niche": "coffee"}
        onboard(root, url=None, answers=answers)
    assert all(victim.read_text(encoding="utf-8") == "precious" for victim in victims)
    assert "Fernwood" in (root / "brand_brief.md").read_text(encoding="utf-8")
    assert not (root / "brand" / "site_facts.json").is_symlink()


def test_the_voice_manifest_and_lines_are_not_written_through_links(tmp_path: Path) -> None:
    root = tmp_path / "project"
    init_project(root, "kettles")
    folder = root / "voice" / "v"
    victims = [
        _linked(tmp_path, folder / "manifest.json"),
        _linked(tmp_path, folder / "manifest.json.tmp"),
        _linked(tmp_path, folder / "line_01.wav"),
    ]
    track = generate_voice(
        root, VoiceSpec(engine="fake", language="en", lines=["Bonjour le monde"]), "v",
        engine=FakeEngine(),
    )  # fmt: skip
    assert all(victim.read_text(encoding="utf-8") == "precious" for victim in victims)
    assert track.lines[0].duration > 0 and (folder / "manifest.json").is_file()
    assert not (folder / "manifest.json").is_symlink()
    assert not (folder / "line_01.wav").is_symlink()


def test_the_worker_job_file_goes_through_a_rename() -> None:
    import inspect

    assert "replace_text(jobs_path" in inspect.getsource(voice_engines.WorkerEngine._run_files)
    assert voice_generate.replace_text is not None


def test_the_helper_environments_remember_their_interpreter_through_a_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from adanvil.vision.setup import vision_setup

    cache = tmp_path / "cache"
    monkeypatch.setenv("ADANVIL_CACHE", str(cache))
    monkeypatch.setenv("ADANVIL_VISION_DIR", str(tmp_path / "models"))

    def fake_run(command: list[str], **_kw: Any) -> subprocess.CompletedProcess:
        if len(command) > 2 and command[1] == "venv":  # `uv venv <dir>`: make the interpreter
            python = Path(command[2]) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("shutil.which", lambda name: "uv" if name == "uv" else None)
    setups = (
        ("voice_config.json", lambda: voice_setup_module.voice_setup(tmp_path / "venv1")),
        ("vision_config.json", lambda: vision_setup(tmp_path / "venv2")),
        ("music_config.json", lambda: musicgen.music_setup(tmp_path / "venv3")),
    )
    for filename, setup in setups:
        victim = _linked(tmp_path, cache / filename)
        setup()
        assert victim.read_text(encoding="utf-8") == "precious", filename
        assert json.loads((cache / filename).read_text(encoding="utf-8"))["python"]
        assert not (cache / filename).is_symlink()
        (cache / filename).unlink()
