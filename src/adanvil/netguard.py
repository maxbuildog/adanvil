"""A guarded HTTP client: the only way AdAnvil downloads anything.

AdAnvil is driven by an AI agent that can be steered by text hidden in a web page, a track
record or a project folder. A URL that reaches the network layer is therefore untrusted: it
must not be able to read a local file (``file:``), call a service of the user's machine or
network (``http://127.0.0.1:...``, ``http://169.254.169.254/...``), follow a redirect there,
fill the disk or hold the tool for ever. Everything that downloads goes through :func:`fetch`
(the body in memory) or :func:`fetch_to_file` (the body streamed to a new file).

The rules, standard library only (``urllib``, ``http.client``, ``socket``, ``ipaddress``):

* Only ``http`` and ``https`` URLs, without credentials in them. Anything else (``file:``,
  ``ftp:``, ``data:``) is refused before any I/O.
* The host must resolve only to global addresses: loopback, private, link-local (169.254.0.0/16,
  fe80::/10), multicast, reserved, unspecified, carrier-grade NAT (100.64.0.0/10), unique-local
  IPv6 and the IPv4-mapped, NAT64 and 6to4 forms of those are refused. The check is made on the
  name before the request, and again when the socket connects: the connection goes to an
  address that was just checked, and the peer address of the connected socket is checked once
  more, which also defeats DNS rebinding. When the environment configures a proxy, the peer is
  the proxy and its address is not checked, but the name of the target still is.
* At most 5 redirects, and every hop is checked like the first request (scheme and host). The
  credentials headers (``Authorization``, ``Cookie``) are dropped when a redirect changes
  host.
* The body is read in pieces and refused beyond ``max_bytes``. An overall deadline, on top of
  the socket timeout, closes the connection of a server that drips one byte at a time.

``ADANVIL_ALLOW_PRIVATE_URLS=1`` lifts the address rule, for development against a local
server (still http and https only, with the other limits). Never set it for an agent session.
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import Any

from adanvil import __version__

ALLOW_PRIVATE_ENV = "ADANVIL_ALLOW_PRIVATE_URLS"
MAX_REDIRECTS = 5
_CHUNK = 64 * 1024
_SENSITIVE_HEADERS = ("authorization", "cookie", "proxy-authorization")
_DEFAULT_PORTS = {"http": 80, "https": 443}

_BLOCKED_V4 = tuple(
    ipaddress.ip_network(network)
    for network in (
        "0.0.0.0/8",  # "this" network, unspecified
        "10.0.0.0/8",
        "100.64.0.0/10",  # carrier-grade NAT
        "127.0.0.0/8",  # loopback
        "169.254.0.0/16",  # link-local, cloud metadata services
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",  # multicast
        "240.0.0.0/4",  # reserved, broadcast
    )
)
_BLOCKED_V6 = tuple(
    ipaddress.ip_network(network)
    for network in (
        "::/96",  # unspecified, loopback, IPv4-compatible
        "64:ff9b:1::/48",
        "100::/64",
        "2001::/23",
        "2001:db8::/32",
        "3fff::/20",
        "5f00::/16",
        "fc00::/7",  # unique local
        "fe80::/10",  # link-local
        "fec0::/10",
        "ff00::/8",  # multicast
    )
)


class NetguardError(Exception):
    """A download was refused or cut short by the guard; the message says why."""


@dataclass(frozen=True)
class Response:
    """The answer to a guarded request. ``body`` is empty when it was streamed to a file."""

    status: int
    headers: Message
    body: bytes
    url: str
    size: int


def private_urls_allowed() -> bool:
    """Whether ``ADANVIL_ALLOW_PRIVATE_URLS`` lifts the address rule (development only)."""

    return os.environ.get(ALLOW_PRIVATE_ENV, "").strip().casefold() in {"1", "true", "yes", "on"}


def is_public_address(value: str) -> bool:
    """Whether ``value`` (an IP address) is a global address a download may talk to."""

    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        embedded = address.ipv4_mapped
        if embedded is None and address in ipaddress.ip_network("64:ff9b::/96"):
            embedded = ipaddress.IPv4Address(address.packed[-4:])  # NAT64
        if embedded is None and address in ipaddress.ip_network("2002::/16"):
            embedded = ipaddress.IPv4Address(address.packed[2:6])  # 6to4
        if embedded is not None:
            return is_public_address(str(embedded))
        return address.is_global and not any(address in net for net in _BLOCKED_V6)
    return address.is_global and not any(address in net for net in _BLOCKED_V4)


def _resolve(host: str, port: int, allow_private: bool) -> list[tuple[Any, ...]]:
    """The ``(family, type, proto, sockaddr)`` of ``host``; refused unless all are global."""

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise NetguardError(f"cannot resolve the host {host!r}: {exc}") from exc
    if not infos:
        raise NetguardError(f"cannot resolve the host {host!r}")
    addresses = []
    for family, kind, proto, _canonical, sockaddr in infos:
        if not allow_private and not is_public_address(str(sockaddr[0])):
            raise NetguardError(
                f"{host!r} resolves to {sockaddr[0]}, which is not a public internet address: "
                "only public http and https servers can be downloaded from"
            )
        addresses.append((family, kind, proto, sockaddr))
    return addresses


def _check_url(url: str, allow_private: bool) -> urllib.parse.SplitResult:
    """Refuse a URL that is not a plain public ``http(s)`` address; return its parts."""

    try:
        parts = urllib.parse.urlsplit(url.strip())
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise NetguardError(f"not a valid URL: {exc}") from exc
    if parts.scheme.lower() not in _DEFAULT_PORTS:
        raise NetguardError(
            "only http and https URLs can be downloaded "
            f"(this one is {parts.scheme + ':' if parts.scheme else 'not a URL'})"
        )
    if not host:
        raise NetguardError("the URL has no host")
    if "@" in parts.netloc:
        raise NetguardError("URLs with a user name or password in them are refused")
    _resolve(host, port or _DEFAULT_PORTS[parts.scheme.lower()], allow_private)
    return parts


class _Guard:
    """The state of one request: redirects counted, deadline, sockets to cut at the deadline."""

    def __init__(self, deadline: float, allow_private: bool) -> None:
        self.deadline = deadline
        self.allow_private = allow_private
        self.expired = False
        self.hops = 0
        self._lock = threading.Lock()
        self._watched: list[socket.socket] = []
        self._timer = threading.Timer(deadline, self._expire)
        self._timer.daemon = True

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.cancel()
        with self._lock:
            watched, self._watched = self._watched, []
        for sock in watched:
            sock.close()

    def message(self) -> str:
        return f"the download took longer than {self.deadline:g} s and was stopped"

    def check(self) -> None:
        if self.expired:
            raise NetguardError(self.message())

    def watch(self, sock: socket.socket) -> None:
        # A duplicate of the socket: shutting it down cuts the connection even while another
        # thread waits on it (and after the TLS layer took the original over).
        spare = sock.dup()
        with self._lock:
            if self.expired:
                spare.close()
                raise NetguardError(self.message())
            self._watched.append(spare)

    def _expire(self) -> None:
        with self._lock:
            self.expired = True
            watched = list(self._watched)
        for sock in watched:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _verify_peer(sock: socket.socket, host: str) -> None:
    peer = sock.getpeername()[0]
    if not is_public_address(str(peer)):
        raise NetguardError(
            f"{host!r} connected to {peer}, which is not a public internet address: refused"
        )


def _open_socket(
    host: str, port: int, timeout: float | None, guard: _Guard, *, direct: bool
) -> socket.socket:
    guard.check()
    if not direct:
        # A proxy from the environment: the peer is the proxy, the target name was checked.
        sock = socket.create_connection((host, port), timeout)
    else:
        sock = _connect_checked(host, port, timeout, guard.allow_private)
    try:
        guard.watch(sock)
    except BaseException:
        sock.close()
        raise
    return sock


def _connect_checked(
    host: str, port: int, timeout: float | None, allow_private: bool
) -> socket.socket:
    last: OSError | None = None
    for family, kind, proto, sockaddr in _resolve(host, port, allow_private):
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(timeout)
            sock.connect(sockaddr)
            if not allow_private:
                _verify_peer(sock, host)
        except NetguardError:
            sock.close()
            raise
        except OSError as exc:
            sock.close()
            last = exc
            continue
        return sock
    raise last or OSError(f"could not connect to {host}")


def _connection_class(base: type[http.client.HTTPConnection], guard: _Guard, direct: bool):
    class GuardedConnection(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._create_connection = self._guarded_connection

        def _guarded_connection(
            self, address: tuple[str, int], timeout: float | None = None, source: Any = None
        ) -> socket.socket:
            return _open_socket(address[0], address[1], timeout, guard, direct=direct)

    return GuardedConnection


def _prepare(request: urllib.request.Request, guard: _Guard) -> bool:
    """Check the target of ``request``; return whether the connection goes straight to it."""

    parts = _check_url(request.full_url, guard.allow_private)
    guard.check()
    dialed = urllib.parse.urlsplit("//" + request.host)
    default = _DEFAULT_PORTS[parts.scheme.lower()]
    return (dialed.hostname, dialed.port or default) == (parts.hostname, parts.port or default)


class _HttpHandler(urllib.request.HTTPHandler):
    def __init__(self, guard: _Guard) -> None:
        super().__init__()
        self._guard = guard

    def http_open(self, req: urllib.request.Request):
        direct = _prepare(req, self._guard)
        return self.do_open(_connection_class(http.client.HTTPConnection, self._guard, direct), req)


class _HttpsHandler(urllib.request.HTTPSHandler):
    def __init__(self, guard: _Guard) -> None:
        super().__init__()
        self._guard = guard

    def https_open(self, req: urllib.request.Request):
        direct = _prepare(req, self._guard)
        connection = _connection_class(http.client.HTTPSConnection, self._guard, direct)
        return self.do_open(connection, req, context=self._context)


def _origin(url: str) -> tuple[str, str, int]:
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or ""), parts.port or _DEFAULT_PORTS.get(scheme, 0)


class _RedirectHandler(urllib.request.HTTPRedirectHandler):
    max_repeats = MAX_REDIRECTS + 1  # our own count, with its own message, comes first

    def __init__(self, guard: _Guard) -> None:
        super().__init__()
        self._guard = guard

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        self._guard.hops += 1
        if self._guard.hops > MAX_REDIRECTS:
            raise NetguardError(f"more than {MAX_REDIRECTS} redirects: refused")
        fp.close()  # the body of a redirect is never read: a server could drip it for ever
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _origin(new.full_url) != _origin(req.full_url):
            for store in (new.headers, new.unredirected_hdrs):
                for name in [key for key in store if key.lower() in _SENSITIVE_HEADERS]:
                    del store[name]
        return new


def _opener(guard: _Guard) -> urllib.request.OpenerDirector:
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(),  # the proxies of the environment, as urlopen does
        urllib.request.UnknownHandler(),
        _HttpHandler(guard),
        _HttpsHandler(guard),
        urllib.request.HTTPDefaultErrorHandler(),
        _RedirectHandler(guard),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


def _get(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    headers: Mapping[str, str] | None,
    deadline: float | None,
    write: Any,
) -> Response:
    allow_private = private_urls_allowed()
    _check_url(url, allow_private)  # scheme and host are refused before any I/O
    if deadline is None:
        deadline = max(4 * float(timeout), 30.0)
    guard = _Guard(deadline, allow_private)
    request_headers = {"User-Agent": f"AdAnvil/{__version__}", **(headers or {})}
    request = urllib.request.Request(url.strip(), headers=request_headers, method="GET")
    guard.start()
    try:
        response = _opener(guard).open(request, timeout=timeout)
        with response:
            declared = response.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > max_bytes:
                raise NetguardError(
                    f"the response is {int(declared)} bytes, more than the {max_bytes} allowed: "
                    "refused"
                )
            size = 0
            while True:
                guard.check()
                chunk = response.read1(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise NetguardError(f"the response is larger than {max_bytes} bytes: refused")
                write(chunk)
            guard.check()  # a connection cut at the deadline looks like the end of the body
            if declared.isdigit() and size != int(declared):
                raise NetguardError(f"the download stopped at {size} of {declared} bytes")
            return Response(response.status, response.headers, b"", response.geturl(), size)
    except urllib.error.HTTPError as exc:
        exc.close()
        if guard.expired:
            raise NetguardError(guard.message()) from exc
        raise
    except NetguardError:
        raise
    except Exception as exc:
        if guard.expired:
            raise NetguardError(guard.message()) from exc
        raise
    finally:
        guard.stop()


def fetch(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    headers: Mapping[str, str] | None = None,
    deadline: float | None = None,
) -> Response:
    """GET ``url`` and return its body (at most ``max_bytes``) with the status, headers and
    final URL. ``timeout`` is the socket timeout; ``deadline`` (seconds, default four times the
    timeout and at least 30) bounds the whole request. A refusal raises :class:`NetguardError`;
    an HTTP error status raises ``urllib.error.HTTPError`` like ``urlopen``."""

    pieces: list[bytes] = []
    response = _get(
        url, timeout=timeout, max_bytes=max_bytes, headers=headers, deadline=deadline,
        write=pieces.append,
    )  # fmt: skip
    body = b"".join(pieces)
    return Response(response.status, response.headers, body, response.url, response.size)


def fetch_to_file(
    url: str,
    destination: str | os.PathLike[str],
    *,
    timeout: float,
    max_bytes: int,
    headers: Mapping[str, str] | None = None,
    deadline: float | None = None,
) -> Response:
    """Stream the body of ``url`` to ``destination``, which must not exist (it is created
    exclusively, so a planted link is never followed) and is removed when the download fails.
    The limits are those of :func:`fetch`; the default deadline is ten minutes."""

    path = Path(destination)
    if deadline is None:
        deadline = max(600.0, 4 * float(timeout))
    out = open(path, "xb")  # noqa: SIM115 (an existing path raises here and is left alone)
    try:
        with out:
            return _get(
                url, timeout=timeout, max_bytes=max_bytes, headers=headers, deadline=deadline,
                write=out.write,
            )  # fmt: skip
    except BaseException:
        path.unlink(missing_ok=True)
        raise


__all__ = [
    "ALLOW_PRIVATE_ENV",
    "MAX_REDIRECTS",
    "NetguardError",
    "Response",
    "fetch",
    "fetch_to_file",
    "is_public_address",
    "private_urls_allowed",
]
