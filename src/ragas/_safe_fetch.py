"""SSRF-guarded HTTP download for URLs taken from untrusted dataset content.

Private helper shared by the multimodal image loaders
(``ragas.prompt.multi_modal_prompt`` and
``ragas.metrics.collections.multi_modal_faithfulness.util``).

Two checks run on the initial request and on every redirect hop:

* the host that ``requests``/``urllib3`` will actually contact is refused if
  it is ``localhost`` or a non-public IP literal;
* on a direct connection the *connected peer IP* is validated at
  socket-connect time, so a hostname that resolves to an internal address
  (including via DNS rebinding) is refused before any request is sent.

When the request goes through an HTTP(S)/SOCKS proxy the peer is the proxy, so
the second check cannot see the real target. In that case the target hostname
is resolved locally and refused if it maps to a non-public address; if it
cannot be resolved locally (proxy-only networks) the request is left to the
proxy.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import typing as t

import requests
from requests.adapters import HTTPAdapter
from requests.utils import select_proxy

# NOTE: the guard subclasses urllib3 connection/pool internals. These are
# stable across urllib3 1.26 and 2.x but are not a public API; the test suite
# asserts that a loopback URL is refused, so a future urllib3 refactor cannot
# silently void the guard.
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.util import parse_url

logger = logging.getLogger(__name__)

# Maximum redirect hops to follow for an untrusted URL.
MAX_REDIRECTS = 10

_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")


class BlockedAddressError(Exception):
    """Raised when a URL targets a non-public (SSRF-unsafe) address."""


def is_public_ip(ip_str: str) -> bool:
    """True only if ``ip_str`` is a globally-routable public unicast address.

    IPv4-mapped / 6to4 / Teredo / NAT64 IPv6 addresses are unwrapped first so
    an embedded private or link-local IPv4 (e.g. the ``169.254.169.254`` cloud
    metadata endpoint) cannot slip through. ``is_global`` rejects CGNAT
    100.64.0.0/10, 0.0.0.0/8, loopback, private, link-local and reserved space.
    """
    try:
        ip: t.Any = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    elif getattr(ip, "sixtofour", None) is not None:
        ip = ip.sixtofour
    elif getattr(ip, "teredo", None) is not None:
        ip = ip.teredo[1]  # tunnelled client endpoint
    elif ip.version == 6 and ip in _NAT64_PREFIX:
        ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if ip.is_multicast or getattr(ip, "is_site_local", False):
        return False
    return bool(ip.is_global)


def _ip_literal(host: str) -> t.Optional[str]:
    """Return ``host`` as a canonical IP string if it is an IP literal.

    Also recognises the legacy IPv4 spellings a resolver accepts
    (``2130706433``, ``0x7f.1``, ``127.1``).
    """
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        return socket.inet_ntoa(socket.inet_aton(host))
    except (OSError, ValueError):
        return None


def _resolve(host: str) -> t.List[str]:
    """Resolve ``host`` to its IP addresses (raises ``OSError`` on failure)."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, None)]


def _check_target_host(url: str, via_proxy: bool) -> None:
    """Refuse ``url`` if its host is, or locally resolves to, a non-public address.

    The host is extracted with urllib3's parser - the one that decides where
    the connection goes - so it cannot disagree with the request target (e.g.
    ``http://127.0.0.1:6666\\@1.1.1.1``).
    """
    host = parse_url(url).host
    if not host:
        raise BlockedAddressError(f"no host in URL {url!r}")
    host = host.strip("[]").split("%", 1)[0].rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise BlockedAddressError(f"blocked non-public host {host!r}")

    literal = _ip_literal(host)
    if literal is not None:
        if not is_public_ip(literal):
            raise BlockedAddressError(f"blocked non-public address {literal}")
        return

    if via_proxy:
        # The connected peer will be the proxy, so the peer-IP check cannot
        # see the target. Resolve it here instead; if this host has no usable
        # DNS (proxy-only network) the proxy is trusted to resolve it.
        try:
            addresses = _resolve(host)
        except OSError:
            return
        for address in addresses:
            if not is_public_ip(address):
                raise BlockedAddressError(
                    f"blocked host {host!r} resolving to non-public address {address}"
                )


def _assert_peer_public(sock: t.Any) -> None:
    try:
        peer_ip = sock.getpeername()[0]
    except (OSError, AttributeError, IndexError, TypeError) as exc:
        raise BlockedAddressError("could not determine peer address") from exc
    if not is_public_ip(peer_ip):
        raise BlockedAddressError(f"blocked non-public address {peer_ip}")


class _GuardedHTTPConnection(HTTPConnection):
    def connect(self) -> None:  # type: ignore[override]
        super().connect()
        _assert_peer_public(self.sock)


class _GuardedHTTPSConnection(HTTPSConnection):
    def connect(self) -> None:  # type: ignore[override]
        super().connect()
        _assert_peer_public(self.sock)


class _GuardedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _GuardedHTTPConnection  # type: ignore[assignment]


class _GuardedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _GuardedHTTPSConnection  # type: ignore[assignment]


class _SSRFGuardAdapter(HTTPAdapter):
    """requests adapter that validates the target of the initial request and
    of every redirect hop.

    Direct connections validate the *dialed* peer address (not a prior DNS
    lookup), which makes them immune to DNS rebinding / TOCTOU. Proxied
    requests use requests' normal proxy managers, so proxy and CA settings
    keep working; their target host is validated before the request is handed
    to the proxy.
    """

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):  # type: ignore[override]
        super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _GuardedHTTPConnectionPool,
            "https": _GuardedHTTPSConnectionPool,
        }

    def send(  # type: ignore[override]
        self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None
    ):
        url = request.url or ""
        via_proxy = select_proxy(url, proxies) is not None
        _check_target_host(url, via_proxy)
        return super().send(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )


def fetch_url_bytes(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    allow_internal: bool = False,
) -> bytes:
    """Download ``url`` and return its body, refusing non-public targets.

    Args:
        url: http(s) URL to download.
        timeout: Request timeout in seconds.
        max_bytes: Maximum accepted body size.
        allow_internal: Skip the SSRF guard (dangerous; explicit opt-in only).

    Raises:
        BlockedAddressError: The URL, or a redirect hop, targets a non-public address.
        requests.exceptions.RequestException: The download failed.
        ValueError: The body is larger than ``max_bytes``.
    """
    with requests.Session() as session:
        session.max_redirects = MAX_REDIRECTS
        if allow_internal:
            logger.warning(
                "SSRF guard bypassed for URL %s (internal targets allowed)", url
            )
        else:
            adapter = _SSRFGuardAdapter(max_retries=0)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

        with session.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(f"content length {content_length} exceeds {max_bytes}")

            body = bytearray()
            for chunk in response.iter_content(chunk_size=8192):
                body += chunk
                if len(body) > max_bytes:
                    raise ValueError(f"download exceeded {max_bytes} bytes")
            return bytes(body)
