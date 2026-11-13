"""Security regression tests for multimodal image loading.

Covers CVE-2026-6587 / GHSA-95ww-475f-pr4f and #2754: an image reference taken
from ``retrieved_contexts`` must never make Ragas fetch a non-public (internal)
address - directly, via redirect, via a hostname that resolves internally, or
via a URL that ``urlparse`` and ``requests`` read differently - and must never
read a local file unless the user opted in.

The same guard backs three entry points, all exercised here: the collections
``MultiModalFaithfulness`` and ``MultiModalRelevance`` metrics and the legacy
``ImageTextPromptValue``.

These tests also pin the behaviour of the urllib3 connection internals the
guard relies on, so a future urllib3 refactor that broke the guard would fail
here rather than silently. Nothing here touches the network: DNS and
non-loopback connects are stubbed out by the ``net`` fixture.
"""

import io
import ipaddress
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, urlparse

import pytest
import requests
from PIL import Image

import ragas._safe_fetch as safe_fetch
from ragas._safe_fetch import BlockedAddressError, fetch_url_bytes, is_public_ip
from ragas.metrics.collections.multi_modal_faithfulness.util import (
    build_multimodal_message_content,
)
from ragas.metrics.collections.multi_modal_relevance.util import (
    build_multimodal_relevance_message_content,
)
from ragas.prompt import multi_modal_prompt
from ragas.prompt.multi_modal_prompt import ImageTextPromptValue

METADATA_URL = "http://169.254.169.254/latest/meta-data/"
PROXY_ENV_VARS = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"]


def _valid_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


PNG = _valid_png_bytes()


@pytest.fixture(autouse=True)
def net(monkeypatch):
    """Keep every test off the network.

    * proxy settings from the environment are cleared;
    * DNS only answers for names registered in ``net.dns`` (and IP literals);
    * connecting to anything but loopback fails and is recorded in
      ``net.refused_connects``.
    """
    for var in PROXY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "*")

    state = SimpleNamespace(dns={}, refused_connects=[])
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect

    def fake_getaddrinfo(host, *args, **kwargs):
        if isinstance(host, str):
            if host in state.dns:
                host = state.dns[host]
            else:
                try:
                    ipaddress.ip_address(host)
                except ValueError:
                    raise socket.gaierror(socket.EAI_NONAME, "no DNS in tests")
        return real_getaddrinfo(host, *args, **kwargs)

    def loopback_only_connect(self, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
            state.refused_connects.append(address)
            raise OSError("network access is disabled in tests")
        return real_connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", loopback_only_connect)
    return state


@pytest.fixture
def server():
    """A real HTTP server on 127.0.0.1.

    ``/redirect?to=<url>`` answers 302 to ``<url>``; every other path serves a
    valid PNG, so a refused fetch proves the guard fired and not merely that an
    image failed to decode. It answers absolute-URI requests the same way, so
    it doubles as an HTTP proxy. ``server.seen`` lists the request targets.
    """
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence test output
            pass

        def do_GET(self):
            seen.append(self.path)
            parsed = urlparse(self.path)
            if parsed.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", parse_qs(parsed.query)["to"][0])
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(PNG)))
            self.end_headers()
            self.wfile.write(PNG)

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield SimpleNamespace(port=port, url=f"http://127.0.0.1:{port}", seen=seen)
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def proxy(server, monkeypatch):
    """Route plain-http requests through ``server`` acting as an HTTP proxy."""
    monkeypatch.delenv("NO_PROXY")
    monkeypatch.setenv("HTTP_PROXY", server.url)
    return server


def _fetch(url: str, **kwargs) -> bytes:
    return fetch_url_bytes(url, timeout=5, max_bytes=1024 * 1024, **kwargs)


def _redirect(base: str, target: str) -> str:
    return f"{base}/redirect?to={quote(target, safe='')}"


class TestIpClassification:
    """Unit tests for the address classifier."""

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "10.0.0.1",
            "192.168.1.1",
            "172.16.0.1",
            "169.254.169.254",  # cloud metadata
            "100.64.1.1",  # CGNAT (RFC 6598)
            "100.100.100.200",  # Alibaba Cloud metadata (inside CGNAT)
            "0.0.0.0",
            "224.0.0.1",  # multicast
            "255.255.255.255",
            "::1",
            "fc00::1",
            "fe80::1",
            "fec0::1",  # deprecated site-local
            "ff02::1",  # multicast
            "::ffff:169.254.169.254",  # IPv4-mapped metadata
            "::ffff:10.0.0.1",
            "64:ff9b::a9fe:a9fe",  # NAT64-mapped metadata
        ],
    )
    def test_non_public_addresses_are_rejected(self, ip):
        assert is_public_ip(ip) is False

    @pytest.mark.parametrize(
        "ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"]
    )
    def test_public_addresses_are_allowed(self, ip):
        assert is_public_ip(ip) is True

    def test_unparseable_is_rejected(self):
        assert is_public_ip("not-an-ip") is False


class TestGuardedFetch:
    """The shared guard, on direct (non-proxied) connections."""

    @pytest.mark.parametrize(
        "url",
        [
            METADATA_URL,
            "http://100.64.1.1/image.png",  # CGNAT
            "http://100.100.100.200/latest/meta-data/",
            "http://10.0.0.5/image.png",
            "http://[::1]/image.png",
            "http://[::ffff:169.254.169.254]/image.png",
            "http://2130706433/image.png",  # 127.0.0.1 as an integer
            "http://0x7f.1/image.png",  # 127.0.0.1 in hex shorthand
            "http://localhost/image.png",
            "http://LOCALHOST./image.png",
        ],
    )
    def test_internal_target_is_refused_without_connecting(self, url, net):
        with pytest.raises(BlockedAddressError):
            _fetch(url)
        assert net.refused_connects == []

    def test_loopback_server_is_refused(self, server):
        with pytest.raises(BlockedAddressError):
            _fetch(f"{server.url}/image.png")
        assert server.seen == []

    def test_urlparse_requests_mismatch_is_refused(self, server, net):
        """#2754: urlparse reads the host as 1.1.1.1, requests dials 127.0.0.1."""
        url = f"http://127.0.0.1:{server.port}\\@1.1.1.1"
        assert urlparse(url).hostname == "1.1.1.1"
        with pytest.raises(BlockedAddressError):
            _fetch(url)
        assert server.seen == []
        assert net.refused_connects == []

    def test_hostname_resolving_to_internal_is_refused_at_connect(self, server, net):
        """The dialed peer is checked, so DNS (re)binding cannot bypass it."""
        net.dns["rebind.test"] = "127.0.0.1"
        with pytest.raises(BlockedAddressError):
            _fetch(f"http://rebind.test:{server.port}/image.png")
        assert server.seen == []

    def test_redirect_to_internal_address_is_refused(self, server, net, monkeypatch):
        """Every redirect hop is validated, not just the first URL."""
        net.dns["public.test"] = "127.0.0.1"
        net.dns["internal.test"] = "127.0.0.1"
        # Treat only the first dialed peer (public.test) as public.
        verdicts = iter([True])
        monkeypatch.setattr(
            safe_fetch, "is_public_ip", lambda ip: next(verdicts, is_public_ip(ip))
        )
        first_hop = f"http://public.test:{server.port}"

        for target in (METADATA_URL, f"http://internal.test:{server.port}/image.png"):
            verdicts = iter([True])
            server.seen.clear()
            with pytest.raises(BlockedAddressError):
                _fetch(_redirect(first_hop, target))
            assert len(server.seen) == 1  # the redirect itself, never the target
        assert net.refused_connects == []

    def test_redirect_to_non_http_scheme_is_refused(self, server):
        with pytest.raises(requests.exceptions.RequestException):
            _fetch(_redirect(server.url, "file:///etc/passwd"), allow_internal=True)

    def test_opt_in_allows_internal_targets(self, server):
        """With the guard off the same fetches succeed, so the guard is the only gate."""
        assert _fetch(f"{server.url}/image.png", allow_internal=True) == PNG
        redirected = _redirect(server.url, f"{server.url}/image.png")
        assert _fetch(redirected, allow_internal=True) == PNG

    def test_size_limit(self, server):
        with pytest.raises(ValueError):
            fetch_url_bytes(
                f"{server.url}/image.png",
                timeout=5,
                max_bytes=len(PNG) - 1,
                allow_internal=True,
            )


class TestGuardedFetchViaProxy:
    """With a proxy the dialed peer is the proxy, so the target is checked up front."""

    def test_proxy_is_used_for_public_targets(self, proxy, monkeypatch):
        """Proxy users keep working, even though the proxy itself is internal."""
        monkeypatch.setattr(safe_fetch, "_resolve", lambda host: ["93.184.216.34"])
        assert _fetch("http://images.test/image.png") == PNG
        assert proxy.seen == ["http://images.test/image.png"]

    @pytest.mark.parametrize(
        "url",
        [
            METADATA_URL,
            "http://100.64.1.1/image.png",
            "http://127.0.0.1:6666/image.png",
            "http://127.0.0.1:6666\\@1.1.1.1",
            "http://localhost:6666/image.png",
        ],
    )
    def test_internal_literal_is_not_sent_to_proxy(self, url, proxy):
        with pytest.raises(BlockedAddressError):
            _fetch(url)
        assert proxy.seen == []

    def test_hostname_resolving_to_internal_is_not_sent_to_proxy(
        self, proxy, monkeypatch
    ):
        monkeypatch.setattr(safe_fetch, "_resolve", lambda host: ["10.0.0.5"])
        with pytest.raises(BlockedAddressError):
            _fetch("http://intranet.test/image.png")
        assert proxy.seen == []

    def test_redirect_to_internal_is_not_sent_to_proxy(self, proxy, monkeypatch):
        monkeypatch.setattr(safe_fetch, "_resolve", lambda host: ["93.184.216.34"])
        with pytest.raises(BlockedAddressError):
            _fetch(_redirect("http://images.test", METADATA_URL))
        assert len(proxy.seen) == 1  # the redirect itself, never the target

    def test_locally_unresolvable_hostname_is_left_to_proxy(self, proxy, monkeypatch):
        """Known limit: in a proxy-only network the proxy resolves the name."""

        def no_dns(host):
            raise socket.gaierror(socket.EAI_NONAME, "no DNS")

        monkeypatch.setattr(safe_fetch, "_resolve", no_dns)
        assert _fetch("http://images.test/image.png") == PNG


def _faithfulness_loads_image(item: str) -> bool:
    content = build_multimodal_message_content(
        instruction="i", response="r", retrieved_contexts=[item]
    )
    return any(block["type"] == "image_url" for block in content)


def _relevance_loads_image(item: str) -> bool:
    content = build_multimodal_relevance_message_content(
        instruction="i", user_input="q", response="r", retrieved_contexts=[item]
    )
    return any(block["type"] == "image_url" for block in content)


def _legacy_loads_image(item: str) -> bool:
    content = ImageTextPromptValue(items=[item]).to_messages()[0].content
    return any(
        isinstance(block, dict) and block["type"] == "image_url" for block in content
    )


LOADERS = {
    "collections_faithfulness": _faithfulness_loads_image,
    "collections_relevance": _relevance_loads_image,
    "legacy_prompt": _legacy_loads_image,
}
COLLECTIONS_LOADERS = {k: v for k, v in LOADERS.items() if k.startswith("collections")}


@pytest.mark.parametrize("loads_image", LOADERS.values(), ids=LOADERS.keys())
class TestLoadersRefuseInternalUrls:
    """End to end through each public entry point."""

    @pytest.mark.parametrize(
        "url",
        [
            METADATA_URL,
            "http://100.64.1.1/image.png",  # CGNAT
            "http://localhost/image.png",
        ],
    )
    def test_internal_address(self, loads_image, url, net):
        assert loads_image(url) is False
        assert net.refused_connects == []

    def test_loopback_server(self, loads_image, server):
        assert loads_image(f"{server.url}/image.png") is False
        assert server.seen == []

    def test_urlparse_requests_mismatch(self, loads_image, server):
        assert loads_image(f"http://127.0.0.1:{server.port}\\@1.1.1.1") is False
        assert server.seen == []

    def test_redirect_to_internal(self, loads_image, server, net, monkeypatch):
        net.dns["public.test"] = "127.0.0.1"
        verdicts = iter([True])  # only the first dialed peer counts as public
        monkeypatch.setattr(
            safe_fetch, "is_public_ip", lambda ip: next(verdicts, is_public_ip(ip))
        )
        url = _redirect(f"http://public.test:{server.port}", f"{server.url}/image.png")
        assert loads_image(url) is False
        assert len(server.seen) == 1  # the redirect itself, never the target

    def test_opt_in_allows_internal(self, loads_image, server, monkeypatch):
        monkeypatch.setattr(multi_modal_prompt, "ALLOW_INTERNAL_TARGETS", True)
        assert loads_image(f"{server.url}/image.png") is True
        assert server.seen == ["/image.png"]


@pytest.fixture
def image_dir(tmp_path):
    """``allowed/pic.png`` and ``outside/secret.png`` under a temp directory."""
    for name in ("allowed/pic.png", "outside/secret.png"):
        path = tmp_path / name
        path.parent.mkdir()
        path.write_bytes(PNG)
    return tmp_path


@pytest.mark.parametrize("loads_image", LOADERS.values(), ids=LOADERS.keys())
def test_local_file_is_not_read_by_default(loads_image, image_dir, monkeypatch):
    monkeypatch.chdir(image_dir)
    assert loads_image(str(image_dir / "allowed" / "pic.png")) is False
    assert loads_image("allowed/pic.png") is False


@pytest.mark.parametrize(
    "loads_image", COLLECTIONS_LOADERS.values(), ids=COLLECTIONS_LOADERS.keys()
)
class TestCollectionsLocalFileOptIn:
    @pytest.fixture(autouse=True)
    def opt_in(self, image_dir, monkeypatch):
        monkeypatch.setattr(multi_modal_prompt, "ALLOW_LOCAL_FILE_ACCESS", True)
        monkeypatch.setattr(
            multi_modal_prompt, "ALLOWED_IMAGE_BASE_DIR", str(image_dir / "allowed")
        )

    def test_reads_inside_base_dir(self, loads_image, image_dir):
        assert loads_image("pic.png") is True
        assert loads_image(str(image_dir / "allowed" / "pic.png")) is True

    def test_refuses_outside_base_dir(self, loads_image, image_dir):
        assert loads_image(str(image_dir / "outside" / "secret.png")) is False
        assert loads_image("../outside/secret.png") is False
        assert (
            loads_image(os.path.join("..", "allowed", "..", "outside", "secret.png"))
            is False
        )

    def test_refuses_symlink_leaving_base_dir(self, loads_image, image_dir):
        link = image_dir / "allowed" / "link.png"
        try:
            link.symlink_to(image_dir / "outside" / "secret.png")
        except OSError:
            pytest.skip("symlinks are not available")
        assert loads_image("link.png") is False

    def test_requires_base_dir(self, loads_image, image_dir, monkeypatch):
        monkeypatch.setattr(multi_modal_prompt, "ALLOWED_IMAGE_BASE_DIR", None)
        monkeypatch.chdir(image_dir / "allowed")
        assert loads_image("pic.png") is False
