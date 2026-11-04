"""SSRF regression tests for MultiModalFaithfulness image fetching.

Covers CVE-2026-6587 / GHSA-95ww-475f-pr4f: a caller-supplied
``retrieved_contexts`` URL must never cause the server to fetch a non-public
(internal) address, directly or via redirect, and a configured proxy must not
silently void the guard.

These tests also pin the behaviour of the urllib3 connection internals the
guard relies on, so a future urllib3 refactor that broke the guard would fail
here rather than silently.
"""

import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from ragas.metrics.collections.multi_modal_faithfulness.util import (
    _ip_is_public,
    _SSRFGuardAdapter,
    _BlockedAddressError,
    _try_process_url,
)


def _valid_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def loopback_image_server():
    """A real HTTP server on 127.0.0.1 that serves a valid PNG.

    It serves a genuine image so that a blocked fetch proves the SSRF guard
    fired, not merely that an image failed to decode. ``/redirect`` 302s to the
    image to exercise the per-hop check.
    """
    png = _valid_png_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence test output
            pass

        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/image.png")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()


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
            "0.0.0.0",
            "::1",
            "fc00::1",
            "fe80::1",
            "::ffff:169.254.169.254",  # IPv4-mapped metadata
            "::ffff:10.0.0.1",
        ],
    )
    def test_non_public_addresses_are_rejected(self, ip):
        assert _ip_is_public(ip) is False

    @pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
    def test_public_addresses_are_allowed(self, ip):
        assert _ip_is_public(ip) is True

    def test_unparseable_is_rejected(self):
        assert _ip_is_public("not-an-ip") is False


class TestUrlFetchSSRF:
    """Integration tests against a real loopback server."""

    def test_direct_loopback_is_blocked(self, loopback_image_server):
        url = f"http://127.0.0.1:{loopback_image_server}/image.png"
        assert _try_process_url(url) is None

    def test_loopback_by_name_is_blocked(self, loopback_image_server):
        url = f"http://localhost:{loopback_image_server}/image.png"
        assert _try_process_url(url) is None

    def test_redirect_to_loopback_is_blocked(self, loopback_image_server):
        url = f"http://127.0.0.1:{loopback_image_server}/redirect"
        assert _try_process_url(url) is None

    def test_guard_is_the_only_gate(self, loopback_image_server, monkeypatch):
        """With the peer treated as public, the same fetch succeeds.

        This proves the block is the SSRF guard and that the download/redirect
        pipeline otherwise works end to end.
        """
        import ragas.metrics.collections.multi_modal_faithfulness.util as util

        real = util._ip_is_public
        monkeypatch.setattr(
            util,
            "_ip_is_public",
            lambda ip: True if ip.startswith("127.") else real(ip),
        )
        url = f"http://127.0.0.1:{loopback_image_server}/image.png"
        result = _try_process_url(url)
        assert result is not None
        assert result["mime_type"] == "image/png"
        assert result["encoded_data"]

    def test_env_proxy_does_not_bypass_guard(
        self, loopback_image_server, monkeypatch
    ):
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
        url = f"http://127.0.0.1:{loopback_image_server}/image.png"
        assert _try_process_url(url) is None


class TestProxyRefused:
    def test_proxy_manager_is_refused(self):
        adapter = _SSRFGuardAdapter(max_retries=0)
        with pytest.raises(_BlockedAddressError):
            adapter.proxy_manager_for("http://127.0.0.1:1")
