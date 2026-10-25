"""Unit tests for URL validation that guards the fetch/OCR paths."""

from __future__ import annotations

import pytest

from lumi.util.net import validate_fetch_url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "http://example.com/a?b=c",
        "https://example.com:8443/path",
        "http://8.8.8.8/x",  # globally routable literal
        "http://[2606:4700::1111]/",
        "https://sub.domain.example.org/docs",
    ],
)
def test_globally_reachable_urls_pass(url: str) -> None:
    assert validate_fetch_url(url) is None


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "//example.com",  # no scheme
        "http://",  # no host
    ],
)
def test_bad_scheme_or_missing_host(url: str) -> None:
    assert validate_fetch_url(url) is not None


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",  # loopback
        "10.0.0.5",  # RFC1918
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "0.0.0.0",
        "[::1]",
        "[::ffff:127.0.0.1]",  # IPv4-mapped loopback
        "[::ffff:169.254.169.254]",  # IPv4-mapped metadata
        "[fd00::1]",  # IPv6 unique-local
        "[fe80::1]",  # IPv6 link-local
    ],
)
def test_non_global_literal_ips_are_refused(host: str) -> None:
    assert validate_fetch_url(f"http://{host}/") is not None


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "foo.localhost",
        "meta.localhost",
        "printer.local",
        "api.internal",
        "db.localdomain",
    ],
)
def test_private_hostnames_are_refused(host: str) -> None:
    assert validate_fetch_url(f"https://{host}/x") is not None