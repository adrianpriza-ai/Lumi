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


@pytest.mark.parametrize(
    "url",
    [
        # UTS46 maps the ideographic full stop to '.', so a fetcher sees
        # 127.0.0.1 while urlsplit's hostname is not an ASCII literal. A
        # resolver without IDN support (musl) never resolved it at all.
        "http://127\u30020\u30020\u30021/",
        # WHATWG parsers read the authority up to the backslash as userinfo,
        # so the fetcher connects to 127.0.0.1 while urlsplit reports
        # example.com — a parser differential, refused rather than guessed at.
        "http://127.0.0.1\\@example.com/",
    ],
)
def test_parser_differential_hosts_are_refused(url: str) -> None:
    assert validate_fetch_url(url) is not None


def test_an_idn_host_is_checked_as_the_punycode_the_fetcher_resolves() -> None:
    # 'münchen.example' encodes to an ASCII label before the host checks run;
    # the example TLD is not resolvable here either, which is not a refusal.
    assert validate_fetch_url("https://münchen.example/") is None


def test_long_hostname_label_does_not_crash_resolver() -> None:
    # A label exceeding 63 ASCII characters causes socket.getaddrinfo to raise
    # UnicodeEncodeError via idna encoding; this should be caught cleanly rather
    # than raising an unhandled exception (R21).
    assert validate_fetch_url("https://" + "a" * 64 + ".example.com/") is None
