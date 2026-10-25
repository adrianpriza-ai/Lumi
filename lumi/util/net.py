"""URL validation for the tools that fetch remote content.

``web.fetch`` hands an arbitrary model-supplied URL to a provider to read, and
``ocr`` can hand one to an OCR API. Before either dispatch, the URL is checked
here so the process (or whatever fetches on its behalf) is not pointed at a
loopback, link-local, metadata or RFC1918 address — the SSRF shapes that reach
host-local services, the cloud metadata endpoint (169.254.169.254) or the rest
of a private network.

The check is deliberately a property of the URL's *host*, so nothing here needs
to know which provider answers. ``validate_fetch_url`` returns ``None`` when the
URL is acceptable and a human-readable reason when it is not; callers turn the
reason into a ``ToolError`` / failed result of their own.

Two caveats the model of the deployment should keep in mind, noted in the audit:

* Firecrawl, Exa, Tavily and You.com fetch on *their* infrastructure, so a
  blocked URL would have been requested by someone else's box. Blocking it
  still prevents the metadata-steal and saves their eg&nbsp;/&nbsp;billing blast
  radius, but the SSRF target is not Lumi's own host when those providers
  answer.
* A local MCP server (``npx -y some-server``) fetches inside the deployment.
  Hostname resolution here is a best-effort hint only: DNS may be answered
  differently by the fetcher, so the authoritative guard for that path is the
  address check on literal IPs and the hostname blocklist below.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

#: Hostnames that always mean "somewhere on this machine or the local subnet",
#: matched case-insensitively; ``*.local`` is mDNS, ``*.localhost`` and
#: ``*.localdomain`` resolve on most Unix systems.
_PRIVATE_HOSTLETS = (
    "localhost",
    "localdomain",
    "local",
    "internal",
)

#: Public hosts that are never fetching a document — see the module docstring.
_RESERVED_HOSTNAMES = frozenset({"0.0.0.0"})


def _host_reason(host: str) -> str | None:
    """A reason *host* is refused, else ``None``.

    *host* may be a literal IP or a name. Names are checked against the
    blocklist and, as a best-effort hint, resolved and rejected when every
    address answers to a non-globally-routable IP.
    """
    lowered = host.lower().strip("[]").rstrip(".")
    if lowered in _RESERVED_HOSTNAMES:
        return "unspecified address"
    for suffix in _PRIVATE_HOSTLETS:
        if lowered == suffix or lowered.endswith("." + suffix):
            return "loopback/local hostname"

    # A literal IP gets the authoritative check: refuse unless globally reachable.
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        pass
    else:
        return None if ip.is_global else f"non-routable address ({ip})"

    # A name resolves inside our process, not inside the fetcher; treat a
    # rejection here as a warning-grade guard rather than proof.
    try:
        addrs = socket.getaddrinfo(lowered, None)
    except OSError:
        return None  # unresolvable here does not mean malformed
    within = [a[4][0] for a in addrs]
    for text in within:
        try:
            ip = ipaddress.ip_address(text.split("%")[0])
        except ValueError:
            continue
        if not ip.is_global:
            return f"resolves to non-routable address ({text})"
    return None


def validate_fetch_url(url: str) -> str | None:
    """Check an http(s) URL before it is dispatched to a fetcher.

    Returns ``None`` if *url* is fine to fetch, otherwise a short human
    reason. Only ``http``/``https`` are considered: anything else is left to
    the caller's scheme check (the tools enforce the prefix before this).
    """
    try:
        parts = urlsplit(url)
    except ValueError as exc:  # pragma: no cover - urlsplit is lenient
        return f"malformed URL: {exc}"
    scheme = (parts.scheme or "").lower()
    if scheme not in ("http", "https"):
        return f"unsupported scheme {scheme or '(none)'!r}"
    host = parts.hostname
    if not host:
        return "missing hostname"
    return _host_reason(host)


__all__ = ["validate_fetch_url"]