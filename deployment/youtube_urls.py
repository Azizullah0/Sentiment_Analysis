"""Parse YouTube video/channel URLs and resolve handles to UC… IDs."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

# Standard YouTube video id length
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")
_HANDLE_RE = re.compile(r"^@([\w.-]+)$", re.UNICODE)


def extract_video_id(text: str) -> Optional[str]:
    """Extract an 11-char video id from a URL or bare id."""
    raw = (text or "").strip()
    if not raw:
        return None
    if _VIDEO_ID_RE.match(raw):
        return raw

    # Allow pasting with surrounding whitespace / accidental quotes
    raw = raw.strip("\"'")

    try:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    except Exception:  # noqa: BLE001
        return None

    host = (parsed.netloc or "").lower().replace("www.", "")
    path = parsed.path or ""

    if host in ("youtu.be", "www.youtu.be"):
        candidate = path.strip("/").split("/")[0]
        return candidate if _VIDEO_ID_RE.match(candidate) else None

    if "youtube.com" in host or host == "m.youtube.com" or host.endswith(".youtube.com"):
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            candidate = qs["v"][0]
            return candidate if _VIDEO_ID_RE.match(candidate) else None
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            candidate = parts[1]
            return candidate if _VIDEO_ID_RE.match(candidate) else None

    return None


def extract_channel_id(text: str) -> Optional[str]:
    """Extract UC… from a /channel/UC… URL or bare channel id."""
    raw = (text or "").strip().strip("\"'")
    if not raw:
        return None
    if _CHANNEL_ID_RE.match(raw):
        return raw

    try:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    except Exception:  # noqa: BLE001
        return None

    path = parsed.path or ""
    m = re.search(r"/channel/(UC[A-Za-z0-9_-]{22})", path)
    if m:
        return m.group(1)
    return None


def extract_channel_handle(text: str) -> Optional[str]:
    """Return handle without @, from URL or bare @handle."""
    raw = (text or "").strip().strip("\"'")
    if not raw:
        return None

    bare = _HANDLE_RE.match(raw)
    if bare:
        return bare.group(1)

    try:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    except Exception:  # noqa: BLE001
        return None

    path = parsed.path or ""
    m = re.search(r"/@([\w.-]+)", path)
    if m:
        return m.group(1)
    return None


def extract_legacy_channel_slug(text: str) -> Optional[Tuple[str, str]]:
    """Return ('c'|'user', slug) for /c/Name or /user/Name URLs."""
    raw = (text or "").strip().strip("\"'")
    if not raw:
        return None
    try:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    except Exception:  # noqa: BLE001
        return None
    path = parsed.path or ""
    m = re.search(r"/(c|user)/([^/?#]+)", path)
    if m:
        return m.group(1), m.group(2)
    return None


def resolve_channel_id(youtube: Any, text: str) -> str:
    """
    Resolve channel paste to a UC… id.
    Uses URL/id extractors first; then forHandle / search via the API.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Channel field is empty")

    direct = extract_channel_id(raw)
    if direct:
        return direct

    handle = extract_channel_handle(raw)
    if handle:
        resp = (
            youtube.channels()
            .list(part="id", forHandle=handle)
            .execute()
        )
        items = resp.get("items") or []
        if items:
            return items[0]["id"]
        raise ValueError(f"Could not resolve YouTube handle @{handle} to a channel id")

    legacy = extract_legacy_channel_slug(raw)
    if legacy:
        kind, slug = legacy
        query = slug.replace("_", " ")
        resp = (
            youtube.search()
            .list(part="snippet", type="channel", q=query, maxResults=5)
            .execute()
        )
        items = resp.get("items") or []
        if not items:
            raise ValueError(f"Could not resolve YouTube /{kind}/{slug} to a channel id")
        # Prefer exact customUrl / title match when possible
        slug_l = slug.lower()
        for item in items:
            sn = item.get("snippet") or {}
            custom = (sn.get("customUrl") or "").lstrip("@").lower()
            title = (sn.get("title") or "").replace(" ", "").lower()
            if custom == slug_l or title == slug_l.replace("_", ""):
                return item["snippet"]["channelId"]
        return items[0]["snippet"]["channelId"]

    raise ValueError(
        "Unrecognized channel input. Paste a channel URL, @handle, or UC… id."
    )


def normalize_job_inputs(
    video_lines: Optional[Sequence[str]],
    channel_text: Optional[str],
    youtube: Any = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Normalize mixed URL/id inputs to (video_ids, channel_id).
    youtube client is required when channel_text needs API resolution.
    """
    video_ids: List[str] = []
    errors: List[str] = []
    for line in video_lines or []:
        token = (line or "").strip()
        if not token:
            continue
        vid = extract_video_id(token)
        if vid:
            if vid not in video_ids:
                video_ids.append(vid)
        else:
            errors.append(f"Could not parse video id from: {token[:80]}")

    channel_id: Optional[str] = None
    channel_raw = (channel_text or "").strip()
    if channel_raw:
        if youtube is None and not extract_channel_id(channel_raw):
            raise ValueError(
                "Channel URL/@handle requires a YouTube API client to resolve"
            )
        if extract_channel_id(channel_raw):
            channel_id = extract_channel_id(channel_raw)
        else:
            if youtube is None:
                raise ValueError("YouTube client required to resolve channel handle/URL")
            channel_id = resolve_channel_id(youtube, channel_raw)

    if errors:
        raise ValueError("; ".join(errors))
    if not video_ids and not channel_id:
        raise ValueError("Provide at least one video URL/id and/or a channel URL/@handle/UC…")

    return video_ids, channel_id
