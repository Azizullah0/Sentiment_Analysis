"""Lightweight tests for YouTube URL parsing (no API key required)."""

from deployment.youtube_urls import (
    extract_channel_handle,
    extract_channel_id,
    extract_legacy_channel_slug,
    extract_video_id,
    normalize_job_inputs,
)


def test_extract_video_id_forms():
    assert extract_video_id("16e75OffBTA") == "16e75OffBTA"
    assert (
        extract_video_id("https://www.youtube.com/watch?v=16e75OffBTA")
        == "16e75OffBTA"
    )
    assert extract_video_id("https://youtu.be/16e75OffBTA") == "16e75OffBTA"
    assert (
        extract_video_id("https://www.youtube.com/shorts/16e75OffBTA")
        == "16e75OffBTA"
    )


def test_extract_channel_forms():
    cid = "UCabcdefghijklmnopqrstuv"
    assert extract_channel_id(cid) == cid
    assert extract_channel_id(f"https://www.youtube.com/channel/{cid}") == cid
    assert extract_channel_handle("@SomeHandle") == "SomeHandle"
    assert (
        extract_channel_handle("https://www.youtube.com/@SomeHandle")
        == "SomeHandle"
    )
    assert extract_legacy_channel_slug("https://www.youtube.com/c/Name") == (
        "c",
        "Name",
    )


def test_normalize_videos_only():
    ids, ch = normalize_job_inputs(
        ["https://youtu.be/16e75OffBTA", "16e75OffBTA"],
        None,
        youtube=None,
    )
    assert ids == ["16e75OffBTA"]
    assert ch is None


def test_normalize_rejects_bad_video():
    try:
        # Must not be exactly 11 [A-Za-z0-9_-] chars (bare ids are accepted as-is)
        normalize_job_inputs(["totally-invalid-youtube-input"], None, youtube=None)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Could not parse" in str(exc)
