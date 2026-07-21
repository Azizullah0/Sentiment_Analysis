"""
Fetch public YouTube comments and label them with DEEP-Dari.

Usage (from repo root):
  set YOUTUBE_API_KEY=...
  python -m deployment.youtube_batch --video-id VIDEO_ID --max-comments 500
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths import PATHS  # noqa: E402

from deployment.predictor import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    EmotionPredictor,
    OTHERS_LABEL,
)

EMOTION_LABELS = [
    "Hope",
    "Happy",
    "Neutral",
    "Surprise",
    "Disgust",
    "Sad",
    "Anger",
    "Fear",
    OTHERS_LABEL,
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_youtube_client(api_key: str):
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install -r deployment/requirements.txt"
        ) from exc
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def resolve_channel_video_ids(youtube, channel_id: str, max_videos: int) -> List[str]:
    """Resolve uploads playlist videos for a channel."""
    ch = (
        youtube.channels()
        .list(part="contentDetails", id=channel_id)
        .execute()
    )
    items = ch.get("items") or []
    if not items:
        raise ValueError(f"Channel not found: {channel_id}")
    uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    video_ids: List[str] = []
    page_token = None
    while len(video_ids) < max_videos:
        resp = (
            youtube.playlistItems()
            .list(
                part="contentDetails",
                playlistId=uploads,
                maxResults=min(50, max_videos - len(video_ids)),
                pageToken=page_token,
            )
            .execute()
        )
        for item in resp.get("items") or []:
            vid = item["contentDetails"]["videoId"]
            video_ids.append(vid)
            if len(video_ids) >= max_videos:
                break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return video_ids


def fetch_comments_for_video(
    youtube,
    video_id: str,
    max_comments: int,
) -> List[Dict[str, Any]]:
    """Paginate commentThreads.list for one video."""
    rows: List[Dict[str, Any]] = []
    page_token = None
    while len(rows) < max_comments:
        remaining = max_comments - len(rows)
        resp = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, remaining),
                pageToken=page_token,
                textFormat="plainText",
                order="time",
            )
            .execute()
        )
        for item in resp.get("items") or []:
            sn = item["snippet"]["topLevelComment"]["snippet"]
            rows.append(
                {
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "video_id": video_id,
                    "author_channel_id": (sn.get("authorChannelId") or {}).get(
                        "value"
                    ),
                    "published_at": sn.get("publishedAt"),
                    "text_original": sn.get("textDisplay") or sn.get("textOriginal") or "",
                }
            )
            if len(rows) >= max_comments:
                break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return rows


def fetch_all_comments(
    youtube,
    video_ids: Sequence[str],
    max_comments: int,
) -> List[Dict[str, Any]]:
    """Fetch comments across videos until max_comments total."""
    all_rows: List[Dict[str, Any]] = []
    per_video = max(1, max_comments // max(len(video_ids), 1))
    for i, vid in enumerate(video_ids):
        remaining = max_comments - len(all_rows)
        if remaining <= 0:
            break
        # Last video takes the remainder so we can hit the cap
        limit = remaining if i == len(video_ids) - 1 else min(per_video, remaining)
        try:
            batch = fetch_comments_for_video(youtube, vid, limit)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Skipping video {vid}: {exc}")
            continue
        all_rows.extend(batch)
        print(f"  fetched {len(batch)} comments from {vid} (total {len(all_rows)})")
    return all_rows[:max_comments]


def label_comments(
    predictor: EmotionPredictor,
    comments: Sequence[Dict[str, Any]],
    min_confidence: float,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    labeled: List[Dict[str, Any]] = []
    texts = [c["text_original"] for c in comments]
    for start in range(0, len(texts), batch_size):
        chunk_comments = comments[start : start + batch_size]
        chunk_texts = texts[start : start + batch_size]
        preds = predictor.predict(chunk_texts, min_confidence=min_confidence)
        if isinstance(preds, dict):
            preds = [preds]
        for comment, pred in zip(chunk_comments, preds):
            labeled.append(
                {
                    **comment,
                    "text_clean": pred.get("text_clean", ""),
                    "label": pred["label"],
                    "raw_emotion": pred.get("raw_emotion") or "",
                    "confidence": pred["confidence"],
                    "abstain": pred["abstain"],
                    "abstain_reason": pred.get("abstain_reason"),
                }
            )
    return labeled


def write_sqlite(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    conn = sqlite3.connect(path)
    try:
        df.to_sql("labeled_comments", conn, if_exists="replace", index=False)
    finally:
        conn.close()


def build_summary(
    rows: Sequence[Dict[str, Any]],
    min_confidence: float,
    video_ids: Sequence[str],
) -> Dict[str, Any]:
    counts = Counter(r["label"] for r in rows)
    non_others = [r for r in rows if r["label"] != OTHERS_LABEL]
    fear_anger = [
        r for r in non_others if r["label"] in ("Fear", "Anger")
    ]
    confidences = [float(r["confidence"]) for r in rows if r.get("raw_emotion")]
    return {
        "n_comments": len(rows),
        "video_ids": list(video_ids),
        "min_confidence": min_confidence,
        "label_counts": {lab: int(counts.get(lab, 0)) for lab in EMOTION_LABELS},
        "others_rate": (counts.get(OTHERS_LABEL, 0) / len(rows)) if rows else 0.0,
        "mean_confidence_raw": (
            sum(confidences) / len(confidences) if confidences else None
        ),
        "fear_anger_count_non_others": len(fear_anger),
        "fear_anger_rate_non_others": (
            len(fear_anger) / len(non_others) if non_others else 0.0
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label real YouTube comments with DEEP-Dari (Others if uncertain)."
    )
    p.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="YouTube video ID (repeatable).",
    )
    p.add_argument(
        "--channel-id",
        default=None,
        help="Channel ID: fetch comments from recent uploads.",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=5,
        help="Max uploads to pull when using --channel-id (default 5).",
    )
    p.add_argument(
        "--max-comments",
        type=int,
        default=500,
        help="Maximum comments to fetch/label (default 500).",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Override model directory (default: A4 / incremental / seed).",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Below this → label Others (default 0.50).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default outputs/deployment/youtube_batch_<ts>/).",
    )
    p.add_argument(
        "--others-sample-size",
        type=int,
        default=50,
        help="How many Others rows to write to others_sample.csv.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: Set YOUTUBE_API_KEY in the environment "
            "(see deployment/.env.example)."
        )
        return 1

    if not args.video_id and not args.channel_id:
        print("ERROR: Provide --video-id and/or --channel-id.")
        return 1

    youtube = build_youtube_client(api_key)
    video_ids = list(args.video_id)
    if args.channel_id:
        print(f"Resolving uploads for channel {args.channel_id}...")
        video_ids.extend(
            resolve_channel_video_ids(youtube, args.channel_id, args.max_videos)
        )
    # de-dupe preserve order
    seen = set()
    unique_vids = []
    for v in video_ids:
        if v not in seen:
            seen.add(v)
            unique_vids.append(v)
    video_ids = unique_vids

    print(f"Fetching up to {args.max_comments} comments from {len(video_ids)} video(s)...")
    comments = fetch_all_comments(youtube, video_ids, args.max_comments)
    if not comments:
        print("No comments fetched (disabled comments, private video, or empty).")
        return 1

    print(f"Loading model and labeling {len(comments)} comments...")
    predictor = EmotionPredictor(
        model_path=args.model_path,
        min_confidence=args.min_confidence,
    )
    labeled = label_comments(
        predictor,
        comments,
        min_confidence=args.min_confidence,
        batch_size=args.batch_size,
    )

    out_dir = args.out_dir or os.path.join(
        PATHS["deployment_outputs"],
        f"youtube_batch_{_utc_stamp()}",
    )
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(labeled)
    csv_path = os.path.join(out_dir, "labeled_comments.csv")
    sqlite_path = os.path.join(out_dir, "labeled_comments.sqlite")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_sqlite(sqlite_path, labeled)

    summary = build_summary(labeled, args.min_confidence, video_ids)
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    review = df[df["label"].isin(["Fear", "Anger"])].copy()
    review_path = os.path.join(out_dir, "review_candidates.csv")
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    others = df[df["label"] == OTHERS_LABEL].head(args.others_sample_size)
    others_path = os.path.join(out_dir, "others_sample.csv")
    others.to_csv(others_path, index=False, encoding="utf-8-sig")

    # Also append high-confidence Fear/Anger to global review queue
    queue_path = os.path.join(PATHS["deployment_outputs"], "review_queue.jsonl")
    from deployment.predictor import append_review_queue

    n_queued = 0
    for row in labeled:
        if append_review_queue(row, queue_path):
            n_queued += 1

    print(f"\nWrote outputs to: {out_dir}")
    print(f"  labeled_comments.csv  ({len(df)} rows)")
    print(f"  summary.json          Others rate={summary['others_rate']:.3f}")
    print(f"  review_candidates.csv ({len(review)} Fear/Anger)")
    print(f"  review_queue.jsonl    (+{n_queued} lines)")
    print("Label counts:", summary["label_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
