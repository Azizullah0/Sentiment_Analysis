"""
Optional live-chat monitor (run only after youtube_batch works).

Polls liveChatMessages for an active broadcast, labels comments with the same
Others abstention rule, and appends Fear/Anger to the review queue.

Usage:
  set YOUTUBE_API_KEY=...
  python -m deployment.youtube_monitor --video-id LIVE_VIDEO_ID --poll-seconds 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths import PATHS  # noqa: E402

from deployment.predictor import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    EmotionPredictor,
    append_review_queue,
)
from deployment.batch_service import build_youtube_client  # noqa: E402


def get_live_chat_id(youtube, video_id: str) -> str:
    resp = (
        youtube.videos()
        .list(part="liveStreamingDetails", id=video_id)
        .execute()
    )
    items = resp.get("items") or []
    if not items:
        raise ValueError(f"Video not found: {video_id}")
    details = items[0].get("liveStreamingDetails") or {}
    chat_id = details.get("activeLiveChatId")
    if not chat_id:
        raise ValueError(
            "No activeLiveChatId — video may not be a live broadcast in progress."
        )
    return chat_id


def parse_args():
    p = argparse.ArgumentParser(description="Poll YouTube live chat and label comments.")
    p.add_argument("--video-id", required=True, help="Live broadcast video ID.")
    p.add_argument("--poll-seconds", type=int, default=20, help="Seconds between polls.")
    p.add_argument("--model-path", default=None)
    p.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
    )
    p.add_argument(
        "--max-polls",
        type=int,
        default=0,
        help="Stop after N polls (0 = run until Ctrl+C).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        print("ERROR: Set YOUTUBE_API_KEY.")
        return 1

    youtube = build_youtube_client(api_key)
    chat_id = get_live_chat_id(youtube, args.video_id)
    print(f"Live chat id: {chat_id}")

    predictor = EmotionPredictor(
        model_path=args.model_path,
        min_confidence=args.min_confidence,
    )
    queue_path = os.path.join(PATHS["deployment_outputs"], "review_queue.jsonl")
    seen: Set[str] = set()
    page_token: Optional[str] = None
    polls = 0

    print("Monitoring (Ctrl+C to stop). Fear/Anger → review_queue.jsonl")
    try:
        while True:
            resp = (
                youtube.liveChatMessages()
                .list(
                    liveChatId=chat_id,
                    part="snippet,authorDetails",
                    pageToken=page_token,
                )
                .execute()
            )
            texts = []
            meta = []
            for item in resp.get("items") or []:
                cid = item["id"]
                if cid in seen:
                    continue
                seen.add(cid)
                sn = item.get("snippet") or {}
                text = sn.get("displayMessage") or ""
                texts.append(text)
                meta.append(
                    {
                        "comment_id": cid,
                        "video_id": args.video_id,
                        "published_at": sn.get("publishedAt"),
                        "text_original": text,
                    }
                )

            if texts:
                preds = predictor.predict(texts, min_confidence=args.min_confidence)
                if isinstance(preds, dict):
                    preds = [preds]
                for m, pred in zip(meta, preds):
                    row = {**m, **pred}
                    label = pred["label"]
                    print(
                        f"[{label}] conf={pred['confidence']:.2f} "
                        f"raw={pred.get('raw_emotion')} | {m['text_original'][:80]}"
                    )
                    append_review_queue(row, queue_path)

            page_token = resp.get("nextPageToken")
            # API suggests polling interval; respect --poll-seconds as a floor
            wait_ms = int(resp.get("pollingIntervalMillis") or 0)
            sleep_s = max(args.poll_seconds, wait_ms / 1000.0)
            polls += 1
            if args.max_polls and polls >= args.max_polls:
                break
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
