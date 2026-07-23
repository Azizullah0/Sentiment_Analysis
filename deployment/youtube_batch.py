"""
Fetch public YouTube comments and label them with DEEP-Dari.

Usage (from repo root):
  export YOUTUBE_API_KEY=...
  python -m deployment.youtube_batch --video-id VIDEO_ID --max-comments 500
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deployment.batch_service import (  # noqa: E402
    build_youtube_client,
    run_batch,
)
from deployment.predictor import DEFAULT_MIN_CONFIDENCE  # noqa: E402

# Re-export for youtube_monitor and external imports
__all__ = ["build_youtube_client", "main", "parse_args"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label real YouTube comments with DEEP-Dari (Others if uncertain)."
    )
    p.add_argument("--video-id", action="append", default=[], help="YouTube video ID.")
    p.add_argument("--channel-id", default=None, help="Channel UC… ID.")
    p.add_argument("--max-videos", type=int, default=5)
    p.add_argument("--max-comments", type=int, default=500)
    p.add_argument("--model-path", default=None)
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--others-sample-size", type=int, default=50)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        print("ERROR: Set YOUTUBE_API_KEY (see deployment/.env.example).")
        return 1
    if not args.video_id and not args.channel_id:
        print("ERROR: Provide --video-id and/or --channel-id.")
        return 1
    try:
        result = run_batch(
            api_key=api_key,
            video_ids=args.video_id,
            channel_id=args.channel_id,
            max_videos=args.max_videos,
            max_comments=args.max_comments,
            model_path=args.model_path,
            min_confidence=args.min_confidence,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
            others_sample_size=args.others_sample_size,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        return 1
    print("Label counts:", result["summary"].get("label_counts"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
