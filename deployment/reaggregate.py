"""Refresh summary.json for existing youtube_batch_* folders (legacy Others→Excluded)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deployment.batch_service import deployment_root, reaggregate_run_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reaggregate deployment batch summaries (remap unusable Others → Excluded)."
    )
    parser.add_argument(
        "--run-id",
        help="Single run folder name (youtube_batch_...). Default: all under deployment outputs.",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Only rewrite summary.json; leave CSV/SQLite unchanged.",
    )
    args = parser.parse_args()

    root = deployment_root()
    if args.run_id:
        paths = [os.path.join(root, args.run_id)]
    else:
        paths = [
            os.path.join(root, name)
            for name in sorted(os.listdir(root))
            if name.startswith("youtube_batch_")
            and os.path.isdir(os.path.join(root, name))
        ]

    for path in paths:
        if not os.path.isdir(path):
            print(f"[SKIP] not found: {path}")
            continue
        summary = reaggregate_run_dir(path, rewrite_csv=not args.no_rewrite)
        print(
            f"{os.path.basename(path)}: "
            f"excluded={summary['exclusion_rate']:.3f} "
            f"others_usable={summary['others_rate_among_usable']:.3f} "
            f"n={summary['n_fetched']}"
        )


if __name__ == "__main__":
    main()
