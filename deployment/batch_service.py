"""Shared YouTube batch fetch/label/write + run/job helpers for CLI and API."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from config.paths import PATHS
from deployment.predictor import (
    DEFAULT_MIN_CONFIDENCE,
    EmotionPredictor,
    EXCLUDED_LABEL,
    OTHERS_LABEL,
    append_review_queue,
)

USABLE_LABELS = [
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

EMOTION_LABELS = USABLE_LABELS + [EXCLUDED_LABEL]

LogFn = Callable[[str], None]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def deployment_root() -> str:
    root = PATHS.get("deployment_outputs") or os.path.join("outputs", "deployment")
    os.makedirs(root, exist_ok=True)
    return root


def build_youtube_client(api_key: str):
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency. Install with: pip install -r deployment/requirements.txt"
        ) from exc
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def resolve_channel_video_ids(youtube, channel_id: str, max_videos: int) -> List[str]:
    ch = youtube.channels().list(part="contentDetails", id=channel_id).execute()
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
            video_ids.append(item["contentDetails"]["videoId"])
            if len(video_ids) >= max_videos:
                break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return video_ids


def fetch_comments_for_video(
    youtube, video_id: str, max_comments: int
) -> List[Dict[str, Any]]:
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
                    "author_channel_id": (sn.get("authorChannelId") or {}).get("value"),
                    "published_at": sn.get("publishedAt"),
                    "text_original": sn.get("textDisplay")
                    or sn.get("textOriginal")
                    or "",
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
    log: Optional[LogFn] = None,
) -> List[Dict[str, Any]]:
    _log = log or print
    all_rows: List[Dict[str, Any]] = []
    per_video = max(1, max_comments // max(len(video_ids), 1))
    for i, vid in enumerate(video_ids):
        remaining = max_comments - len(all_rows)
        if remaining <= 0:
            break
        limit = remaining if i == len(video_ids) - 1 else min(per_video, remaining)
        try:
            batch = fetch_comments_for_video(youtube, vid, limit)
        except Exception as exc:  # noqa: BLE001
            _log(f"[WARN] Skipping video {vid}: {exc}")
            continue
        all_rows.extend(batch)
        _log(f"  fetched {len(batch)} comments from {vid} (total {len(all_rows)})")
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
    counts_all = Counter(r["label"] for r in rows)
    usable = [r for r in rows if r["label"] != EXCLUDED_LABEL]
    counts_usable = Counter(r["label"] for r in usable)
    n_fetched = len(rows)
    n_excluded = int(counts_all.get(EXCLUDED_LABEL, 0))
    n_usable = len(usable)
    n_others = int(counts_usable.get(OTHERS_LABEL, 0))
    non_others_usable = [r for r in usable if r["label"] != OTHERS_LABEL]
    fear_anger = [r for r in non_others_usable if r["label"] in ("Fear", "Anger")]
    confidences = [
        float(r["confidence"])
        for r in usable
        if r.get("raw_emotion")
    ]
    exclusion_reasons = Counter(
        r.get("abstain_reason") or "unknown"
        for r in rows
        if r["label"] == EXCLUDED_LABEL
    )
    per_video: Dict[str, int] = Counter(r["video_id"] for r in rows)
    label_counts_usable = {
        lab: int(counts_usable.get(lab, 0)) for lab in USABLE_LABELS
    }
    label_counts_all = {
        lab: int(counts_all.get(lab, 0)) for lab in EMOTION_LABELS
    }
    return {
        "n_comments": n_fetched,
        "n_fetched": n_fetched,
        "n_excluded": n_excluded,
        "n_usable": n_usable,
        "n_others": n_others,
        "exclusion_rate": (n_excluded / n_fetched) if n_fetched else 0.0,
        "others_rate_among_usable": (n_others / n_usable) if n_usable else 0.0,
        # Legacy field: among all fetched (includes Excluded as non-Others)
        "others_rate": (n_others / n_fetched) if n_fetched else 0.0,
        "video_ids": list(video_ids),
        "min_confidence": min_confidence,
        "label_counts": label_counts_usable,
        "label_counts_usable": label_counts_usable,
        "label_counts_all": label_counts_all,
        "exclusion_reasons": dict(exclusion_reasons),
        "mean_confidence_raw": (
            sum(confidences) / len(confidences) if confidences else None
        ),
        "fear_anger_count_non_others": len(fear_anger),
        "fear_anger_rate_non_others": (
            len(fear_anger) / len(non_others_usable) if non_others_usable else 0.0
        ),
        "comments_per_video": dict(per_video),
    }


def remap_legacy_others_to_excluded(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map old Others+unusable_text rows to Excluded without re-running the model."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        label = row.get("label")
        reason = row.get("abstain_reason")
        if label == OTHERS_LABEL and reason in (
            "unusable_text",
            "empty",
            "emoji_only",
            "too_short",
            "non_persian",
        ):
            row["label"] = EXCLUDED_LABEL
            row["raw_emotion"] = ""
            row["confidence"] = 0.0
            row["abstain"] = True
            if reason == "unusable_text":
                row["abstain_reason"] = "non_persian"
        out.append(row)
    return out


def reaggregate_run_dir(path: str, rewrite_csv: bool = True) -> Dict[str, Any]:
    """Refresh summary.json (and optionally CSV) for an existing batch folder."""
    csv_path = os.path.join(path, "labeled_comments.csv")
    sqlite_path = os.path.join(path, "labeled_comments.sqlite")
    if os.path.isfile(sqlite_path):
        conn = sqlite3.connect(sqlite_path)
        try:
            df = pd.read_sql_query("SELECT * FROM labeled_comments", conn)
        finally:
            conn.close()
    elif os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No labeled comments in {path}")

    rows = df.where(pd.notnull(df), None).to_dict(orient="records")
    rows = remap_legacy_others_to_excluded(rows)

    summary_path = os.path.join(path, "summary.json")
    old = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            old = json.load(f)
    video_ids = old.get("video_ids") or sorted(
        {r["video_id"] for r in rows if r.get("video_id")}
    )
    min_confidence = float(old.get("min_confidence") or DEFAULT_MIN_CONFIDENCE)
    summary = build_summary(rows, min_confidence, video_ids)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if rewrite_csv:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        write_sqlite(sqlite_path, rows)

    return summary


def dedupe_video_ids(video_ids: Sequence[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for v in video_ids:
        if v and v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def run_batch(
    *,
    api_key: str,
    video_ids: Optional[Sequence[str]] = None,
    channel_id: Optional[str] = None,
    max_videos: int = 5,
    max_comments: int = 500,
    model_path: Optional[str] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    batch_size: int = 32,
    out_dir: Optional[str] = None,
    others_sample_size: int = 50,
    log: Optional[LogFn] = None,
) -> Dict[str, Any]:
    """Fetch, label, and write a youtube_batch_* folder. Returns result dict."""
    _log = log or print
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is required")
    video_ids = list(video_ids or [])
    if not video_ids and not channel_id:
        raise ValueError("Provide video_ids and/or channel_id")

    youtube = build_youtube_client(api_key)
    if channel_id:
        _log(f"Resolving uploads for channel {channel_id}...")
        video_ids.extend(resolve_channel_video_ids(youtube, channel_id, max_videos))
    video_ids = dedupe_video_ids(video_ids)
    if not video_ids:
        raise ValueError("No video IDs resolved")

    _log(f"Fetching up to {max_comments} comments from {len(video_ids)} video(s)...")
    comments = fetch_all_comments(youtube, video_ids, max_comments, log=_log)
    if not comments:
        raise RuntimeError(
            "No comments fetched (disabled comments, private video, or empty)."
        )

    _log(f"Loading model and labeling {len(comments)} comments...")
    predictor = EmotionPredictor(
        model_path=model_path, min_confidence=min_confidence
    )
    labeled = label_comments(
        predictor, comments, min_confidence=min_confidence, batch_size=batch_size
    )

    out_dir = out_dir or os.path.join(
        deployment_root(), f"youtube_batch_{utc_stamp()}"
    )
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(labeled)
    csv_path = os.path.join(out_dir, "labeled_comments.csv")
    sqlite_path = os.path.join(out_dir, "labeled_comments.sqlite")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_sqlite(sqlite_path, labeled)

    summary = build_summary(labeled, min_confidence, video_ids)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    review = df[df["label"].isin(["Fear", "Anger"])].copy()
    review.to_csv(
        os.path.join(out_dir, "review_candidates.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    others = df[df["label"] == OTHERS_LABEL].head(others_sample_size)
    others.to_csv(
        os.path.join(out_dir, "others_sample.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    queue_path = os.path.join(deployment_root(), "review_queue.jsonl")
    n_queued = sum(1 for row in labeled if append_review_queue(row, queue_path))

    _log(f"Wrote outputs to: {out_dir}")
    _log(f"  labeled_comments.csv  ({len(df)} rows)")
    _log(
        f"  summary.json          excluded={summary['exclusion_rate']:.3f} "
        f"others_among_usable={summary['others_rate_among_usable']:.3f}"
    )
    _log(f"  review_candidates.csv ({len(review)} Fear/Anger)")
    _log(f"  review_queue.jsonl    (+{n_queued} lines)")

    return {
        "out_dir": out_dir,
        "run_id": os.path.basename(out_dir),
        "summary": summary,
        "n_queued": n_queued,
        "n_review": int(len(review)),
    }


# ----- Run browsing helpers -----


def list_runs() -> List[Dict[str, Any]]:
    root = deployment_root()
    runs = []
    if not os.path.isdir(root):
        return runs
    for name in sorted(os.listdir(root), reverse=True):
        if not name.startswith("youtube_batch_"):
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        summary_path = os.path.join(path, "summary.json")
        summary = {}
        if os.path.isfile(summary_path):
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
        mtime = os.path.getmtime(path)
        runs.append(
            {
                "run_id": name,
                "path": path,
                "mtime": mtime,
                "mtime_iso": datetime.fromtimestamp(
                    mtime, tz=timezone.utc
                ).isoformat(),
                "n_comments": summary.get("n_comments", 0),
                "n_usable": summary.get("n_usable"),
                "n_excluded": summary.get("n_excluded"),
                "exclusion_rate": summary.get("exclusion_rate"),
                "others_rate": summary.get("others_rate"),
                "others_rate_among_usable": summary.get("others_rate_among_usable"),
                "fear_anger_count": summary.get("fear_anger_count_non_others", 0),
                "label_counts": summary.get("label_counts", {}),
                "label_counts_usable": summary.get("label_counts_usable"),
                "label_counts_all": summary.get("label_counts_all"),
                "video_ids": summary.get("video_ids", []),
                "min_confidence": summary.get("min_confidence"),
            }
        )
    return runs


def run_dir(run_id: str) -> str:
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("Invalid run_id")
    path = os.path.join(deployment_root(), run_id)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Run not found: {run_id}")
    return path


def load_run_detail(run_id: str) -> Dict[str, Any]:
    path = run_dir(run_id)
    summary_path = os.path.join(path, "summary.json")
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    return {
        "run_id": run_id,
        "path": path,
        "summary": summary,
        "has_csv": os.path.isfile(os.path.join(path, "labeled_comments.csv")),
        "has_sqlite": os.path.isfile(os.path.join(path, "labeled_comments.sqlite")),
        "csv_url": f"/api/runs/{run_id}/export.csv",
    }


def _load_comments_df(run_id: str) -> pd.DataFrame:
    path = run_dir(run_id)
    sqlite_path = os.path.join(path, "labeled_comments.sqlite")
    csv_path = os.path.join(path, "labeled_comments.csv")
    if os.path.isfile(sqlite_path):
        conn = sqlite3.connect(sqlite_path)
        try:
            return pd.read_sql_query("SELECT * FROM labeled_comments", conn)
        finally:
            conn.close()
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No comments file for run {run_id}")


def query_comments(
    run_id: str,
    *,
    label: Optional[str] = None,
    video_id: Optional[str] = None,
    abstain: Optional[bool] = None,
    usable_only: bool = False,
    q: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
) -> Dict[str, Any]:
    df = _load_comments_df(run_id)
    if usable_only:
        df = df[df["label"] != EXCLUDED_LABEL]
    if label:
        df = df[df["label"] == label]
    if video_id:
        df = df[df["video_id"] == video_id]
    if abstain is not None and "abstain" in df.columns:
        def _as_bool(v):
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in ("true", "1", "yes")

        df = df[df["abstain"].map(_as_bool) == bool(abstain)]
    if q:
        mask = df["text_original"].astype(str).str.contains(q, case=False, na=False)
        if "text_clean" in df.columns:
            mask = mask | df["text_clean"].astype(str).str.contains(
                q, case=False, na=False
            )
        df = df[mask]
    total = int(len(df))
    page = df.iloc[offset : offset + limit]
    records = page.where(pd.notnull(page), None).to_dict(orient="records")
    # normalize types for JSON
    for r in records:
        if "abstain" in r and r["abstain"] is not None:
            r["abstain"] = bool(r["abstain"]) if not isinstance(r["abstain"], str) else r["abstain"] in (
                "True",
                "true",
                "1",
            )
        if "confidence" in r and r["confidence"] is not None:
            r["confidence"] = float(r["confidence"])
    return {"total": total, "offset": offset, "limit": limit, "items": records}


def load_review_candidates(run_id: str) -> List[Dict[str, Any]]:
    path = run_dir(run_id)
    review_path = os.path.join(path, "review_candidates.csv")
    if os.path.isfile(review_path):
        df = pd.read_csv(review_path)
    else:
        result = query_comments(run_id, limit=100000)
        items = [
            i
            for i in result["items"]
            if i.get("label") in ("Fear", "Anger")
        ]
        return items
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    ann_path = os.path.join(path, "review_annotations.json")
    annotations = {}
    if os.path.isfile(ann_path):
        with open(ann_path, encoding="utf-8") as f:
            annotations = json.load(f)
    for r in records:
        cid = r.get("comment_id")
        r["checked"] = bool(annotations.get(cid, {}).get("checked", False))
        r["note"] = annotations.get(cid, {}).get("note")
        if r.get("confidence") is not None:
            r["confidence"] = float(r["confidence"])
    return records


def save_review_annotations(
    run_id: str, updates: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    path = run_dir(run_id)
    ann_path = os.path.join(path, "review_annotations.json")
    annotations = {}
    if os.path.isfile(ann_path):
        with open(ann_path, encoding="utf-8") as f:
            annotations = json.load(f)
    for cid, payload in updates.items():
        cur = annotations.get(cid, {})
        cur.update(payload)
        annotations[cid] = cur
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    return annotations


def load_review_queue(limit: int = 200) -> List[Dict[str, Any]]:
    path = os.path.join(deployment_root(), "review_queue.jsonl")
    if not os.path.isfile(path):
        return []
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    items = []
    for line in lines[-limit:]:
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    items.reverse()
    return items


# ----- Async jobs -----

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def list_jobs() -> List[Dict[str, Any]]:
    with _jobs_lock:
        return [dict(j) for j in sorted(_jobs.values(), key=lambda x: x["created_at"], reverse=True)]


def start_batch_job(
    *,
    video_ids: Optional[List[str]] = None,
    channel_id: Optional[str] = None,
    max_videos: int = 5,
    max_comments: int = 500,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    from .youtube_urls import normalize_job_inputs

    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is not set on the server")

    youtube = build_youtube_client(api_key)
    video_ids, channel_id = normalize_job_inputs(video_ids, channel_id, youtube)

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "logs": [],
        "error": None,
        "run_id": None,
        "out_dir": None,
        "summary": None,
        "params": {
            "video_ids": video_ids or [],
            "channel_id": channel_id,
            "max_videos": max_videos,
            "max_comments": max_comments,
            "min_confidence": min_confidence,
        },
    }
    with _jobs_lock:
        _jobs[job_id] = job

    def _log(msg: str) -> None:
        with _jobs_lock:
            _jobs[job_id]["logs"].append(msg)
            # keep last 200 lines
            _jobs[job_id]["logs"] = _jobs[job_id]["logs"][-200:]

    def _worker() -> None:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
        try:
            result = run_batch(
                api_key=api_key,
                video_ids=video_ids,
                channel_id=channel_id,
                max_videos=max_videos,
                max_comments=max_comments,
                min_confidence=min_confidence,
                model_path=model_path or os.environ.get("DEPLOYMENT_MODEL_PATH"),
                log=_log,
            )
            with _jobs_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["run_id"] = result["run_id"]
                _jobs[job_id]["out_dir"] = result["out_dir"]
                _jobs[job_id]["summary"] = result["summary"]
        except Exception as exc:  # noqa: BLE001
            _log(traceback.format_exc())
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(exc)

    threading.Thread(target=_worker, daemon=True).start()
    return dict(job)
