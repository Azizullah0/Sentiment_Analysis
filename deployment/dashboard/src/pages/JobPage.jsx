import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

/** Client-side video id extract (regex only; mirrors deployment/youtube_urls.py). */
function extractVideoId(text) {
  const raw = (text || "").trim().replace(/^["']|["']$/g, "");
  if (!raw) return null;
  if (/^[A-Za-z0-9_-]{11}$/.test(raw)) return raw;

  let url;
  try {
    url = new URL(raw.includes("://") ? raw : `https://${raw}`);
  } catch {
    return null;
  }
  const host = url.hostname.replace(/^www\./, "").toLowerCase();
  if (host === "youtu.be") {
    const id = url.pathname.replace(/^\//, "").split("/")[0];
    return /^[A-Za-z0-9_-]{11}$/.test(id) ? id : null;
  }
  if (host.includes("youtube.com")) {
    const v = url.searchParams.get("v");
    if (v && /^[A-Za-z0-9_-]{11}$/.test(v)) return v;
    const m = url.pathname.match(/^\/(shorts|embed|live|v)\/([A-Za-z0-9_-]{11})/);
    if (m) return m[2];
  }
  return null;
}

function tokenizeVideoInput(text) {
  return text
    .split(/[\n,]+/)
    .flatMap((line) => line.trim().split(/\s+/))
    .map((s) => s.trim())
    .filter(Boolean);
}

export default function JobPage({ t }) {
  const [videoText, setVideoText] = useState("");
  const [channelId, setChannelId] = useState("");
  const [maxVideos, setMaxVideos] = useState(5);
  const [maxComments, setMaxComments] = useState(200);
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [job, setJob] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);

  const parsedPreview = useMemo(() => {
    const ids = [];
    for (const token of tokenizeVideoInput(videoText)) {
      const id = extractVideoId(token);
      if (id && !ids.includes(id)) ids.push(id);
    }
    return ids;
  }, [videoText]);

  useEffect(() => {
    if (!job?.job_id || job.status === "done" || job.status === "failed") return undefined;
    const id = setInterval(() => {
      api
        .job(job.job_id)
        .then(setJob)
        .catch((e) => setErr(e.message));
    }, 2000);
    return () => clearInterval(id);
  }, [job?.job_id, job?.status]);

  async function onStart(e) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    // Send raw tokens (URLs or ids); server normalizes and resolves @handles
    const video_ids = tokenizeVideoInput(videoText);
    try {
      const started = await api.startJob({
        video_ids,
        channel_id: channelId.trim() || null,
        max_videos: Number(maxVideos),
        max_comments: Number(maxComments),
        min_confidence: Number(minConfidence),
      });
      setJob(started);
    } catch (ex) {
      setErr(ex.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <h2 style={{ marginTop: 0 }}>{t.job_title}</h2>
      <form className="form-grid card" onSubmit={onStart}>
        <label>
          {t.job_videos}
          <textarea
            value={videoText}
            onChange={(e) => setVideoText(e.target.value)}
            placeholder={t.job_videos_ph}
          />
          {parsedPreview.length > 0 && (
            <p className="muted" style={{ margin: "0.35rem 0 0", fontSize: "0.85rem" }}>
              {t.job_parsed_ids}: {parsedPreview.join(", ")}
            </p>
          )}
        </label>
        <label>
          {t.job_channel}
          <input
            value={channelId}
            onChange={(e) => setChannelId(e.target.value)}
            placeholder={t.job_channel_ph}
          />
          <p className="muted" style={{ margin: "0.35rem 0 0", fontSize: "0.85rem" }}>
            {t.job_channel_hint}
          </p>
        </label>
        <label>
          {t.job_max_videos}
          <input
            type="number"
            min={1}
            max={50}
            value={maxVideos}
            onChange={(e) => setMaxVideos(e.target.value)}
          />
        </label>
        <label>
          {t.job_max_comments}
          <input
            type="number"
            min={1}
            max={20000}
            value={maxComments}
            onChange={(e) => setMaxComments(e.target.value)}
          />
        </label>
        <label>
          {t.job_min_conf}
          <input
            type="number"
            step="0.05"
            min={0}
            max={1}
            value={minConfidence}
            onChange={(e) => setMinConfidence(e.target.value)}
          />
        </label>
        <button className="btn" type="submit" disabled={busy}>
          {t.job_start}
        </button>
        {err && <p className="status-bad">{err}</p>}
      </form>

      {job && (
        <div className="card" style={{ marginTop: "1rem" }}>
          <h3>
            {t.job_status}: <span className="badge">{job.status}</span>
          </h3>
          {job.run_id && (
            <p>
              <Link to={`/runs/${job.run_id}`}>{job.run_id}</Link>
            </p>
          )}
          {job.error && <p className="status-bad">{job.error}</p>}
          <h4 className="muted">{t.job_logs}</h4>
          <div className="logs">{(job.logs || []).join("\n") || "—"}</div>
        </div>
      )}
    </section>
  );
}
