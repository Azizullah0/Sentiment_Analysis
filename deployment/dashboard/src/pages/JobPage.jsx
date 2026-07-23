import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

export default function JobPage({ t }) {
  const [videoText, setVideoText] = useState("");
  const [channelId, setChannelId] = useState("");
  const [maxVideos, setMaxVideos] = useState(5);
  const [maxComments, setMaxComments] = useState(200);
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [job, setJob] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);

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
    const video_ids = videoText
      .split(/[\n,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
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
            placeholder="16e75OffBTA"
          />
        </label>
        <label>
          {t.job_channel}
          <input
            value={channelId}
            onChange={(e) => setChannelId(e.target.value)}
            placeholder="UCxxxxxxxx"
          />
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
