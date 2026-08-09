import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from "recharts";
import { api } from "../api";

const COLORS = {
  Hope: "#6fbf8a",
  Happy: "#d4a15a",
  Neutral: "#8a8474",
  Surprise: "#7eb6c9",
  Disgust: "#9a7bb8",
  Sad: "#6b8cae",
  Anger: "#d07a4c",
  Fear: "#c97b84",
  Others: "#5a564c",
  Excluded: "#3d4540",
};

const USABLE_LABELS = [
  "Hope",
  "Happy",
  "Neutral",
  "Surprise",
  "Disgust",
  "Sad",
  "Anger",
  "Fear",
  "Others",
];

const ALL_LABELS = [...USABLE_LABELS, "Excluded"];

const VIEW_KEY = "deepdari_run_view_mode";

function loadViewMode() {
  try {
    const v = sessionStorage.getItem(VIEW_KEY);
    if (v === "all" || v === "usable") return v;
  } catch {
    /* ignore */
  }
  return "usable";
}

export default function RunDetailPage({ t }) {
  const { runId } = useParams();
  const [detail, setDetail] = useState(null);
  const [comments, setComments] = useState(null);
  const [review, setReview] = useState([]);
  const [label, setLabel] = useState("");
  const [videoId, setVideoId] = useState("");
  const [q, setQ] = useState("");
  const [err, setErr] = useState(null);
  const [saveMsg, setSaveMsg] = useState("");
  const [viewMode, setViewMode] = useState(loadViewMode);

  const setView = (mode) => {
    setViewMode(mode);
    try {
      sessionStorage.setItem(VIEW_KEY, mode);
    } catch {
      /* ignore */
    }
  };

  const chartLabels = viewMode === "all" ? ALL_LABELS : USABLE_LABELS;

  const loadComments = useCallback(() => {
    if (viewMode === "usable" && label === "Excluded") {
      setComments({ total: 0, offset: 0, limit: 100, items: [] });
      return;
    }
    api
      .comments(runId, {
        label,
        video_id: videoId,
        q,
        limit: 100,
        offset: 0,
        usable_only: viewMode === "usable",
      })
      .then(setComments)
      .catch((e) => setErr(e.message));
  }, [runId, label, videoId, q, viewMode]);

  useEffect(() => {
    api
      .run(runId)
      .then(setDetail)
      .catch((e) => setErr(e.message));
    api
      .review(runId)
      .then((d) => setReview(d.items || []))
      .catch(() => setReview([]));
  }, [runId]);

  useEffect(() => {
    loadComments();
  }, [loadComments]);

  const chartData = useMemo(() => {
    const s = detail?.summary || {};
    const counts =
      viewMode === "all"
        ? s.label_counts_all || s.label_counts || {}
        : s.label_counts_usable || s.label_counts || {};
    return chartLabels.map((name) => ({
      name,
      label: t[name] || name,
      count: counts[name] || 0,
    }));
  }, [detail, t, viewMode, chartLabels]);

  const videoChart = useMemo(() => {
    const per = detail?.summary?.comments_per_video || {};
    return Object.entries(per).map(([vid, count]) => ({
      name: vid.slice(0, 8),
      full: vid,
      count,
    }));
  }, [detail]);

  const videos = detail?.summary?.video_ids || [];

  async function saveReview() {
    const items = review.map((r) => ({
      comment_id: r.comment_id,
      checked: !!r.checked,
      note: r.note || null,
    }));
    await api.saveReview(runId, items);
    setSaveMsg(t.saved);
    setTimeout(() => setSaveMsg(""), 2000);
  }

  if (err) return <div className="empty">{t.error}: {err}</div>;
  if (!detail) return <div className="empty">{t.loading}</div>;

  const s = detail.summary || {};
  const othersPct =
    viewMode === "usable"
      ? s.others_rate_among_usable ?? s.others_rate
      : s.others_rate;
  const commentCount =
    viewMode === "usable" ? s.n_usable ?? s.n_comments : s.n_fetched ?? s.n_comments;

  return (
    <section>
      <Link to="/" className="muted">
        ← {t.back}
      </Link>
      <h2 style={{ marginBottom: "0.25rem" }}>{runId}</h2>
      <p className="muted">
        {t.overview}
        {s.model_id ? ` · ${t.job_model_short} ${s.model_id}` : ""}
      </p>

      <div className="card view-toggle-card">
        <div className="toolbar" style={{ marginBottom: "0.35rem" }}>
          <span className="muted" style={{ marginInlineEnd: "0.5rem" }}>
            {t.view_mode}
          </span>
          <div className="segmented" role="group" aria-label={t.view_mode}>
            <button
              type="button"
              className={viewMode === "usable" ? "seg active" : "seg"}
              onClick={() => setView("usable")}
            >
              {t.view_usable}
            </button>
            <button
              type="button"
              className={viewMode === "all" ? "seg active" : "seg"}
              onClick={() => setView("all")}
            >
              {t.view_all}
            </button>
          </div>
        </div>
        <p className="muted" style={{ margin: 0, fontSize: "0.9rem" }}>
          {viewMode === "usable" ? t.view_hint_usable : t.view_hint_all}
        </p>
      </div>

      <div className="grid kpis">
        <div className="card kpi">
          <div className="value">{commentCount ?? "—"}</div>
          <div className="label">
            {viewMode === "usable" ? t.usable_count : t.comments}
          </div>
        </div>
        <div className="card kpi">
          <div className="value">
            {othersPct != null ? `${(othersPct * 100).toFixed(1)}%` : "—"}
          </div>
          <div className="label">{t.others_rate}</div>
        </div>
        <div className="card kpi">
          <div className="value">
            {s.exclusion_rate != null
              ? `${(s.exclusion_rate * 100).toFixed(1)}%`
              : "—"}
          </div>
          <div className="label">{t.exclusion_rate}</div>
        </div>
        <div className="card kpi">
          <div className="value">{s.fear_anger_count_non_others ?? 0}</div>
          <div className="label">{t.fear_anger}</div>
        </div>
        <div className="card kpi">
          <div className="value">{videos.length}</div>
          <div className="label">{t.videos}</div>
        </div>
      </div>

      <div className="charts-row">
        <div className="card chart-panel">
          <h3>{t.distribution}</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData}>
              <CartesianGrid stroke="rgba(232,226,214,0.08)" vertical={false} />
              <XAxis
                dataKey="label"
                tick={{ fill: "#9a927f", fontSize: 11 }}
                interval={0}
                angle={-20}
                textAnchor="end"
                height={60}
              />
              <YAxis tick={{ fill: "#9a927f", fontSize: 11 }} allowDecimals={false} />
              <Tooltip
                contentStyle={{
                  background: "#161c18",
                  border: "1px solid rgba(232,226,214,0.15)",
                }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {chartData.map((entry) => (
                  <Cell key={entry.name} fill={COLORS[entry.name] || "#d4a15a"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card chart-panel">
          <h3>{t.per_video}</h3>
          {videoChart.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={videoChart} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid stroke="rgba(232,226,214,0.08)" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#9a927f", fontSize: 11 }} />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={70}
                  tick={{ fill: "#9a927f", fontSize: 11 }}
                />
                <Tooltip
                  formatter={(v, _n, p) => [v, p.payload.full]}
                  contentStyle={{
                    background: "#161c18",
                    border: "1px solid rgba(232,226,214,0.15)",
                  }}
                />
                <Bar dataKey="count" fill="#d4a15a" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">—</p>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <div className="toolbar">
          <h3 style={{ margin: 0, flex: 1 }}>{t.table}</h3>
          <a className="btn secondary" href={`/api/runs/${runId}/export.csv`}>
            {t.export}
          </a>
        </div>
        <div className="toolbar">
          <select value={label} onChange={(e) => setLabel(e.target.value)}>
            <option value="">{t.filter_all}</option>
            {chartLabels.map((l) => (
              <option key={l} value={l}>
                {t[l] || l}
              </option>
            ))}
          </select>
          <select value={videoId} onChange={(e) => setVideoId(e.target.value)}>
            <option value="">{t.filter_video}: {t.filter_all}</option>
            {videos.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
          <input
            placeholder={t.search}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            style={{ minWidth: "12rem", flex: 1 }}
          />
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t.filter_label}</th>
                <th>{t.confidence}</th>
                <th>{t.raw}</th>
                <th>{t.table}</th>
                <th>video</th>
              </tr>
            </thead>
            <tbody>
              {(comments?.items || []).map((row) => (
                <tr key={row.comment_id}>
                  <td>
                    <span className={`badge ${row.label}`}>{t[row.label] || row.label}</span>
                  </td>
                  <td>
                    {row.confidence != null ? Number(row.confidence).toFixed(3) : "—"}
                  </td>
                  <td>{row.raw_emotion || "—"}</td>
                  <td className="rtl-text">{row.text_original}</td>
                  <td className="muted">{row.video_id}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ marginTop: "0.6rem" }}>
          {comments ? `${comments.items.length} / ${comments.total}` : ""}
        </p>
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <div className="toolbar">
          <h3 style={{ margin: 0, flex: 1 }}>{t.review}</h3>
          <button type="button" className="btn" onClick={saveReview}>
            {t.save_review}
          </button>
          {saveMsg && <span className="status-ok">{saveMsg}</span>}
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t.checked}</th>
                <th>{t.filter_label}</th>
                <th>{t.confidence}</th>
                <th>{t.table}</th>
              </tr>
            </thead>
            <tbody>
              {review.map((row, idx) => (
                <tr key={row.comment_id || idx}>
                  <td>
                    <input
                      type="checkbox"
                      checked={!!row.checked}
                      onChange={(e) => {
                        const next = [...review];
                        next[idx] = { ...row, checked: e.target.checked };
                        setReview(next);
                      }}
                    />
                  </td>
                  <td>
                    <span className={`badge ${row.label}`}>{t[row.label] || row.label}</span>
                  </td>
                  <td>
                    {row.confidence != null ? Number(row.confidence).toFixed(3) : "—"}
                  </td>
                  <td className="rtl-text">{row.text_original}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
