import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

const PAGE_SIZE = 8;

function pct(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(Number(x) * 100).toFixed(1)}%`;
}

function formatRange(offset, count, total, t) {
  if (!total) return t.page_empty || "0";
  const from = offset + 1;
  const to = offset + count;
  return (t.page_of || "{from}–{to} of {total}")
    .replace("{from}", String(from))
    .replace("{to}", String(to))
    .replace("{total}", String(total));
}

export default function RunsPage({ t }) {
  const [runs, setRuns] = useState(null);
  const [err, setErr] = useState(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    api
      .runs()
      .then((d) => {
        setRuns(d.runs || []);
        setPage(0);
      })
      .catch((e) => setErr(e.message));
  }, []);

  const total = runs?.length ?? 0;
  const offset = page * PAGE_SIZE;
  const slice = useMemo(
    () => (runs || []).slice(offset, offset + PAGE_SIZE),
    [runs, offset]
  );

  if (err) return <div className="empty">{t.error}: {err}</div>;
  if (!runs) return <div className="empty">{t.loading}</div>;
  if (!runs.length) return <div className="empty">{t.runs_empty}</div>;

  return (
    <section className="run-screen">
      <h2 style={{ marginTop: 0 }}>{t.runs_title}</h2>
      <div className="run-list">
        {slice.map((r) => (
          <article key={r.run_id} className="run-item">
            <div>
              <strong>{r.run_id}</strong>
              <div className="meta">
                {r.model_id ? (
                  <>
                    {t.job_model_short} {r.model_id}
                    {" · "}
                  </>
                ) : null}
                {r.n_usable != null ? r.n_usable : r.n_comments} {t.usable_count}
                {" · "}
                {t.others_rate}{" "}
                {pct(
                  r.others_rate_among_usable != null
                    ? r.others_rate_among_usable
                    : r.others_rate
                )}
                {" · "}
                {t.exclusion_rate} {pct(r.exclusion_rate)}
                {" · "}
                {t.fear_anger} {r.fear_anger_count ?? 0}
                {" · "}
                {(r.video_ids || []).length} {t.videos}
              </div>
            </div>
            <Link className="btn secondary" to={`/runs/${r.run_id}`}>
              {t.open_run}
            </Link>
          </article>
        ))}
      </div>
      <div className="pager">
        <button
          type="button"
          className="btn secondary"
          disabled={page <= 0}
          onClick={() => setPage((p) => Math.max(0, p - 1))}
        >
          {t.prev}
        </button>
        <span className="pager-meta muted">
          {formatRange(offset, slice.length, total, t)}
        </span>
        <button
          type="button"
          className="btn secondary"
          disabled={offset + PAGE_SIZE >= total}
          onClick={() => setPage((p) => p + 1)}
        >
          {t.next}
        </button>
      </div>
    </section>
  );
}
