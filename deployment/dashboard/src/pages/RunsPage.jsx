import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

function pct(x) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${(Number(x) * 100).toFixed(1)}%`;
}

export default function RunsPage({ t }) {
  const [runs, setRuns] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api
      .runs()
      .then((d) => setRuns(d.runs || []))
      .catch((e) => setErr(e.message));
  }, []);

  if (err) return <div className="empty">{t.error}: {err}</div>;
  if (!runs) return <div className="empty">{t.loading}</div>;
  if (!runs.length) return <div className="empty">{t.runs_empty}</div>;

  return (
    <section>
      <h2 style={{ marginTop: 0 }}>{t.runs_title}</h2>
      <div className="run-list">
        {runs.map((r) => (
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
    </section>
  );
}
