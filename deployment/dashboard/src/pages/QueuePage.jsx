import { useEffect, useMemo, useState } from "react";
import { api } from "../api";

const PAGE_SIZE = 8;

function formatRange(offset, count, total, t) {
  if (!total) return t.page_empty || "0";
  const from = offset + 1;
  const to = offset + count;
  return (t.page_of || "{from}–{to} of {total}")
    .replace("{from}", String(from))
    .replace("{to}", String(to))
    .replace("{total}", String(total));
}

export default function QueuePage({ t }) {
  const [items, setItems] = useState(null);
  const [err, setErr] = useState(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    api
      .queue()
      .then((d) => setItems(d.items || []))
      .catch((e) => setErr(e.message));
  }, []);

  const total = items?.length ?? 0;
  const offset = page * PAGE_SIZE;
  const slice = useMemo(
    () => (items || []).slice(offset, offset + PAGE_SIZE),
    [items, offset]
  );

  if (err) return <div className="empty">{t.error}: {err}</div>;
  if (!items) return <div className="empty">{t.loading}</div>;

  return (
    <section className="run-screen">
      <h2 style={{ marginTop: 0 }}>{t.queue_title}</h2>
      {!items.length ? (
        <div className="empty">—</div>
      ) : (
        <div className="card run-panel-table">
          <div className="table-wrap table-fit">
            <table>
              <thead>
                <tr>
                  <th>{t.filter_label}</th>
                  <th>{t.confidence}</th>
                  <th>{t.table}</th>
                </tr>
              </thead>
              <tbody>
                {slice.map((row, i) => (
                  <tr key={row.comment_id || i}>
                    <td>
                      <span className={`badge ${row.label}`}>
                        {t[row.label] || row.label}
                      </span>
                    </td>
                    <td>
                      {row.confidence != null
                        ? Number(row.confidence).toFixed(3)
                        : "—"}
                    </td>
                    <td className="rtl-text">
                      {row.text_original || row.text || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
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
        </div>
      )}
    </section>
  );
}
