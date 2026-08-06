import { useEffect, useState } from "react";
import { api } from "../api";

export default function QueuePage({ t }) {
  const [items, setItems] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api
      .queue()
      .then((d) => setItems(d.items || []))
      .catch((e) => setErr(e.message));
  }, []);

  if (err) return <div className="empty">{t.error}: {err}</div>;
  if (!items) return <div className="empty">{t.loading}</div>;

  return (
    <section>
      <h2 style={{ marginTop: 0 }}>{t.queue_title}</h2>
      {!items.length ? (
        <div className="empty">—</div>
      ) : (
        <div className="table-wrap card" style={{ padding: 0 }}>
          <table>
            <thead>
              <tr>
                <th>{t.filter_label}</th>
                <th>{t.confidence}</th>
                <th>{t.table}</th>
              </tr>
            </thead>
            <tbody>
              {items.map((row, i) => (
                <tr key={row.comment_id || i}>
                  <td>
                    <span className={`badge ${row.label}`}>{t[row.label] || row.label}</span>
                  </td>
                  <td>
                    {row.confidence != null ? Number(row.confidence).toFixed(3) : "—"}
                  </td>
                  <td className="rtl-text">
                    {row.text_original || row.text || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
