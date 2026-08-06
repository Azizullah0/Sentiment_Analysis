export default function AboutPage({ t }) {
  return (
    <section className="card about-card">
      <h2 style={{ marginTop: 0 }}>{t.about_title}</h2>
      <p>{t.about_intro}</p>
      <ol className="about-steps">
        <li>{t.about_step_clean}</li>
        <li>{t.about_step_gate}</li>
        <li>{t.about_step_model}</li>
        <li>{t.about_step_abstain}</li>
      </ol>
      <p className="muted">{t.about_ethics}</p>
      <dl className="about-defs">
        <div>
          <dt>{t.Excluded}</dt>
          <dd>{t.about_excluded}</dd>
        </div>
        <div>
          <dt>{t.Others}</dt>
          <dd>{t.about_others}</dd>
        </div>
      </dl>
    </section>
  );
}
