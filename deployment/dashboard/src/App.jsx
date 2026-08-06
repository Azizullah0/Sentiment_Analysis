import { useEffect, useMemo, useState } from "react";
import { Link, NavLink, Route, Routes } from "react-router-dom";
import en from "./i18n/en.json";
import fa from "./i18n/fa.json";
import { api } from "./api";
import RunsPage from "./pages/RunsPage";
import RunDetailPage from "./pages/RunDetailPage";
import JobPage from "./pages/JobPage";
import QueuePage from "./pages/QueuePage";
import AboutPage from "./pages/AboutPage";

const DICTS = { en, fa };

export default function App() {
  const [lang, setLang] = useState(
    () => localStorage.getItem("deepdari_lang") || "en"
  );
  const [health, setHealth] = useState(null);
  const t = useMemo(() => DICTS[lang] || en, [lang]);

  useEffect(() => {
    document.documentElement.lang = lang === "fa" ? "fa" : "en";
    document.documentElement.dir = lang === "fa" ? "rtl" : "ltr";
    localStorage.setItem("deepdari_lang", lang);
  }, [lang]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "error" }));
  }, []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <h1>
            <Link to="/" style={{ color: "inherit" }}>
              {t.brand}
            </Link>
          </h1>
          <p>{t.tagline}</p>
          {health && (
            <p className="muted" style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
              <span className={health.status === "ok" ? "status-ok" : "status-bad"}>
                {health.status === "ok" ? t.model_ok : t.model_bad}
              </span>
              {" · "}
              {health.youtube_key_set ? "YouTube API ✓" : t.no_api_key}
              {typeof health.n_runs === "number" ? ` · ${health.n_runs} runs` : ""}
            </p>
          )}
        </div>
        <div className="nav-row">
          <NavLink to="/" end>
            {t.nav_runs}
          </NavLink>
          <NavLink to="/job">{t.nav_job}</NavLink>
          <NavLink to="/queue">{t.nav_queue}</NavLink>
          <NavLink to="/about">{t.nav_about}</NavLink>
          <div className="lang-toggle" role="group" aria-label="Language">
            <button
              type="button"
              className={lang === "en" ? "active" : ""}
              onClick={() => setLang("en")}
            >
              {t.lang_en}
            </button>
            <button
              type="button"
              className={lang === "fa" ? "active" : ""}
              onClick={() => setLang("fa")}
            >
              {t.lang_fa}
            </button>
          </div>
        </div>
      </header>

      <Routes>
        <Route path="/" element={<RunsPage t={t} />} />
        <Route path="/runs/:runId" element={<RunDetailPage t={t} />} />
        <Route path="/job" element={<JobPage t={t} />} />
        <Route path="/queue" element={<QueuePage t={t} />} />
        <Route path="/about" element={<AboutPage t={t} />} />
      </Routes>
    </div>
  );
}
