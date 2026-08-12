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
import DeepDariLogo from "./DeepDariLogo";

const DICTS = { en, fa };
const THEME_KEY = "deepdari_theme";

function loadTheme() {
  try {
    const v = localStorage.getItem(THEME_KEY);
    if (v === "dark" || v === "light") return v;
  } catch {
    /* ignore */
  }
  return "light";
}

export default function App() {
  const [lang, setLang] = useState(
    () => localStorage.getItem("deepdari_lang") || "en"
  );
  const [theme, setTheme] = useState(loadTheme);
  const [health, setHealth] = useState(null);
  const t = useMemo(() => DICTS[lang] || en, [lang]);

  useEffect(() => {
    document.documentElement.lang = lang === "fa" ? "fa" : "en";
    document.documentElement.dir = lang === "fa" ? "rtl" : "ltr";
    localStorage.setItem("deepdari_lang", lang);
  }, [lang]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "error" }));
  }, []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <h1>
            <Link to="/" className="brand-link">
              <DeepDariLogo title={t.brand} />
              <span>{t.brand}</span>
            </Link>
          </h1>
          <p>{t.tagline}</p>
          {health && (
            <p className="muted brand-meta">
              <span className={health.status === "ok" ? "status-ok" : "status-bad"}>
                {health.status === "ok" ? t.model_ok : t.model_bad}
              </span>
              {" · "}
              {health.youtube_key_set ? "YouTube API ✓" : t.no_api_key}
              {typeof health.n_runs === "number" ? ` · ${health.n_runs} runs` : ""}
              {" · "}
              <span title="UI build marker">ui:header-fit</span>
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
          <div className="lang-toggle" role="group" aria-label={t.theme_toggle}>
            <button
              type="button"
              className={theme === "light" ? "active" : ""}
              onClick={() => setTheme("light")}
            >
              {t.theme_light}
            </button>
            <button
              type="button"
              className={theme === "dark" ? "active" : ""}
              onClick={() => setTheme("dark")}
            >
              {t.theme_dark}
            </button>
          </div>
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
        <Route path="/runs/:runId" element={<RunDetailPage t={t} theme={theme} />} />
        <Route path="/job" element={<JobPage t={t} />} />
        <Route path="/queue" element={<QueuePage t={t} />} />
        <Route path="/about" element={<AboutPage t={t} />} />
      </Routes>
    </div>
  );
}
