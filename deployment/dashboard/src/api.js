async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (res.headers.get("content-type")?.includes("application/json")) {
    return res.json();
  }
  return res;
}

export const api = {
  health: () => request("/api/health"),
  runs: () => request("/api/runs"),
  run: (id) => request(`/api/runs/${id}`),
  comments: (id, params = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
    });
    return request(`/api/runs/${id}/comments?${q}`);
  },
  review: (id) => request(`/api/runs/${id}/review`),
  saveReview: (id, items) =>
    request(`/api/runs/${id}/review`, {
      method: "POST",
      body: JSON.stringify({ items }),
    }),
  queue: () => request("/api/review-queue"),
  models: () => request("/api/models"),
  startJob: (body) =>
    request("/api/jobs", { method: "POST", body: JSON.stringify(body) }),
  job: (id) => request(`/api/jobs/${id}`),
  jobs: () => request("/api/jobs"),
};
