# Thesis & report readiness checklist

Concrete gaps against Chapters 6–7 and submission polish. Status: **done** / **partial** / **TODO (you)** / **TODO (after new runs)**.

## Already strong (keep)

| Item | Where |
|------|--------|
| Clear RQs + contributions | Ch. 1 |
| A0–A5 numbers + multi-seed | Ch. 5 |
| Filtering leakage story (98.6% vs 75.56%) | Ch. 5 / 7 |
| Excluded vs Others redesign + algorithm | Ch. 6.3 |
| Honest limitations (single annotator, small deploy sample) | Ch. 7.4 |
| Reproducibility appendix commands | Appendix |
| Ethics (human-in-loop, no automated action) | Ch. 6.4 |

---

## Chapter 6 — Deployment

| # | Item | Status | Action |
|---|------|--------|--------|
| D1 | Architecture figure + components | **done** | — |
| D2 | Runs 1–2 tables (initial pipeline) | **done** | Keep; label clearly as pre-redesign |
| D3 | Dual rates under redesigned pipeline | **TODO (after new runs)** | Re-run same video + channel; fill exclusion_rate + others_rate_among_usable |
| D4 | Usable-only emotion chart for new runs | **TODO (after new runs)** | Export from dashboard Usable-only view |
| D5 | Dashboard screenshots (EN + FA) in appendix | **TODO (you)** | Capture Runs list, Run detail, New job URL paste, Review queue |
| D6 | In-domain labeled YouTube gold + P/R | **TODO (you)** | Annotate ~200–400 comments; report Fear/Anger + Macro-F1; κ if 2+ annotators |
| D7 | URL autoparse mentioned | **partial** | Optional one sentence in Ch. 6.1 ingestion bullet (URLs / @handles) |
| D8 | Resolved job params (URL → id) in narrative | **optional** | Screenshot of New job preview + job logs |

## Chapter 7 — Discussion

| # | Item | Status | Action |
|---|------|--------|--------|
| S1 | RQ1–RQ4 answered | **done** | — |
| S2 | Filtering vs augmentation argument | **done** | — |
| S3 | Test-split vs wild gap | **done** | Update after D3/D6 with new numbers |
| S4 | Limitations list | **done** | After D3, soften/update item 8 (redesign not yet active) |
| S5 | Future work | **done** (Ch. 8) | Align with D6 as first priority |

## Front matter / submission

| # | Item | Status | Action |
|---|------|--------|--------|
| F1 | Supervisor name/title | **TODO (you)** | `titlepage.tex` |
| F2 | Programme code / name vs Studienblatt | **TODO (you)** | Confirm `066 921` |
| F3 | Acknowledgements | **TODO (you)** | Personal text |
| F4 | Abstract / Kurzfassung length | **partial** | Re-read after number freeze |
| F5 | Freeze paper ↔ thesis ↔ README numbers | **TODO (you)** | One pass before print |

## Figures / appendix pack

| # | Item | Status |
|---|------|--------|
| G1 | Ablation Macro-F1 / Fear F1 bars | **done** (Ch. 5) |
| G2 | A4 confusion matrix | check if present in Ch. 5 |
| G3 | Multi-seed error bars | check Ch. 5 |
| G4 | YouTube redesigned dual-rate table | **TODO (after new runs)** |
| G5 | Dashboard screenshot plate | **TODO (you)** |

## GitHub / reproducibility

| # | Item | Status |
|---|------|--------|
| R1 | Root README landing (abstract + best numbers) | see repo update |
| R2 | `CITATION.cff` + LICENSE | see repo update |
| R3 | Rotate exposed YouTube API key | **TODO (you)** — do in Google Cloud Console |
| R4 | Tag `thesis-v1.0` after number freeze | **TODO (you)** |
| R5 | Do not commit `Master_Thesis_Cvetanovic/` | gitignored |
| R6 | Datasets/models not in git | already ignored |

---

## Suggested order (next sessions)

1. Fill title page (F1–F3).  
2. On DGX: `git pull` → rebuild dashboard → re-run video + channel jobs → copy `summary.json` rates into Ch. 6 (D3–D4).  
3. Screenshot dashboard for appendix (D5).  
4. Annotate a small YouTube gold set (D6) — highest grade impact.  
5. Freeze numbers; tag release; rotate API key (F5, R3–R4).

## Commands for D3 (redesigned runs)

```bash
export YOUTUBE_API_KEY=...
# Same sources as thesis Runs 1–2, with current filter-then-classify code:
python -m deployment.youtube_batch --video-id 'VIDEO_OR_URL' --max-comments 100
python -m deployment.youtube_batch --channel-id '@handle_or_UC…' --max-videos 5 --max-comments 500
```

From each run’s `summary.json`, record at least:

- `n_comments`, `n_usable`, `n_excluded`, `exclusion_rate`
- `others_rate_among_usable`, `label_counts_usable`
- `fear_anger_count`
