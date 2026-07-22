# Antara — web

Minimal Next.js frontend for the Antara capstone project: an overview
page, a results page (real PSNR/SSIM/LPIPS numbers pulled from
`README.md`/`docs/PLAN.md` at the repo root), an interactive
before/after slider demo comparing Farneback vs. fine-tuned FILM on a
real cyclone frame, and a live page that runs that same comparison
against whatever GOES-19 scans NOAA most recently published.

Deliberately minimal design: monochrome (white/black, with a dark-mode
variant), no component library. Every page except `/live` is fully
static — every number and image generated ahead of time by the Python
pipeline in `src/`. `/live` is the one page that fetches at runtime,
calling the FastAPI service in `src/api/live.py` (needs to be running
locally — see the repo root README's "Web frontend" section).

## Develop

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Structure

```
app/
  page.tsx            — overview / landing page
  results/page.tsx     — results tables + calm-vs-cyclone bar chart
  demo/page.tsx        — before/after slider demo
  live/page.tsx        — live pipeline page (calls src/api/live.py at runtime)
  layout.tsx, globals.css — shared shell + design system
components/            — SiteHeader, SiteFooter, ResultsTable, PsnrBarChart,
                          BeforeAfterSlider, LivePipeline
lib/results.ts          — single source of truth for the numbers shown on /results and /demo;
                          mirrors the repo root README.md's Status section
lib/liveApi.ts          — client for the live pipeline API (src/api/live.py)
public/demo/            — real Farneback/FILM/ground-truth PNGs for one cyclone triplet,
                          generated via src/baseline/farneback_interpolate.py and
                          src/deep/film_interpolate.py
```

## Build

```bash
npm run build
```

Produces a fully static export (no server-side rendering needed — every
page is prerendered at build time).
