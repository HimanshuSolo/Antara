# Antara — web

Minimal, static Next.js frontend for the Antara capstone project: an
overview page, a results page (real PSNR/SSIM/LPIPS numbers pulled from
`README.md`/`docs/PLAN.md` at the repo root), and an interactive
before/after slider demo comparing Farneback vs. fine-tuned FILM on a
real cyclone frame.

Deliberately minimal design: monochrome (white/black, with a dark-mode
variant), no component library, no client-side data fetching — every
number and image is static, generated ahead of time by the Python
pipeline in `src/`.

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
  layout.tsx, globals.css — shared shell + design system
components/            — SiteHeader, SiteFooter, ResultsTable, PsnrBarChart, BeforeAfterSlider
lib/results.ts          — single source of truth for the numbers shown on /results and /demo;
                          mirrors the repo root README.md's Status section
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
