import Link from "next/link";
import { headlineStats } from "@/lib/results";

const STEPS = [
  {
    n: "01",
    title: "Classical baseline",
    body: "Dense Farneback optical flow between frame t−1 and t+1, warped and blended at t=0.5 — the traditional method, and the thing being beaten.",
  },
  {
    n: "02",
    title: "Learned interpolation",
    body: "Google's FILM estimates flow and learns how to blend/repair the warped result, fine-tuned on real satellite triplets instead of trained from scratch.",
  },
  {
    n: "03",
    title: "Stratified evaluation",
    body: "Results are split into calm weather vs. cyclone/storm windows, curated from IBTrACS track data — exactly the fast, non-linear case classical methods fail on.",
  },
];

export default function Home() {
  return (
    <>
      <section className="section container">
        <div className="section-label">Satellite temporal super-resolution</div>
        <h1>Filling in the frames between satellite scans.</h1>
        <p className="lede">
          Geostationary satellites image the same region every 10–30 minutes — too coarse to
          track cyclones, storm cells, or smoke plumes in near-real-time. Antara generates the
          missing frame in between using a learned, optical-flow-based method, and measures
          specifically where it holds up better than classical interpolation on fast, non-linear
          cloud motion — the case ISRO&apos;s <em>&quot;Fill in the Frames Seamlessly&quot;</em> problem
          statement calls out directly.
        </p>
        <div className="btn-row">
          <Link href="/results" className="btn btn--primary">
            View results
          </Link>
          <Link href="/demo" className="btn btn--secondary">
            Try the demo
          </Link>
          <Link href="/live" className="btn btn--secondary">
            Run it live
          </Link>
        </div>

        <div className="stat-grid">
          {headlineStats.map((stat) => (
            <div className="stat" key={stat.label}>
              <div className="stat__value">{stat.value}</div>
              <div className="stat__label">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="section section--tight container">
        <h2>How it works</h2>
        <div className="card-grid">
          {STEPS.map((step) => (
            <div className="card" key={step.n}>
              <div className="step-number">{step.n}</div>
              <h3>{step.title}</h3>
              <p className="prose">{step.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section section--tight container">
        <h2>Why it&apos;s self-supervised</h2>
        <p className="prose">
          Training data is real triplets <code>(t−1, t, t+1)</code> pulled straight from the
          GOES-16 satellite archive on AWS Open Data — frame <code>t</code> is the label, held
          out during inference and used only to compute the loss during training. No manual
          annotation anywhere in this pipeline.
        </p>
      </section>
    </>
  );
}
