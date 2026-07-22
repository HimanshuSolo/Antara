import Link from "next/link";
import { headlineStats } from "@/lib/results";

const STEPS = [
  {
    n: "01",
    title: "Classical baseline",
    body: "Dense Farneback optical flow is computed between frame t−1 and t+1, then warped and blended to estimate the frame at t = 0.5. This traditional method serves as the baseline for comparison.",
  },
  {
    n: "02",
    title: "Learned interpolation",
    body: "Google's FILM model estimates optical flow and learns to blend and repair the warped result. It is fine-tuned on real satellite image triplets rather than trained from scratch.",
  },
  {
    n: "03",
    title: "Stratified evaluation",
    body: "Results are stratified into calm-weather and cyclone/storm subsets, curated using IBTrACS cyclone track data — the fast, non-linear motion conditions under which classical methods are known to fail.",
  },
];

export default function Home() {
  return (
    <>
      <section className="section container">
        <div className="section-label">Satellite temporal super-resolution</div>
        <h1>Reconstructing missing frames between satellite scans.</h1>
        <p className="lede">
          Geostationary satellites image the same region every 10–30 minutes, which is too
          coarse to track cyclones, storm cells, or smoke plumes in near real time. This project
          addresses ISRO&apos;s <em>&quot;Fill in the Frames Seamlessly&quot;</em> problem
          statement by generating the missing intermediate frame using a learned,
          optical-flow-based interpolation method, and evaluates specifically where it improves
          upon classical interpolation under fast, non-linear cloud motion.
        </p>
        <div className="btn-row">
          <Link href="/results" className="btn btn--primary">
            View results
          </Link>
          <Link href="/demo" className="btn btn--secondary">
            Interactive demo
          </Link>
          <Link href="/live" className="btn btn--secondary">
            Live pipeline
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
        <h2>Methodology</h2>
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
        <h2>Self-supervised training</h2>
        <p className="prose">
          Training data consists of real triplets <code>(t−1, t, t+1)</code> extracted from the
          GOES-16 satellite archive on AWS Open Data. Frame <code>t</code> serves as the
          ground-truth label, held out during inference and used only to compute the training
          loss. No manual annotation is required at any stage of this pipeline.
        </p>
      </section>
    </>
  );
}
