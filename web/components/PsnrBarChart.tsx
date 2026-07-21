import { stratifiedResults } from "@/lib/results";

const MAX_PSNR = 40;

export default function PsnrBarChart() {
  const rows = stratifiedResults.flatMap((subset) =>
    subset.results.map((result) => ({
      key: `${subset.subset}-${result.method}`,
      label: `${result.method} — ${subset.subset}`,
      value: result.psnr,
    })),
  );

  return (
    <div className="bar-chart">
      {rows.map((row) => (
        <div key={row.key}>
          <div className="bar-row__label">
            <span>{row.label}</span>
            <span className="value">{row.value.toFixed(2)} dB</span>
          </div>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{ width: `${Math.min(100, (row.value / MAX_PSNR) * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
