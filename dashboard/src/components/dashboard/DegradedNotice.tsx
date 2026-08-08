import type { DataCompleteness } from "@/types/forecast";

interface DegradedNoticeProps {
  completeness: DataCompleteness;
}

export default function DegradedNotice({
  completeness,
}: DegradedNoticeProps) {
  if (completeness.status !== "degraded") return null;

  return (
    <aside className="degraded-notice" aria-labelledby="degraded-title">
      <div>
        <p className="section-kicker">Forecast completed with limited context</p>
        <h3 id="degraded-title">Some supporting information was unavailable</h3>
      </div>
      <ul>
        {completeness.issues.map((issue) => (
          <li key={issue.source}>{issue.message}</li>
        ))}
      </ul>
      <p>
        The forecast above is still valid for the information that was
        available.
      </p>
    </aside>
  );
}
