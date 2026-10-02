import type { ColumnSource, DatasetColumn, DatasetRelease } from "@/types/dataset";

const SOURCE_LABELS: Record<ColumnSource, string> = {
  youtube: "YouTube API",
  collected: "Our collection",
  computed: "Computed by us",
};

const SOURCE_NOTES: Record<ColumnSource, string> = {
  youtube: "returned by the YouTube Data API, stored as it came.",
  collected: "observed by our own polling, four times a day.",
  computed: "derived by us from other fields, by the rule shown.",
};

function formatDate(value: string): string {
  return new Date(value).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
}

function formatBytes(bytes: number): string {
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}

function groupColumns(columns: DatasetColumn[]): [string, DatasetColumn[]][] {
  const groups = new Map<string, DatasetColumn[]>();
  for (const column of columns) {
    groups.set(column.group, [...(groups.get(column.group) ?? []), column]);
  }
  return [...groups.entries()];
}

function Section({
  index,
  kicker,
  title,
  id,
  children,
}: {
  index: string;
  kicker: string;
  title: string;
  id: string;
  children: React.ReactNode;
}) {
  return (
    <section className="insight-section" aria-labelledby={id}>
      <p className="section-index">{index}</p>
      <div>
        <p className="section-kicker">{kicker}</p>
        <h2 id={id}>{title}</h2>
        {children}
      </div>
    </section>
  );
}

export default function DatasetView({ release }: { release: DatasetRelease }) {
  const tolerance = release.toleranceHours;
  const groups = groupColumns(release.columns);

  return (
    <>
      <dl className="insight-stats insight-stats--four">
        <div>
          <dt>Videos</dt>
          <dd>{release.videos.toLocaleString("en-LK")}</dd>
        </div>
        <div>
          <dt>Sri Lankan channels</dt>
          <dd>{release.channels.toLocaleString("en-LK")}</dd>
        </div>
        <div>
          <dt>Columns</dt>
          <dd>{release.columns.length}</dd>
        </div>
        <div>
          <dt>Published</dt>
          <dd>
            {formatDate(release.periodStart)} to {formatDate(release.periodEnd)}
          </dd>
        </div>
      </dl>

      <div className="dataset-downloads" aria-label="Downloads">
        {release.files.map((file) => (
          <a key={file.name} className="dataset-download" href={file.url} download>
            <span className="dataset-download__format">{file.format}</span>
            <strong>{file.name}</strong>
            <span>{formatBytes(file.bytes)}</span>
            <code title="SHA-256 checksum">sha256 {file.sha256.slice(0, 16)}…</code>
          </a>
        ))}
      </div>

      <div className="insight-grid">
        <Section index="01" kicker="Collection" title="Where the data comes from" id="collection-title">
          <p>
            Every video published by a tracked Sri Lankan channel was picked up within hours of
            going live, then polled four times a day through the YouTube Data API until day 30.
            Channels count as Sri Lankan when their YouTube profile declares Sri Lanka as their
            country.
          </p>
          <p>
            The channel figures in each row are the ones from just before the video was published,
            so the table can be used to predict a video&apos;s views from what was known at the
            time without the result leaking in.
          </p>
        </Section>

        <Section index="02" kicker="Labels" title="Reading the day-7 to day-30 columns" id="labels-title">
          <p>
            Day N is the poll nearest to exactly N × 24 hours after the video was published.
            Polls drift and collection sometimes paused, so each day carries{" "}
            <code>dN_hours_off</code>: how far that poll was from the mark. Use a day&apos;s figures
            only when it is within ±{tolerance} hours and <code>dN_views</code> is present.
          </p>
          <div className="effect-table-wrap">
            <table className="effect-table">
              <caption className="sr-only">Videos with a usable figure at each day</caption>
              <thead>
                <tr>
                  <th scope="col">Day</th>
                  <th scope="col">Videos with a usable figure</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(release.labelled).map(([day, count]) => (
                  <tr key={day}>
                    <th scope="row">Day {day.replace("day", "")}</th>
                    <td>{count.toLocaleString("en-LK")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section index="03" kicker="Data dictionary" title="Every column" id="columns-title">
          <ul className="dataset-sources">
            {(Object.keys(SOURCE_LABELS) as ColumnSource[]).map((source) => (
              <li key={source}>
                <span className={`source-badge source-badge--${source}`}>{SOURCE_LABELS[source]}</span>{" "}
                {SOURCE_NOTES[source]}
              </li>
            ))}
          </ul>
          {groups.map(([group, columns]) => (
            <div key={group} className="effect-table-wrap dataset-columns">
              <table className="effect-table">
                <caption>{group}</caption>
                <thead>
                  <tr>
                    <th scope="col">Column</th>
                    <th scope="col">Type</th>
                    <th scope="col">Meaning</th>
                    <th scope="col">Source and derivation</th>
                    <th scope="col">Empty</th>
                  </tr>
                </thead>
                <tbody>
                  {columns.map((column) => (
                    <tr key={column.name}>
                      <th scope="row">
                        <code>{column.name}</code>
                      </th>
                      <td>{column.type}</td>
                      <td>{column.meaning}</td>
                      <td>
                        <span className={`source-badge source-badge--${column.source}`}>
                          {SOURCE_LABELS[column.source]}
                        </span>{" "}
                        {column.how}
                      </td>
                      <td>{column.emptyPct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </Section>

        <Section index="04" kicker="Left out" title="Columns you can rebuild" id="rebuild-title">
          <p>
            Anything that can be recomputed exactly from the columns above was left out. To get it
            back:
          </p>
          <ul className="limitations-list">
            <li>
              <strong>Usable label:</strong> <code>abs(dN_hours_off) ≤ {tolerance}</code> and{" "}
              <code>dN_views</code> present.
            </li>
            <li>
              <strong>Channel figures recorded after publishing:</strong>{" "}
              <code>ch_stats_as_of &gt; published_at</code>. No snapshot predated these videos, so
              the earliest one was used.
            </li>
            <li>
              <strong>Sri Lanka time:</strong> <code>published_at</code> plus 5 hours 30 minutes.
            </li>
            <li>
              <strong>Tag count:</strong> the number of <code>|</code>-separated entries in{" "}
              <code>tags</code>.
            </li>
            <li>
              <strong>Thumbnail:</strong> <code>https://i.ytimg.com/vi/&lt;video_id&gt;/hqdefault.jpg</code>.
            </li>
            <li>
              <strong>Rows we modelled:</strong> not a live broadcast, and{" "}
              <code>duration_seconds</code> present.
            </li>
          </ul>
        </Section>

        <Section index="05" kicker="Caveats" title="Before you use it" id="caveats-title">
          <ul className="limitations-list">
            <li>View counts are worldwide. YouTube does not say how many came from Sri Lanka.</li>
            <li>
              A Shorts view is counted every time a Short starts or replays, a looser count than
              for regular videos, so the two are not directly comparable.
            </li>
            <li>
              It covers three months, so festivals and other seasonal effects are not represented.
            </li>
            <li>
              Titles and channel names are as first collected. <code>title_changed</code> marks
              videos whose title was later edited.
            </li>
            <li>Likes are empty when the uploader hides them, and comments when they are turned off.</li>
          </ul>
        </Section>

        <Section index="06" kicker="Citation" title="Using the dataset" id="cite-title">
          <p>
            Our columns and labels are released under{" "}
            <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Titles, tags and
            other uploaded text belong to their creators. Please cite it as:
          </p>
          <p className="dataset-citation">
            Ahamed M.J.S., Ahamed M.U.A. and Ahmedh M.R.R. ViewCastLK: Sri Lankan YouTube viewership
            dataset, release {release.releaseTag}. University of Moratuwa, {release.periodEnd.slice(0, 4)}.
          </p>
          <p className="insight-caption">Release built {formatDate(release.generatedAt)}.</p>
        </Section>
      </div>
    </>
  );
}
