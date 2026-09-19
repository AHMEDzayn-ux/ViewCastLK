import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy policy",
  description: "How ViewCastLK handles connected YouTube creator data.",
};

export default function PrivacyPage() {
  return (
    <main className="page-shell information-page">
      <header className="page-intro page-intro--narrow">
        <p className="section-kicker">Privacy</p>
        <h1>Connected YouTube data</h1>
        <p>
          This project uses read-only Google permissions to provide private,
          creator-specific forecasting features.
        </p>
      </header>

      <div className="privacy-sections">
        <section>
          <h2>What is accessed</h2>
          <p>
            With your consent, ViewCastLK reads your YouTube channel identity,
            video metadata, and YouTube Analytics view data. It cannot upload,
            edit, or delete your YouTube content.
          </p>
        </section>
        <section>
          <h2>Why it is used</h2>
          <p>
            The data is used to compare forecasts with your channel&apos;s actual
            performance and calculate adjustments for your own forecasts.
          </p>
        </section>
        <section>
          <h2>Storage and separation</h2>
          <p>
            Creator data is stored separately from the public training warehouse.
            Google refresh credentials are encrypted and are not made available
            to browser clients. Private YouTube Analytics values are not used to
            train the shared forecasting model.
          </p>
        </section>
        <section>
          <h2>Revocation and deletion</h2>
          <p>
            You can revoke access from your Google Account at any time. The
            ViewCastLK disconnect control also requests Google revocation and
            immediately removes the associated creator-private data.
          </p>
        </section>
      </div>
    </main>
  );
}
