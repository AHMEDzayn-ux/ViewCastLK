interface LoadingStateProps {
  eyebrow?: string;
  title?: string;
  message?: string;
}

export default function LoadingState({
  eyebrow = "Generating your forecast",
  title = "Checking the submitted details",
  message =
    "ViewCastLK is preparing the four cumulative view estimates. No result is shown until the response is ready.",
}: LoadingStateProps) {
  return (
    <section
      className="result-state"
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <span className="loading-indicator" aria-hidden="true" />
      <p className="result-state__eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p>{message}</p>
    </section>
  );
}
