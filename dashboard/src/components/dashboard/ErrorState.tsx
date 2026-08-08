interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
  eyebrow?: string;
  title?: string;
}

export default function ErrorState({
  message,
  onRetry,
  eyebrow = "Forecast unavailable",
  title = "We could not generate a forecast",
}: ErrorStateProps) {
  return (
    <section className="result-state result-state--error" role="alert">
      <p className="result-state__eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p>{message}</p>
      {onRetry && (
        <button className="secondary-button" type="button" onClick={onRetry}>
          Try again
        </button>
      )}
    </section>
  );
}
