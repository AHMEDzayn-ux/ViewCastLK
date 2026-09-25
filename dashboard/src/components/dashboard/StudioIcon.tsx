import type { SVGProps } from "react";

export type StudioIconName = "forecast" | "history" | "channel" | "chart" | "book" | "arrow" | "spark" | "shield" | "play" | "clock" | "plus" | "logout" | "check";

const paths: Record<StudioIconName, React.ReactNode> = {
  forecast: <><rect x="3" y="3" width="18" height="18" rx="4" /><path d="m7 15 4-4 3 2 3-5M14 8h3v3" /></>,
  history: <path d="M3 11a9 9 0 1 1 2.6 7M3 5v6h6M12 7v5l3 2" />,
  channel: <><rect x="3" y="5" width="18" height="14" rx="4" /><path d="m10 9 5 3-5 3Z" /></>,
  chart: <path d="M4 4v16h17M9 15V9M14 15V5M19 15v-5" />,
  book: <path d="M12 6c-3-2-6-2-9-1v14c3-1 6-1 9 1 3-2 6-2 9-1V5c-3-1-6-1-9 1Zm0 0v14" />,
  arrow: <path d="M4 12h15m-6-6 6 6-6 6" />,
  spark: <path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z" />,
  shield: <><path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z" /><path d="m8 12 3 3 5-6" /></>,
  play: <path d="m8 4 12 8-12 8V4Z" />,
  clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
  plus: <path d="M12 5v14M5 12h14" />,
  logout: <path d="M10 4H4v16h6M10 12h11m-4-4 4 4-4 4" />,
  check: <path d="m5 12 4 4L19 6" />,
};

export default function StudioIcon({ name, ...props }: SVGProps<SVGSVGElement> & { name: StudioIconName }) {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>;
}
