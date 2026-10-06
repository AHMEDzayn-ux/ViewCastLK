"use client";

import { useState } from "react";
import Link from "next/link";
import StudioIcon from "@/components/dashboard/StudioIcon";

const CHECKPOINTS = [
  { day: 7, views: "2,840", x: 122, y: 152 },
  { day: 14, views: "4,320", x: 232, y: 106 },
  { day: 21, views: "5,790", x: 342, y: 72 },
  { day: 30, views: "7,240", x: 452, y: 36 },
];

export default function LandingForecastPreview() {
  const [selected, setSelected] = useState(3);
  const checkpoint = CHECKPOINTS[selected];

  return (
    <div className="landing-preview">
      <span className="landing-preview__star" aria-hidden="true">✳</span>
      <div className="landing-idea-card">
        <svg viewBox="0 0 80 64" fill="none" aria-hidden="true"><rect width="80" height="64" rx="6" fill="#e8ebdd" /><circle cx="57" cy="18" r="9" fill="#e7ac70" /><path d="M0 50 25 22 51 50 67 32 80 47V64H0Z" fill="#a0ae8e" /><path d="m0 54 35-18 45 28H0Z" fill="#718267" /><path d="M38 64c-4-10 15-14 8-22" stroke="#f3e6c6" strokeWidth="3" /><path d="m31 28 7 4-7 4v-8Z" fill="#fff" /></svg>
        <div><span>YOUR NEXT BIG IDEA</span><strong>A slow weekend in Ella</strong><p>Travel & Events · 8 min</p></div>
        <StudioIcon name="spark" width="18" height="18" />
      </div>
      <section className="landing-forecast-card" aria-labelledby="landing-preview-title">
        <div className="landing-forecast-card__header"><span><StudioIcon name="chart" width="16" height="16" /> THE FORECAST CANVAS</span><span className="landing-example-badge">Illustrative example</span></div>
        <div className="landing-forecast-card__metric" aria-live="polite"><p id="landing-preview-title">Estimated views · Day {checkpoint.day}</p><strong>{checkpoint.views}<span>views</span></strong><span className="landing-forecast-card__period">A little look ahead.</span></div>
        <svg className="landing-forecast-card__chart" viewBox="0 0 490 210" role="img" aria-label={`Illustrative cumulative view curve with Day ${checkpoint.day} selected. These are example numbers, not a real prediction.`}>
          <defs><linearGradient id="landing-chart-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#d96b40" stopOpacity=".18" /><stop offset="100%" stopColor="#d96b40" stopOpacity="0" /></linearGradient></defs>
          {[36, 88, 140, 192].map((y) => <path key={y} d={`M12 ${y}H477`} stroke="#e7e5df" strokeDasharray="3 6" />)}
          <path d="M12 188C61 184 84 169 122 152S192 119 232 106S301 83 342 72S414 47 452 36V192H12Z" fill="url(#landing-chart-fill)" />
          <path d="M12 188C61 184 84 169 122 152S192 119 232 106S301 83 342 72S414 47 452 36" stroke="#cc6138" strokeWidth="2.5" fill="none" />
          <path d={`M${checkpoint.x} ${checkpoint.y}V192`} stroke="#d47045" strokeOpacity=".5" strokeDasharray="3 5" />
          {CHECKPOINTS.map((point, i) => <circle key={point.day} cx={point.x} cy={point.y} r={selected === i ? 6 : 4} fill={selected === i ? "#c64924" : "#fff"} stroke={selected === i ? "#fff" : "#d79979"} strokeWidth="2.5" />)}
        </svg>
        <div className="landing-checkpoints" aria-label="Explore the illustrative forecast">
          {CHECKPOINTS.map((point, i) => <button key={point.day} type="button" aria-pressed={selected === i} onClick={() => setSelected(i)}><span>DAY</span> {String(point.day).padStart(2, "0")}</button>)}
        </div>
        <p className="landing-forecast-card__disclaimer">Example numbers, not a real prediction.</p>
      </section>
      <Link href="/forecast" className="landing-preview__note"><StudioIcon name="spark" width="23" height="23" /><span>From “what if”<br /><strong>to what’s next.</strong></span><StudioIcon name="arrow" width="20" height="20" /></Link>
    </div>
  );
}
