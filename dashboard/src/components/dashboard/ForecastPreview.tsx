"use client";

import Link from "next/link";
import { useState } from "react";
import StudioIcon from "./StudioIcon";

const CHECKPOINTS = [
  { day: 7, label: "The first impression", text: "An estimate of the total views your video could reach in its first week.", x: 120, y: 168 },
  { day: 14, label: "Finding its audience", text: "See the cumulative estimate two weeks after your video goes live.", x: 240, y: 116 },
  { day: 21, label: "The bigger picture", text: "Look beyond the first burst with a three-week cumulative estimate.", x: 360, y: 82 },
  { day: 30, label: "A month of possibility", text: "Bring the full month into view with your final planning checkpoint.", x: 480, y: 50 },
];

export default function ForecastPreview({ onExample, isAuthenticated }: { onExample: () => void; isAuthenticated: boolean }) {
  const [selected, setSelected] = useState(3);
  const checkpoint = CHECKPOINTS[selected];

  return (
    <>
      <section className="preview-panel" aria-labelledby="preview-title">
        <div className="panel-heading"><span className="panel-icon"><StudioIcon name="chart" /></span><div><h2 id="preview-title">Your forecast canvas</h2><p>A little foresight. A better starting point.</p></div><span className="waiting-badge">Awaiting your idea</span></div>
        <div className="preview-metric"><div><p>Estimated views · Day {checkpoint.day}</p><strong aria-label="No forecast yet">— <span>views</span></strong></div><span className="preview-metric__icon"><StudioIcon name="spark" width="28" height="28" /></span></div>
        <div className="preview-graph">
          <span className="preview-graph__label">YOUR NEXT 30 DAYS</span>
          <svg viewBox="0 0 530 240" role="img" aria-label="Illustration of a forecast curve. No forecast data has been generated.">
            <defs><linearGradient id="preview-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ed713d" stopOpacity=".14" /><stop offset="100%" stopColor="#ed713d" stopOpacity="0" /></linearGradient></defs>
            {[50, 104, 158, 212].map((y) => <path key={y} d={`M20 ${y}H510`} stroke="#e7e5df" strokeDasharray="3 6" />)}
            <path d="M20 205C62 198 74 186 120 168S196 129 240 116S315 92 360 82S433 58 480 50L480 212H20Z" fill="url(#preview-fill)" />
            <path className="preview-graph__curve" d="M20 205C62 198 74 186 120 168S196 129 240 116S315 92 360 82S433 58 480 50" fill="none" stroke="#df9d7f" strokeWidth="2.5" strokeDasharray="5 7" />
            <path d={`M${checkpoint.x} ${checkpoint.y}V212`} stroke="#d47045" strokeOpacity=".5" strokeDasharray="3 5" />
            {CHECKPOINTS.map((point, i) => <circle key={point.day} cx={point.x} cy={point.y} r={i === selected ? 6 : 4} fill={i === selected ? "#db572b" : "#fff"} stroke={i === selected ? "#fff" : "#dbb59f"} strokeWidth="2.5" />)}
            {CHECKPOINTS.map((point) => <text key={point.day} x={point.x} y="237" textAnchor="middle" fontSize="11" fill="#85847d">Day {point.day}</text>)}
          </svg>
          <span className="preview-graph__disclaimer">Illustrative curve · Your results will appear here</span>
        </div>
        <div className="checkpoint-picker" aria-label="Explore forecast checkpoints">
          {CHECKPOINTS.map((point, i) => <button key={point.day} type="button" aria-label={`Day ${point.day}`} aria-pressed={selected === i} onClick={() => setSelected(i)}><span>DAY</span><strong>{String(point.day).padStart(2, "0")}</strong></button>)}
        </div>
        <div className="checkpoint-description" aria-live="polite"><span className="checkpoint-description__dot" /><div><h3>{checkpoint.label}</h3><p>{checkpoint.text}</p></div></div>
        <div className="preview-panel__footer"><span>Curious what a result looks like?</span><button type="button" onClick={onExample}>Try an example <StudioIcon name="arrow" width="16" height="16" /></button></div>
      </section>
      <section className="personal-studio-card">
        <span className="personal-studio-card__icon"><StudioIcon name="channel" width="26" height="26" /></span>
        <div><p className="section-kicker">A forecast with your perspective</p><h2>Every channel has its own story.</h2><p>Connect yours to use eligible channel history for personal adjustments.</p><Link href={isAuthenticated ? "/account" : "/login?next=%2Faccount"}>{isAuthenticated ? "Manage your channel" : "Sign in to connect your channel"}<StudioIcon name="arrow" width="16" height="16" /></Link></div>
        <StudioIcon name="spark" className="personal-studio-card__decoration" width="74" height="74" />
      </section>
      <p className="studio-privacy-note"><StudioIcon name="shield" width="15" height="15" /> Your private Analytics never train the shared model.</p>
    </>
  );
}
