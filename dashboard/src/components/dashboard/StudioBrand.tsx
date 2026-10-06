import Link from "next/link";

export default function StudioBrand() {
  return (
    <Link className="studio-brand" href="/" aria-label="ViewCastLK home">
      <span className="studio-brand__symbol" aria-hidden="true"><i /><i /><i /></span>
      <span>ViewCast<span className="studio-brand__lk">LK</span><small>CREATOR STUDIO</small></span>
    </Link>
  );
}
