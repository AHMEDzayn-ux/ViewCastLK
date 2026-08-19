"""Download YouTube thumbnails and reduce each to a row of image features.

WHY
The EDA can predict only R^2 = 0.072 of the within-channel residual from
pre-publication metadata. Roughly 93% of what makes one video beat another from
the same channel is not in the dataset, and the thumbnail is the largest thing
missing: it is the only part of the pre-publication package a viewer actually
looks at before deciding.

COST
No API quota. Thumbnails come from i.ytimg.com, an ordinary CDN, not the
YouTube Data API -- nothing here touches the 10,000-unit daily budget. The cost
is bandwidth: hqdefault is ~15 KB, so 30,000 videos is roughly 450 MB
downloaded.

Images are NOT kept. Each is decoded, measured and discarded, so the only thing
that persists is a small feature table. Keeping 30,000 JPEGs to compute the
same numbers again later would be storage this project does not have.

THE CAVEAT THAT MUST TRAVEL WITH THE RESULTS
YouTube serves a replaced thumbnail from the same URL, so for an older video
the image fetched today may not be the one it launched with. That is also why
thumbnail *changes* cannot be detected at all. Prefer --max-age-days to bound
the exposure: the more recently a video was published, the likelier the image
is the original.

FEATURES
Deliberately cheap and interpretable rather than a neural embedding: a creator
can act on "your thumbnail is dark and has no face", not on dimension 47 of a
CLIP vector. Faces come from OpenCV's Haar cascade, which is weak on small or
side-on faces -- treat face_count as a lower bound.

Usage:
    python Analysis/fetch_thumbnails.py --limit 500            # try it
    python Analysis/fetch_thumbnails.py --max-age-days 21      # the real run
"""
import argparse
import io
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import dataset_path
DATA = dataset_path()
OUT = os.path.join(HERE, "thumbnail_features.parquet")

WORKERS = 16
TIMEOUT = 15
RETRIES = 2
UA = "ViewCastLK/1.0 (university research; thumbnail feature extraction)"

def _load_face_detector():
    """Haar cascade, or None if this OpenCV build has no cascades.

    OpenCV 5.0 removed the legacy objdetect API and stopped shipping the
    cascade XML files, so cv2.CascadeClassifier does not exist there at all.
    Requirements pin <5 for that reason, but a future environment may not, and
    losing one feature is a far better outcome than the whole extraction
    refusing to start."""
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.isfile(path):
            return None
        clf = cv2.CascadeClassifier(path)
        return None if clf.empty() else clf
    except AttributeError:
        return None


_face = _load_face_detector()
if _face is None:
    print("WARNING: no Haar cascade available -- face features will be null.\n"
          "         pip install 'opencv-python-headless<5' to restore them.",
          file=sys.stderr)
_lock = threading.Lock()


def features(buf):
    """Reduce one JPEG to interpretable measurements. Never raises."""
    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    b, g, r = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), \
        arr[:, :, 2].astype(float)

    # Hasler-Susstrunk colourfulness: how vivid the image reads at a glance.
    rg, yb = r - g, 0.5 * (r + g) - b
    colourfulness = (np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                     + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    if _face is not None:
        faces = _face.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
        face_area = (sum(fw * fh for _, _, fw, fh in faces) / (w * h)
                     if len(faces) else 0.0)
        n_faces, has_face = int(len(faces)), bool(len(faces) > 0)
    else:
        n_faces, has_face, face_area = None, None, None

    edges = cv2.Canny(gray, 100, 200)
    # Text overlays are mostly high-contrast horizontal structure; edge density
    # in the middle band is a crude but honest proxy for "has big text on it".
    band = edges[int(h * .25):int(h * .75), :]

    return {
        "img_w": w, "img_h": h,
        "brightness": float(gray.mean()) / 255,
        "contrast": float(gray.std()) / 255,
        "saturation": float(hsv[:, :, 1].mean()) / 255,
        "hue_mean": float(hsv[:, :, 0].mean()),
        "colourfulness": float(colourfulness),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float(edges.mean()) / 255,
        "text_band_density": float(band.mean()) / 255,
        "face_count": n_faces,
        "has_face": has_face,
        "face_area_ratio": face_area,
        "warm_ratio": float((r.mean() + 1) / (b.mean() + 1)),
        "dark_share": float((gray < 60).mean()),
        "bright_share": float((gray > 200).mean()),
    }


def fetch_one(vid, url):
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                buf = r.read()
            f = features(buf)
            if f is None:
                return None
            f["video_id"] = vid
            f["bytes"] = len(buf)
            return f
        except Exception:
            if attempt == RETRIES:
                return None
            time.sleep(0.4 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap videos processed")
    ap.add_argument("--max-age-days", type=int, default=0,
                    help="only videos published within this many days, which "
                         "bounds the replaced-thumbnail problem")
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    df = pd.read_parquet(DATA, columns=["video_id", "published_at", "eligible",
                                        f"d{args.horizon}_usable",
                                        "thumbnail_url"])
    want = df[df.eligible & df[f"d{args.horizon}_usable"]].copy()
    if args.max_age_days:
        cut = want.published_at.max() - pd.Timedelta(days=args.max_age_days)
        want = want[want.published_at >= cut]

    # Resume: never re-download what has already been measured.
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_parquet(args.out, columns=["video_id"]).video_id)
        print(f"{len(done):,} already extracted, resuming")
    want = want[~want.video_id.isin(done)]
    if args.limit:
        want = want.head(args.limit)

    if want.empty:
        print("nothing to do")
        return

    mb = len(want) * 15 / 1024
    print(f"{len(want):,} thumbnails to fetch (~{mb:.0f} MB, no API quota)")

    rows, failed, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, v, u): v
                for v, u in zip(want.video_id, want.thumbnail_url)}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                rows.append(r)
            else:
                failed += 1
            if i % 500 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i:,}/{len(want):,}  ok={len(rows):,} fail={failed:,}  "
                      f"{rate:.0f}/s  eta {(len(want)-i)/rate/60:.1f} min")

    new = pd.DataFrame(rows)
    if os.path.exists(args.out) and len(done):
        new = pd.concat([pd.read_parquet(args.out), new], ignore_index=True)
    new.to_parquet(args.out, index=False, compression="zstd")

    print(f"\nwrote {args.out}: {len(new):,} rows, {len(new.columns)} columns")
    print(f"failed: {failed:,} ({failed/max(1,len(want)):.1%})")
    print(f"elapsed: {(time.time()-t0)/60:.1f} min")
    print("\nImages were not retained -- only these measurements.")


if __name__ == "__main__":
    main()
