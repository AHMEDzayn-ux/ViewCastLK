"""The list of channels being tracked. Add or remove channels by editing
channel_handles.txt (one @handle per line) — no code changes needed to grow
the roster.
"""


def load_handles(path: str = "channel_handles.txt") -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]
