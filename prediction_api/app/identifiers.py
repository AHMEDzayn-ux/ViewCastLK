"""Bound public channel identifiers without blocking international handles."""
import re
import unicodedata
from urllib.parse import unquote, urlsplit

CHANNEL_ID = re.compile(r"UC[A-Za-z0-9_-]{22}\Z")

def parse_channel_identifier(value: str) -> tuple[str, str]:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 512:
        raise ValueError("invalid_channel_identifier")
    cleaned = value.strip()
    if any(char.isspace() for char in cleaned):
        raise ValueError("The channel URL or identifier should not contain spaces.")
    if "://" in cleaned or cleaned.lower().startswith(("youtube.com/", "www.youtube.com/", "m.youtube.com/")):
        url = urlsplit(cleaned if "://" in cleaned else "https://" + cleaned)
        if (url.scheme not in {"http", "https"} or url.hostname not in
            {"youtube.com", "www.youtube.com", "m.youtube.com"} or url.username or url.password or url.port):
            raise ValueError("invalid_channel_identifier")
        cleaned = unquote(url.path).strip("/")
    if cleaned.startswith("channel/"):
        cleaned = cleaned[8:]
        if not CHANNEL_ID.fullmatch(cleaned): raise ValueError("invalid_channel_identifier")
        return "id", cleaned
    if cleaned.startswith("UC"):
        if not CHANNEL_ID.fullmatch(cleaned): raise ValueError("invalid_channel_identifier")
        return "id", cleaned
    if cleaned.startswith(("c/", "user/")):
        cleaned = cleaned.split("/", 1)[1]
    handle = cleaned.removeprefix("@")
    # Generous bound for mixed-script/legacy handles; YouTube resolves existence.
    if not 1 <= len(handle) <= 100 or any(
        not (unicodedata.category(char)[0] in "LMN" or char in "._-·") for char in handle
    ):
        raise ValueError("invalid_channel_identifier")
    return "handle", "@" + handle
