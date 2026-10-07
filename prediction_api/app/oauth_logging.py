"""Redact Google's callback query from application access logs only."""

import logging


class OAuthCallbackAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Uvicorn's access args are client, method, path+query, version, status.
        # Do not alter request.scope: the callback still needs its Google code.
        if isinstance(record.args, tuple) and len(record.args) == 5:
            target = record.args[2]
            if isinstance(target, str):
                path = target.split("?", 1)[0]
                if path.rstrip("/") == "/auth/youtube/callback":
                    args = list(record.args)
                    args[2] = path
                    record.args = tuple(args)
        return True


def protect_oauth_access_logs() -> None:
    logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, OAuthCallbackAccessFilter) for item in logger.filters):
        logger.addFilter(OAuthCallbackAccessFilter())
