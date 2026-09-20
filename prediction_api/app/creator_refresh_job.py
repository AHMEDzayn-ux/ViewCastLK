"""Weekly creator refresh entry point used by the scheduled workflow."""

from __future__ import annotations

import asyncio

from app.creator_lifecycle import run_creator_refresh_job
from app.creator_store import CreatorStore
from app.model_registry import ModelRegistry
from app.public_roster import PublicRosterStore


async def _run() -> dict[str, int]:
    return await run_creator_refresh_job(
        store=CreatorStore(),
        model_registry=ModelRegistry(),
        roster_store=PublicRosterStore(),
    )


def main() -> None:
    results = asyncio.run(_run())
    print(
        "Creator refresh complete: "
        f"{results['refreshed']} refreshed, "
        f"{results['revoked']} revoked, "
        f"{results['temporary_failure']} temporary failure(s)."
    )
    if results["temporary_failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
