"""Orchestration bounds, separate from all forecast/personalization math."""
import asyncio
from app.personalization import synchronize_creator_history
from app.creator_analytics import CreatorSyncException
from app.youtube import ChannelLookupException

async def bounded_creator_sync(**kwargs):
    try:
        async with asyncio.timeout(300):
            return await synchronize_creator_history(**kwargs)
    except TimeoutError:
        raise CreatorSyncException("Creator synchronization exceeded its time budget") from None

async def bounded_channel_lookup(call, *args):
    try:
        return await asyncio.wait_for(asyncio.to_thread(call, *args), timeout=30)
    except TimeoutError:
        raise ChannelLookupException("Channel statistics are currently unavailable.",
                                     "channel_stats_unavailable", 502) from None
