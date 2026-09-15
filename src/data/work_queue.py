"""Shared-queue runner for multi-endpoint (multi-GPU) annotation / rewrite.

Static sharding (`--num-shards`) splits the work in half up front, so when one
GPU is faster it finishes early and idles. This runner instead fills one
asyncio queue with all pending records and lets every worker pull the next item
when it is free. Faster GPUs naturally process more, and the workers finish
together.

Used by ``data.vlm_caption.run_annotation`` and
``data.prompt_diversify.run_rewrite`` when more than one endpoint is given.
"""
from __future__ import annotations

import asyncio
import time


async def run_queued(records, backends, concurrency, worker, is_ok, write_result,
                     log, progress_every=50, label="done"):
    """Run ``worker(backend, record)`` over a shared queue.

    Args:
        records: list of work items (already in memory).
        backends: one backend per endpoint; ``concurrency`` workers per backend.
        worker: async ``(backend, record) -> result``.
        is_ok: ``result -> bool``.
        write_result: ``result -> None`` (persists the result; called under lock).
    """
    queue: asyncio.Queue = asyncio.Queue()
    for record in records:
        queue.put_nowait(record)
    total = len(records)
    stats = {"ok": 0, "failed": 0}
    lock = asyncio.Lock()
    started = time.time()

    async def loop(backend):
        while True:
            try:
                record = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = await worker(backend, record)
            async with lock:
                ok = bool(is_ok(result))
                stats["ok" if ok else "failed"] += 1
                write_result(result)
                written = stats["ok"] + stats["failed"]
                elapsed = max(time.time() - started, 1e-6)
                rate = written / elapsed
                eta = (total - written) / rate if rate else 0.0
                if progress_every and written % progress_every == 0:
                    log(f"progress {written}/{total} ok={stats['ok']} failed={stats['failed']} "
                        f"rate={rate:.2f}/s eta={eta / 60:.1f}min")

    try:
        await asyncio.gather(*[
            asyncio.create_task(loop(backend))
            for backend in backends for _ in range(max(1, int(concurrency)))
        ])
    finally:
        for backend in backends:
            try:
                await backend.aclose()
            except Exception:
                pass
    stats["total"] = total
    stats["elapsed_s"] = round(time.time() - started, 1)
    return stats
