"""
ais/main.py — the AIS WebSocket daemon's standalone entrypoint. Owns the
connection lifecycle only: connect, hand messages off to
clients/aisstream.py, decide whether a dropped connection is a routine
transient event or a sustained failure, and respond to SIGTERM. Message
parsing, vessel/position/port-call writes, and arrival/departure
detection stay in clients/aisstream.py — see that module's docstring and
CLAUDE.md's AIS daemon section for the full split.

Run with:
    python -m ais.main

Docker's `restart: unless-stopped` is the supervisor now — this process
must not decide its own persistent failure is survivable. A transient
WebSocket drop is handled here, in-process, with capped exponential
backoff (see run_forever()); sustained inability to connect — more than
MAX_CONSECUTIVE_FAILURES attempts in a row — exits non-zero and lets
Docker restart the container. SIGTERM (Docker's stop signal, sent before
SIGKILL) triggers a clean shutdown and exit 0, so a deliberate
`docker compose down` never reads as a crash in the logs.

This daemon writes nothing to job_runs: it's a continuous process, not a
scheduled unit that returns, and a process reporting its own liveness is
exactly the pattern that let Prefect flows record "Completed" through a
dead stack. AIS liveness has to be observed from outside, by something
that checks whether positions are actually still arriving.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from typing import Awaitable, Callable

from clients.aisstream import connect_once, ensure_port_cache_loaded
from clients.base import logger

INITIAL_BACKOFF_SECONDS = 5.0
MAX_BACKOFF_SECONDS = 60.0
MAX_CONSECUTIVE_FAILURES = 10

Connect = Callable[[], Awaitable[None]]


class ConnectionFailureLimitExceeded(RuntimeError):
    """
    Raised by run_forever() after MAX_CONSECUTIVE_FAILURES consecutive
    failed connection attempts. Deliberately never caught anywhere in
    this module's own async machinery — it must propagate out of main()
    and become a non-zero process exit, so Docker's restart policy is
    what recovers from sustained inability to connect. A process must
    not be the thing that decides its own persistent failure is
    survivable.
    """


async def run_forever(
    connect: Connect,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
    max_backoff: float = MAX_BACKOFF_SECONDS,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    max_attempts: int | None = None,
) -> int:
    """
    Drive `connect()` — a zero-argument coroutine performing exactly one
    AISStream connection attempt — in a loop with capped exponential
    backoff.

    A transient drop is handled in-process: whether connect() raises or
    simply returns (the remote end closed cleanly), we wait and retry.
    Only consecutive *raised* failures count toward
    max_consecutive_failures; a single successful connection — however
    briefly it lasted — resets both the failure count and the backoff
    interval back to their starting values. Only hitting the ceiling
    escalates, by raising ConnectionFailureLimitExceeded, to the
    supervisor.

    `max_attempts` bounds the loop for tests only; production code
    (main(), below) never passes it, so this runs until cancelled or the
    ceiling is hit. Returns the final consecutive-failure count when
    max_attempts is exhausted (the test-only exit path).
    """
    consecutive_failures = 0
    backoff = initial_backoff
    attempts = 0

    while max_attempts is None or attempts < max_attempts:
        attempts += 1
        try:
            await connect()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                raise ConnectionFailureLimitExceeded(
                    f"{consecutive_failures} consecutive AIS connection "
                    "failures — escalating to the supervisor"
                ) from exc
            logger.warning(
                "AIS connection attempt failed (%d/%d consecutive): %s — retrying in %.0fs",
                consecutive_failures, max_consecutive_failures, exc, backoff,
            )
            await sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
        else:
            if consecutive_failures:
                logger.info("AIS connection recovered — resetting failure count")
            consecutive_failures = 0
            backoff = initial_backoff
            await sleep(backoff)

    return consecutive_failures


def _install_shutdown_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    """
    SIGTERM is Docker's stop signal and the one CLAUDE.md's contract is
    about (decision 5). SIGINT (Ctrl-C) is handled the same way purely
    for interactive convenience during manual verification — Python's
    default SIGINT behavior is to raise KeyboardInterrupt, which would
    unwind through asyncio.run() as an unhandled exception instead of
    through this module's clean-shutdown path.
    """
    def _handle_shutdown(sig: signal.Signals) -> None:
        logger.info("ais: %s received — closing connection and exiting cleanly", sig.name)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_shutdown, sig)


async def serve(connect: Connect = connect_once) -> None:
    """
    Run the AIS daemon until either SIGTERM arrives (clean exit) or
    run_forever() raises ConnectionFailureLimitExceeded (propagated to
    the caller, which maps it to a non-zero exit code).
    """
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _install_shutdown_handlers(loop, stop_event)

    ensure_port_cache_loaded()

    connection_task = asyncio.ensure_future(run_forever(connect))
    stop_task = asyncio.ensure_future(stop_event.wait())

    done, _pending = await asyncio.wait(
        {connection_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if connection_task in done:
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task
        connection_task.result()  # re-raises ConnectionFailureLimitExceeded, if that's why
        return

    # stop_event fired: SIGTERM. Cancelling lets connect()'s
    # `async with websockets.connect(...)` / `with Session()` context
    # managers unwind on the way out — that's what closes the socket and
    # ends any open DB transaction cleanly.
    connection_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await connection_task


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    try:
        asyncio.run(serve())
    except ConnectionFailureLimitExceeded as exc:
        logger.error("ais: %s", exc)
        return 1
    logger.info("ais: exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
