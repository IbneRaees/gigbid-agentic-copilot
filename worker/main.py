"""Worker container entrypoint (`python -m worker.main`).

Starts the Celery worker when Redis is reachable, or falls back to a graceful
periodic local scouting daemon in offline/standalone environments.
"""

from __future__ import annotations

import logging
import os
import time

from worker.tasks import celery_app, scout_and_prepare_bids

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("gigbid.worker")


def main() -> None:
    mode = os.getenv("WORKER_MODE", "celery").lower()
    if mode == "celery":
        try:
            celery_app.worker_main(argv=["worker", "--loglevel=INFO", "--concurrency=2"])
            return
        except Exception as exc:
            logger.warning(
                "Celery worker failed to connect (%s); entering periodic idle loop.", exc
            )

    interval = int(os.getenv("WORKER_POLL_SECONDS", "300"))
    while True:
        try:
            summary = scout_and_prepare_bids("persona_aiml_01", limit=3)
            logger.info("Offline scout cycle complete: %s", summary)
        except Exception as exc:  # pragma: no cover
            logger.warning("Worker cycle error: %s", exc)
        time.sleep(interval)


if __name__ == "__main__":
    main()
