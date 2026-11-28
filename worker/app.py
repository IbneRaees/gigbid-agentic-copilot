"""Celery application alias for `celery -A worker.app worker`."""

from worker.tasks import celery_app
from worker.tasks import celery_app as app

__all__ = ["app", "celery_app"]
