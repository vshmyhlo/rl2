"""Attach failure causes to the OpenAI client's existing retry messages."""

import logging
import sys

import httpx


def add_retry_cause(record: logging.LogRecord) -> bool:
    # The SDK logs retries inside the exception handler, including across awaits.
    error = sys.exception()
    if error is not None and isinstance(record.msg, str) and record.msg.startswith("Retrying request to "):
        cause = type(error).__name__
        if isinstance(error, httpx.HTTPStatusError):
            cause += f" status={error.response.status_code}"
        record.msg = f"{record.getMessage()} (cause={cause})"
        record.args = ()
    return True
