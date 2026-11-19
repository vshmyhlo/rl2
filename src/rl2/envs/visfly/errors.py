"""Recoverable VisFly scene geometry failures."""


class NoSamplingVolumeError(ValueError):
    """Scene geometry has no flight volume within the configured bounds."""
