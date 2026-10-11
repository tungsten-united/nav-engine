"""Errors of the inference host client (no heavy imports: prepare.py catches these)."""


class InferenceError(RuntimeError):
    """The inference host refused or failed a request (bad input, model mismatch, a failed job)."""


class InferenceUnavailable(InferenceError):
    """The inference host can't be used right now: unreachable, still busy at the deadline, or the Cloudflare Access
    service token was rejected. Never the input's fault, so nothing derived from it may be cached."""


class MissingFeatures(Exception):
    """The inference host doesn't have these images' local features cached: send the images (nav-api then does)."""

    def __init__(self, keys: list[str]):
        super().__init__(f"{len(keys)} images not cached")
        self.keys = keys
