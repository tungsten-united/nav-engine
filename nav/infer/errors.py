"""Errors of the inference host client (no heavy imports: prepare.py catches these)."""


class InferenceError(RuntimeError):
    """The inference host refused or failed a request (bad input, model mismatch, a failed job)."""


class InferenceUnavailable(InferenceError):
    """The inference host can't be used right now: unreachable, still busy at the deadline, or the Cloudflare Access
    service token was rejected. Never the input's fault, so nothing derived from it may be cached."""
