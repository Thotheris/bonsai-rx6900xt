"""Fit classification that does not mistake Windows counters for residency."""

import math

_RESIDENCY = {None, "unknown", "device-resident", "spill"}


def decide(exit_code, shared_bytes, *, health_ok=True, residency=None):
    """Classify a fit probe.

    Windows shared-memory counters are retained as raw evidence but are not a
    residency oracle. A healthy process is therefore ``inconclusive`` until an
    independent method establishes device residency or spill.
    """
    if exit_code is not None and (isinstance(exit_code, bool) or not isinstance(exit_code, int)):
        raise ValueError("exit_code must be an integer or None while the server is alive")
    if not isinstance(health_ok, bool):
        raise ValueError("health_ok must be boolean")
    if isinstance(shared_bytes, bool) or not isinstance(shared_bytes, (int, float)):
        raise ValueError("shared_bytes must be numeric")
    shared = float(shared_bytes)
    if not math.isfinite(shared) or shared < 0:
        raise ValueError("shared_bytes must be finite and non-negative")
    if residency not in _RESIDENCY:
        raise ValueError("residency must be unknown, device-resident, or spill")

    if exit_code not in (None, 0) or not health_ok:
        return "exit"
    if residency == "spill":
        return "spill"
    if residency == "device-resident":
        return "fit"
    return "inconclusive"
