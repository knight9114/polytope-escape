"""Learning Rate Schedules"""

# -------------------------------------------------------------------------
#   Exports
# -------------------------------------------------------------------------

__all__: list[str] = []


# -------------------------------------------------------------------------
#   Imports
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
#   Schedule Functions
# -------------------------------------------------------------------------


def constant(step, *, lr: float) -> float:
    return lr
