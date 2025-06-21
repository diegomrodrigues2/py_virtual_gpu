from __future__ import annotations

from typing import List

from .thread import Thread


class Warp:
    """Represent a group of threads executing in lock-step."""

    def __init__(self, id: int, threads: List[Thread]):
        self.id: int = id
        self.threads: List[Thread] = threads
        self.active_mask: List[bool] = [True] * len(threads)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self) -> None:
        """Execute one instruction for all active threads (stub)."""

        raise NotImplementedError(
            "Execu\u00e7\u00e3o de warp stub – implementar em 3.x"
        )

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        active = sum(self.active_mask)
        return f"<Warp id={self.id} size={len(self.threads)} active={active}>"


__all__ = ["Warp"]
