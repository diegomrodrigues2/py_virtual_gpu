from __future__ import annotations

from typing import List

from .dispatch import Instruction, SIMTStack
from .thread import Thread



class Warp:
    """Represent a group of threads executing in lock-step."""

    def __init__(self, id: int, threads: List[Thread]):
        self.id: int = id
        self.threads: List[Thread] = threads
        self.active_mask: List[bool] = [True] * len(threads)
        self.pc: int = 0
        self.simt_stack = SIMTStack()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self) -> None:
        """Execute one instruction for all active threads (stub)."""

        raise NotImplementedError(
            "Execu\u00e7\u00e3o de warp stub – implementar em 3.x"
        )
    def issue_instruction(self, inst: Instruction) -> None:
        """Issue ``inst`` to the active threads (conceptual stub)."""
        self.pc += 1
        raise NotImplementedError("Dispatch de instru\u00e7\u00e3o stub")

    def handle_divergence(self, predicate: List[bool]) -> None:
        """Handle control-flow divergence for this warp."""
        reconv_pc = self.pc
        self.simt_stack.push(self.active_mask, reconv_pc)
        self.active_mask = predicate.copy()


    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        active = sum(self.active_mask)
        return f"<Warp id={self.id} size={len(self.threads)} active={active}>"


__all__ = ["Warp"]
