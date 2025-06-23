from __future__ import annotations

from typing import List, TYPE_CHECKING

from .dispatch import Instruction, SIMTStack
from .thread import Thread

if TYPE_CHECKING:  # pragma: no cover - circular import typing helper
    from .streaming_multiprocessor import StreamingMultiprocessor



class Warp:
    """Represent a group of threads executing in lock-step."""

    def __init__(self, id: int, threads: List[Thread], sm: "StreamingMultiprocessor"):
        self.id: int = id
        self.threads: List[Thread] = threads
        self.sm = sm
        self.active_mask: List[bool] = [True] * len(threads)
        self.pc: int = 0
        self.simt_stack = SIMTStack()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def fetch_next_instruction(self) -> Instruction:
        """Fetch the next instruction for this warp (stub)."""

        return Instruction("NOP", tuple())

    def evaluate_predicate(self, inst: Instruction) -> List[bool]:
        """Evaluate branch predicate for ``inst`` (stub)."""

        return self.active_mask.copy()

    def execute(self) -> None:
        """Execute one instruction and detect divergence conceptually."""

        inst = self.fetch_next_instruction()
        mask_before = self.active_mask.copy()
        predicate = self.evaluate_predicate(inst)
        if predicate != mask_before:
            self.handle_divergence(predicate)
            self.sm.record_divergence(self, self.pc, mask_before, self.active_mask)
        self.pc += 1
    def issue_instruction(self, inst: Instruction) -> None:
        """Issue ``inst`` to the active threads (conceptual stub)."""
        self.pc += 1

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
