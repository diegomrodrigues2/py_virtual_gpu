from __future__ import annotations

from typing import List, TYPE_CHECKING


def is_coalesced(addrs: List[int], size: int) -> bool:
    """Return ``True`` if ``addrs`` form a contiguous vector of ``size`` bytes."""

    sorted_addrs = sorted(addrs)
    return all(
        sorted_addrs[i] + size == sorted_addrs[i + 1]
        for i in range(len(sorted_addrs) - 1)
    )

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

    def execute(self) -> bool:
        """Execute one instruction and detect divergence conceptually.

        Returns
        -------
        bool
            ``True`` while at least one thread in the warp remains active,
            otherwise ``False`` indicating the warp finished execution.
        """

        # Fetch the instruction pointed by the current ``pc``.
        inst = self.fetch_next_instruction()

        # Evaluate the branch predicate for each thread and check if the
        # resulting mask differs from the current active mask.
        mask_before = self.active_mask.copy()
        predicate = self.evaluate_predicate(inst)
        if predicate != mask_before:
            self.handle_divergence(predicate)
            self.sm.record_divergence(self, self.pc, mask_before, self.active_mask)

        # Execute the instruction only for threads that are active under the
        # updated mask. ``run_step`` is treated as an optional helper used by
        # the tests; other implementations may provide a different method.
        for thread, active in zip(self.threads, self.active_mask):
            if not active:
                continue
            step = getattr(thread, "run_step", None)
            if callable(step):
                step(inst)

        if inst.opcode.startswith(("LD", "ST")) or inst.opcode in {
            "LOAD",
            "STORE",
        }:
            addr_op = inst.operands[0] if len(inst.operands) > 0 else 0
            size = inst.operands[1] if len(inst.operands) > 1 else 0
            space = inst.operands[2] if len(inst.operands) > 2 else "global"
            addr_list: List[int] = []
            for thread, active in zip(self.threads, self.active_mask):
                if not active:
                    continue
                addr = addr_op
                if callable(addr_op):
                    addr = addr_op(thread)
                elif hasattr(addr_op, "offset"):
                    addr = getattr(addr_op, "offset")
                elif isinstance(addr_op, str) and hasattr(thread, addr_op):
                    addr = getattr(thread, addr_op)
                addr_list.append(int(addr))
            self.memory_access(addr_list, size, space)

        self.sm.account_instruction(inst)
        self.pc += 1

        # Check if we've reached a reconvergence point recorded on the SIMT
        # stack. If so, restore the previous mask and record the event.
        top_mask, reconv_pc = self.simt_stack.top()
        if reconv_pc != -1 and self.pc == reconv_pc:
            prev_mask, _ = self.simt_stack.pop()
            mask_before = self.active_mask.copy()
            self.active_mask = prev_mask
            if mask_before != self.active_mask:
                self.sm.record_divergence(self, self.pc, mask_before, self.active_mask)

        # ``True`` means there are still active threads; ``False`` means the SM
        # can discard this warp from the scheduling queue.
        return any(self.active_mask)
    def issue_instruction(self, inst: Instruction) -> None:
        """Issue ``inst`` to the active threads (conceptual stub)."""
        self.sm.account_instruction(inst)
        self.pc += 1

    def memory_access(self, addr_list: List[int], size: int, space: str = "global") -> bytes:
        """Conceptually access memory addresses and record efficiency statistics."""

        if not is_coalesced(addr_list, size):
            self.sm.counters.setdefault("non_coalesced_accesses", 0)
            self.sm.counters["non_coalesced_accesses"] += 1
            self.sm.stats.setdefault("extra_cycles", 0)
            self.sm.stats["extra_cycles"] += 1

        if space == "shared" and hasattr(self.sm.shared_mem, "detect_bank_conflicts"):
            conflicts = self.sm.shared_mem.detect_bank_conflicts(addr_list)
            if conflicts > 0:
                self.sm.counters.setdefault("bank_conflicts", 0)
                self.sm.counters["bank_conflicts"] += conflicts
                self.sm.stats.setdefault("extra_cycles", 0)
                self.sm.stats["extra_cycles"] += conflicts - 1
        return b""

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


__all__ = ["Warp", "is_coalesced"]
