from __future__ import annotations

from typing import List, Tuple


class Instruction:
    """Represent a simplified instruction with an opcode and operands."""

    def __init__(self, opcode: str, operands: Tuple):
        self.opcode = opcode
        self.operands = operands


class SIMTStack:
    """Manage thread masks and reconvergence PCs for divergence handling."""

    def __init__(self) -> None:
        self.stack: List[Tuple[List[bool], int]] = []

    def push(self, mask: List[bool], reconv_pc: int) -> None:
        self.stack.append((mask.copy(), reconv_pc))

    def pop(self) -> Tuple[List[bool], int]:
        return self.stack.pop() if self.stack else ([], -1)

    def top(self) -> Tuple[List[bool], int]:
        return self.stack[-1] if self.stack else ([], -1)


__all__ = ["Instruction", "SIMTStack"]
