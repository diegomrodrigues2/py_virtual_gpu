"""Thread execution unit and register memory abstractions."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from .shared_memory import SharedMemory
from .global_memory import GlobalMemory


class RegisterMemory:
    """Simple key-value store representing registers of a thread."""

    def __init__(self, size_bytes: int) -> None:
        self.size: int = size_bytes
        self._storage: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Basic register operations
    # ------------------------------------------------------------------
    def read(self, name: str) -> Any:
        """Return the value stored in register ``name`` or ``None``."""

        return self._storage.get(name)

    def write(self, name: str, value: Any) -> None:
        """Write ``value`` into register ``name``."""

        self._storage[name] = value

    def clear(self) -> None:
        """Clear all registers."""

        self._storage.clear()


class Thread:
    """Represent a single kernel thread with its own register set."""

    def __init__(
        self,
        thread_idx: Tuple[int, int, int] | None = None,
        block_idx: Tuple[int, int, int] | None = None,
        block_dim: Tuple[int, int, int] | None = None,
        grid_dim: Tuple[int, int, int] | None = None,
        register_mem_size: int = 0,
        shared_mem: SharedMemory | None = None,
        global_mem: GlobalMemory | None = None,
    ) -> None:
        # Indices and dimensions
        self.thread_idx: Tuple[int, int, int] = thread_idx or (0, 0, 0)
        self.block_idx: Tuple[int, int, int] = block_idx or (0, 0, 0)
        self.block_dim: Tuple[int, int, int] = block_dim or (1, 1, 1)
        self.grid_dim: Tuple[int, int, int] = grid_dim or (1, 1, 1)

        # Private register memory
        self.registers = RegisterMemory(register_mem_size)

        # Memory references
        self.shared_mem = shared_mem
        self.global_mem = global_mem

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, kernel_func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Execute ``kernel_func`` for this thread (stub)."""

        raise NotImplementedError(
            "Stub de execução de thread – implementar logicamente na issue 3.x"
        )

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        tx, ty, tz = self.thread_idx
        bx, by, bz = self.block_idx
        return (
            f"<Thread idx=({tx},{ty},{tz}) blk=({bx},{by},{bz}) "
            f"regs={len(self.registers._storage)}>"
        )


__all__ = ["RegisterMemory", "Thread"]
