"""Tiny synchronous pub/sub bus.

Effects subscribe to events ("frame", "release", "tear_open", ...) and
modules `emit` them. Keeping it synchronous avoids threading hazards in the
render loop; subscribers must stay cheap.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Callable, Any


class HookBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[..., None]]] = defaultdict(list)

    def on(self, event: str, fn: Callable[..., None]) -> None:
        self._subs[event].append(fn)

    def off(self, event: str, fn: Callable[..., None]) -> None:
        if fn in self._subs[event]:
            self._subs[event].remove(fn)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        # Iterate over a snapshot so subscribers may mutate during dispatch.
        for fn in list(self._subs.get(event, ())):
            try:
                fn(*args, **kwargs)
            except Exception as e:  # never let one effect crash the loop
                print(f"[hooks] subscriber for '{event}' raised: {e!r}")
