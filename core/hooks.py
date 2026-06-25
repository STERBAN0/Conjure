"""Tiny synchronous pub/sub bus.

Effects subscribe to events ("frame", "release", "tear_open", ...) and
modules `emit` them. Keeping it synchronous avoids threading hazards in the
render loop; subscribers must stay cheap.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


class HookBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[..., None]]] = defaultdict(list)

    def on(self, event: str, fn: Callable[..., None]) -> None:
        self._subs[event].append(fn)

    def off(self, event: str, fn: Callable[..., None]) -> None:
        subs = self._subs.get(event, [])
        if fn in subs:
            subs.remove(fn)

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        # Iterate over a snapshot so subscribers may mutate during dispatch.
        for fn in list(self._subs.get(event, ())):
            try:
                fn(*args, **kwargs)
            except Exception:  # never let one effect crash the loop
                log.warning("subscriber for '%s' raised", event, exc_info=True)
