import collections
from typing import Any, Callable

EventListener = Callable[..., Any]


class EventEmitter:
    listeners: dict[str, list[EventListener]]

    def __init__(self):
        self.listeners = collections.defaultdict(list)

    def add_listener(
        self,
        name: str,
        fn: EventListener,
        prepend: bool = False,
    ) -> None:
        listeners = self.listeners[name]

        if prepend:
            listeners.insert(0, fn)
        else:
            listeners.append(fn)

    def emit(
        self,
        /,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        listeners = self.listeners.get(name)
        if listeners is None:
            return
        if kwargs is None:
            kwargs = {}
        for fn in listeners:
            fn(*args, **kwargs)
