from abc import ABC, abstractmethod
from typing import Callable, Any


class Executor(ABC):
    @abstractmethod
    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        pass


class LocalExecutor(Executor):
    def __call__(self, func: Callable[[Any], None], *args, **kwargs):
        return func(*args, **kwargs)
