from typing import Optional
from collections.abc import Iterable

__all__ = ["batch_iterator"]


def batch_iterator(total: int, batch_size: Optional[int]) -> Iterable[tuple[int, int]]:
    assert total > 0, f"batch_iterator called with {total} elements"
    if batch_size is None:
        return [(0, total)]
    delim = list(range(0, total, batch_size)) + [total]
    return zip(delim[:-1], delim[1:])
