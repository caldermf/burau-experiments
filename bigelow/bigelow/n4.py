from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, Tuple


@dataclass(frozen=True)
class N4StandardForm:
    top_single_left: int
    top_single_middle: int
    top_single_right: int
    top_double_left: int
    endpoints: Tuple[int, int]

    def upper_crossings(self) -> int:
        ordinary = self.top_single_left + self.top_single_middle + self.top_single_right + self.top_double_left
        top_endpoints = sum(1 for endpoint in self.endpoints if endpoint in (1, 2, 3))
        return 2 * ordinary + top_endpoints

    def lower_arches(self) -> int:
        total_hits = self.upper_crossings()
        bottom_endpoints = sum(1 for endpoint in self.endpoints if endpoint == 4)
        if total_hits < bottom_endpoints:
            raise ValueError("inconsistent endpoint data")
        return (total_hits - bottom_endpoints) // 2

    def total_intersections(self) -> int:
        return self.upper_crossings()


def enumerate_standard_forms(max_intersections: int) -> Iterator[N4StandardForm]:
    for counts in product(range(max_intersections + 1), repeat=4):
        for endpoints in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)):
            form = N4StandardForm(*counts, endpoints=endpoints)
            if form.total_intersections() <= max_intersections:
                yield form
