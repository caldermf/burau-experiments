from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Dict, Iterable, Iterator, Mapping, Tuple


class CoefficientRing:
    name: str

    def normalize(self, value: int) -> int:
        raise NotImplementedError

    def add(self, left: int, right: int) -> int:
        return self.normalize(left + right)

    def sub(self, left: int, right: int) -> int:
        return self.normalize(left - right)

    def mul(self, left: int, right: int) -> int:
        return self.normalize(left * right)

    def neg(self, value: int) -> int:
        return self.normalize(-value)

    def is_zero(self, value: int) -> bool:
        return self.normalize(value) == 0

    def format(self, value: int) -> str:
        return str(self.normalize(value))

    @property
    def zero(self) -> int:
        return self.normalize(0)

    @property
    def one(self) -> int:
        return self.normalize(1)


class IntegerRing(CoefficientRing):
    name = "ZZ"

    def normalize(self, value: int) -> int:
        return int(value)


@dataclass(frozen=True)
class FiniteFieldPrime(CoefficientRing):
    modulus: int

    def __post_init__(self) -> None:
        if self.modulus <= 1:
            raise ValueError("modulus must be prime and > 1")

    @property
    def name(self) -> str:
        return f"F_{self.modulus}"

    def normalize(self, value: int) -> int:
        return int(value) % self.modulus

    def format(self, value: int) -> str:
        return str(self.normalize(value))


ZZ = IntegerRing()


@dataclass(frozen=True)
class LaurentPoly:
    ring: CoefficientRing
    terms: Tuple[Tuple[int, int], ...]

    def __init__(self, ring: CoefficientRing, terms: Mapping[int, int] | None = None) -> None:
        normalized: Dict[int, int] = {}
        if terms:
            for exponent, coefficient in terms.items():
                coeff = ring.normalize(coefficient)
                if not ring.is_zero(coeff):
                    normalized[int(exponent)] = coeff
        object.__setattr__(self, "ring", ring)
        object.__setattr__(self, "terms", tuple(sorted(normalized.items())))

    @classmethod
    def _from_terms(cls, ring: CoefficientRing, terms: Tuple[Tuple[int, int], ...]) -> "LaurentPoly":
        obj = object.__new__(cls)
        object.__setattr__(obj, "ring", ring)
        object.__setattr__(obj, "terms", terms)
        return obj

    @classmethod
    def zero(cls, ring: CoefficientRing) -> "LaurentPoly":
        return _cached_zero(ring)

    @classmethod
    def one(cls, ring: CoefficientRing) -> "LaurentPoly":
        return _cached_constant(ring, ring.one)

    @classmethod
    def constant(cls, ring: CoefficientRing, value: int) -> "LaurentPoly":
        return _cached_constant(ring, ring.normalize(value))

    @classmethod
    def q(cls, ring: CoefficientRing, power: int = 1) -> "LaurentPoly":
        return _cached_monomial(ring, power, ring.one)

    def is_zero(self) -> bool:
        return not self.terms

    def to_dict(self) -> Dict[int, int]:
        return dict(self.terms)

    def coefficient(self, exponent: int) -> int:
        return self.to_dict().get(exponent, self.ring.zero)

    def shift(self, amount: int) -> "LaurentPoly":
        if amount == 0 or self.is_zero():
            return self
        return LaurentPoly._from_terms(self.ring, tuple((exp + amount, coeff) for exp, coeff in self.terms))

    def monomial_mul(self, coefficient: int = 1, shift: int = 0) -> "LaurentPoly":
        coeff = self.ring.normalize(coefficient)
        if self.is_zero() or self.ring.is_zero(coeff):
            return LaurentPoly.zero(self.ring)
        if coeff == self.ring.one and shift == 0:
            return self
        return LaurentPoly._from_terms(
            self.ring,
            tuple((exp + shift, self.ring.mul(poly_coeff, coeff)) for exp, poly_coeff in self.terms),
        )

    def convert(self, ring: CoefficientRing) -> "LaurentPoly":
        return LaurentPoly(ring, {exp: coeff for exp, coeff in self.terms})

    def evaluate(self, q_value: int) -> int | Fraction:
        if q_value == 0 and any(exp < 0 for exp, _ in self.terms):
            raise ValueError("cannot evaluate a Laurent polynomial with negative powers at q=0")
        if self.ring == ZZ:
            total = Fraction(0, 1)
            for exponent, coefficient in self.terms:
                if exponent >= 0:
                    power = Fraction(q_value**exponent, 1)
                else:
                    power = Fraction(1, q_value ** (-exponent))
                total += Fraction(coefficient, 1) * power
            return total
        if isinstance(self.ring, FiniteFieldPrime):
            total = self.ring.zero
            inverse = None
            for exponent, coefficient in self.terms:
                if exponent >= 0:
                    power = pow(q_value, exponent, self.ring.modulus)
                else:
                    if inverse is None:
                        inverse = pow(q_value, -1, self.ring.modulus)
                    power = pow(inverse, -exponent, self.ring.modulus)
                total = self.ring.add(total, self.ring.mul(coefficient, power))
            return total
        raise TypeError(f"unsupported ring for evaluation: {self.ring}")

    def __bool__(self) -> bool:
        return not self.is_zero()

    def __neg__(self) -> "LaurentPoly":
        return self.monomial_mul(-1, 0)

    def __add__(self, other: "LaurentPoly") -> "LaurentPoly":
        self._check_ring(other)
        left = self.terms
        right = other.terms
        left_index = 0
        right_index = 0
        result: list[tuple[int, int]] = []
        while left_index < len(left) and right_index < len(right):
            left_exponent, left_coefficient = left[left_index]
            right_exponent, right_coefficient = right[right_index]
            if left_exponent == right_exponent:
                coefficient = self.ring.add(left_coefficient, right_coefficient)
                if not self.ring.is_zero(coefficient):
                    result.append((left_exponent, coefficient))
                left_index += 1
                right_index += 1
            elif left_exponent < right_exponent:
                result.append((left_exponent, left_coefficient))
                left_index += 1
            else:
                result.append((right_exponent, right_coefficient))
                right_index += 1
        if left_index < len(left):
            result.extend(left[left_index:])
        if right_index < len(right):
            result.extend(right[right_index:])
        return LaurentPoly._from_terms(self.ring, tuple(result))

    def __sub__(self, other: "LaurentPoly") -> "LaurentPoly":
        return self + (-other)

    def __mul__(self, other: "LaurentPoly" | int) -> "LaurentPoly":
        if isinstance(other, int):
            return self.monomial_mul(other, 0)
        self._check_ring(other)
        if self.is_zero() or other.is_zero():
            return LaurentPoly.zero(self.ring)
        result: Dict[int, int] = {}
        for left_exp, left_coeff in self.terms:
            for right_exp, right_coeff in other.terms:
                exponent = left_exp + right_exp
                product = self.ring.mul(left_coeff, right_coeff)
                result[exponent] = self.ring.add(result.get(exponent, self.ring.zero), product)
                if self.ring.is_zero(result[exponent]):
                    del result[exponent]
        return LaurentPoly(self.ring, result)

    def __rmul__(self, other: int) -> "LaurentPoly":
        return self * other

    def _check_ring(self, other: "LaurentPoly") -> None:
        if self.ring != other.ring:
            raise TypeError(f"ring mismatch: {self.ring} != {other.ring}")

    def __str__(self) -> str:
        if self.is_zero():
            return "0"
        parts: list[str] = []
        for exponent, coefficient in sorted(self.terms, reverse=True):
            sign = "-" if coefficient < 0 else "+"
            abs_coeff = abs(coefficient)
            if self.ring != ZZ:
                sign = "+"
                abs_coeff = coefficient
            if exponent == 0:
                monomial = self.ring.format(abs_coeff)
            elif exponent == 1:
                monomial = "q" if abs_coeff == 1 else f"{self.ring.format(abs_coeff)}*q"
            else:
                monomial = f"q^{exponent}" if abs_coeff == 1 else f"{self.ring.format(abs_coeff)}*q^{exponent}"
            if not parts:
                if sign == "-":
                    parts.append(f"-{monomial}")
                else:
                    parts.append(monomial)
            else:
                parts.append(f" {sign} {monomial}")
        return "".join(parts)


@dataclass(frozen=True)
class Matrix:
    rows: Tuple[Tuple[LaurentPoly, ...], ...]

    def __init__(self, rows: Iterable[Iterable[LaurentPoly]]) -> None:
        materialized = tuple(tuple(row) for row in rows)
        if not materialized:
            raise ValueError("matrix must have at least one row")
        width = len(materialized[0])
        if width == 0:
            raise ValueError("matrix must have at least one column")
        if any(len(row) != width for row in materialized):
            raise ValueError("ragged matrix")
        object.__setattr__(self, "rows", materialized)

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def ring(self) -> CoefficientRing:
        return self.rows[0][0].ring

    @classmethod
    def zero(cls, ring: CoefficientRing, size: int) -> "Matrix":
        zero = LaurentPoly.zero(ring)
        return cls([[zero for _ in range(size)] for _ in range(size)])

    @classmethod
    def identity(cls, ring: CoefficientRing, size: int) -> "Matrix":
        zero = LaurentPoly.zero(ring)
        one = LaurentPoly.one(ring)
        rows = []
        for row in range(size):
            entries = []
            for col in range(size):
                entries.append(one if row == col else zero)
            rows.append(entries)
        return cls(rows)

    def convert(self, ring: CoefficientRing) -> "Matrix":
        return Matrix([[entry.convert(ring) for entry in row] for row in self.rows])

    def __mul__(self, other: "Matrix") -> "Matrix":
        if self.width != other.height:
            raise ValueError("shape mismatch")
        zero = LaurentPoly.zero(self.ring)
        rows = []
        for row_index in range(self.height):
            current_row = []
            for col_index in range(other.width):
                value = zero
                for inner in range(self.width):
                    value = value + (self.rows[row_index][inner] * other.rows[inner][col_index])
                current_row.append(value)
            rows.append(current_row)
        return Matrix(rows)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False
        return self.rows == other.rows

    def __str__(self) -> str:
        return "[" + ", ".join("[" + ", ".join(str(entry) for entry in row) + "]" for row in self.rows) + "]"


@lru_cache(maxsize=None)
def _cached_zero(ring: CoefficientRing) -> LaurentPoly:
    return LaurentPoly._from_terms(ring, ())


@lru_cache(maxsize=None)
def _cached_constant(ring: CoefficientRing, value: int) -> LaurentPoly:
    coefficient = ring.normalize(value)
    if ring.is_zero(coefficient):
        return _cached_zero(ring)
    return LaurentPoly._from_terms(ring, ((0, coefficient),))


@lru_cache(maxsize=None)
def _cached_monomial(ring: CoefficientRing, exponent: int, value: int) -> LaurentPoly:
    coefficient = ring.normalize(value)
    if ring.is_zero(coefficient):
        return _cached_zero(ring)
    return LaurentPoly._from_terms(ring, ((exponent, coefficient),))
