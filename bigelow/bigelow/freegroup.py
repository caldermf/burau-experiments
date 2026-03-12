from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


Word = Tuple[int, ...]


def reduce_word(word: Iterable[int]) -> Word:
    stack: list[int] = []
    for letter in word:
        if stack and stack[-1] == -letter:
            stack.pop()
        else:
            stack.append(int(letter))
    return tuple(stack)


def multiply_words(*words: Word) -> Word:
    result: list[int] = []
    for word in words:
        for letter in word:
            if result and result[-1] == -letter:
                result.pop()
            else:
                result.append(letter)
    return tuple(result)


def invert_word(word: Word) -> Word:
    return tuple(-letter for letter in reversed(word))


def generator(index: int) -> Word:
    if index <= 0:
        raise ValueError("free-group generators are 1-indexed")
    return (index,)


@dataclass(frozen=True)
class FreeGroupAutomorphism:
    images: Tuple[Word, ...]

    @classmethod
    def identity(cls, rank: int) -> "FreeGroupAutomorphism":
        return cls(tuple((index,) for index in range(1, rank + 1)))

    def apply(self, word: Word) -> Word:
        pieces: list[Word] = []
        for letter in word:
            image = self.images[abs(letter) - 1]
            if letter < 0:
                image = invert_word(image)
            pieces.append(image)
        return multiply_words(*pieces)

    def compose(self, other: "FreeGroupAutomorphism") -> "FreeGroupAutomorphism":
        if len(self.images) != len(other.images):
            raise ValueError("rank mismatch")
        return FreeGroupAutomorphism(tuple(self.apply(image) for image in other.images))

    def inverse(self) -> "FreeGroupAutomorphism":
        rank = len(self.images)
        basis = [generator(index) for index in range(1, rank + 1)]
        candidates = [basis[index] for index in range(rank)]
        for _ in range(rank):
            progress = False
            for index, image in enumerate(self.images):
                if candidates[index] != basis[index] and candidates[index] != invert_word(basis[index]):
                    continue
                for guess in basis + [invert_word(word) for word in basis]:
                    if self.apply(guess) == basis[index]:
                        candidates[index] = guess
                        progress = True
                        break
            if all(self.apply(candidates[index]) == basis[index] for index in range(rank)):
                return FreeGroupAutomorphism(tuple(candidates))
            if not progress:
                break
        raise ValueError("failed to invert free-group automorphism")

    def is_identity(self) -> bool:
        return all(image == (index + 1,) for index, image in enumerate(self.images))

    def key(self) -> Tuple[Word, ...]:
        return self.images
