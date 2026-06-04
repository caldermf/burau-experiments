#!/usr/bin/env python3
"""Compatibility entrypoint for the standalone mod-7 GPU search."""

try:
    from . import search
except ImportError:
    import search


def main():
    search.main()


if __name__ == "__main__":
    main()
