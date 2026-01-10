"""
Database for storing discovered kernel elements, organized by prime.

The database is a JSON file that stores kernel elements as lists of simple braid indices.
It automatically deduplicates entries and tracks metadata like discovery time.
"""

import json
import os
from datetime import datetime
from typing import Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_db.json")


def _load_db(db_path: str) -> dict:
    """Load the database from disk, or return empty structure if not exists."""
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    return {"primes": {}, "metadata": {"created": datetime.now().isoformat()}}


def _save_db(db: dict, db_path: str):
    """Save the database to disk."""
    db["metadata"]["last_updated"] = datetime.now().isoformat()
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)


def _word_to_key(word: list[int]) -> str:
    """Convert a word list to a string key for deduplication."""
    return ",".join(str(x) for x in word)


def _key_to_word(key: str) -> list[int]:
    """Convert a string key back to a word list."""
    return [int(x) for x in key.split(",")]


def add_kernel_elements(prime: int, words: list[list[int]], db_path: str = DEFAULT_DB_PATH) -> tuple[int, int]:
    """
    Add kernel elements for a given prime to the database.
    
    Args:
        prime: The prime p for which these are kernel elements
        words: List of braid words (each word is a list of simple indices)
        db_path: Path to the database file
    
    Returns:
        Tuple of (num_new, num_total) - how many were new, and total count for this prime
    """
    db = _load_db(db_path)
    
    prime_key = str(prime)
    if prime_key not in db["primes"]:
        db["primes"][prime_key] = {
            "elements": {},
            "first_discovered": datetime.now().isoformat()
        }
    
    prime_data = db["primes"][prime_key]
    existing = prime_data["elements"]
    
    num_new = 0
    for word in words:
        key = _word_to_key(word)
        if key not in existing:
            existing[key] = {
                "word": word,
                "length": len(word),
                "discovered": datetime.now().isoformat()
            }
            num_new += 1
    
    if num_new > 0:
        prime_data["last_updated"] = datetime.now().isoformat()
        _save_db(db, db_path)
    
    return num_new, len(existing)


def get_kernel_elements(prime: int, db_path: str = DEFAULT_DB_PATH) -> list[list[int]]:
    """
    Get all kernel elements for a given prime.
    
    Args:
        prime: The prime p
        db_path: Path to the database file
    
    Returns:
        List of braid words (each word is a list of simple indices)
    """
    db = _load_db(db_path)
    prime_key = str(prime)
    
    if prime_key not in db["primes"]:
        return []
    
    elements = db["primes"][prime_key]["elements"]
    return [entry["word"] for entry in elements.values()]


def get_statistics(db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Get statistics about the database.
    
    Returns:
        Dictionary with counts and metadata for each prime
    """
    db = _load_db(db_path)
    
    stats = {
        "total_elements": 0,
        "primes": {}
    }
    
    for prime_key, prime_data in db["primes"].items():
        count = len(prime_data["elements"])
        lengths = [entry["length"] for entry in prime_data["elements"].values()]
        
        stats["primes"][int(prime_key)] = {
            "count": count,
            "min_length": min(lengths) if lengths else None,
            "max_length": max(lengths) if lengths else None,
            "first_discovered": prime_data.get("first_discovered"),
            "last_updated": prime_data.get("last_updated")
        }
        stats["total_elements"] += count
    
    stats["metadata"] = db.get("metadata", {})
    return stats


def print_summary(db_path: str = DEFAULT_DB_PATH):
    """Print a summary of the database contents."""
    stats = get_statistics(db_path)
    
    print("=" * 60)
    print("KERNEL DATABASE SUMMARY")
    print("=" * 60)
    print(f"Total kernel elements: {stats['total_elements']}")
    print()
    
    if not stats["primes"]:
        print("No kernel elements found yet.")
        return
    
    for prime in sorted(stats["primes"].keys()):
        p_stats = stats["primes"][prime]
        print(f"Prime p={prime}:")
        print(f"  Count: {p_stats['count']}")
        if p_stats['count'] > 0:
            print(f"  Length range: {p_stats['min_length']} - {p_stats['max_length']}")
            print(f"  First discovered: {p_stats['first_discovered']}")
            print(f"  Last updated: {p_stats['last_updated']}")
        print()


def export_for_prime(prime: int, output_path: Optional[str] = None, db_path: str = DEFAULT_DB_PATH):
    """
    Export kernel elements for a specific prime to a separate file.
    
    Args:
        prime: The prime p
        output_path: Where to save (defaults to kernel_p{prime}.json)
        db_path: Path to the database file
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(db_path), f"kernel_p{prime}.json")
    
    elements = get_kernel_elements(prime, db_path)
    
    with open(output_path, 'w') as f:
        json.dump({
            "prime": prime,
            "count": len(elements),
            "elements": elements
        }, f, indent=2)
    
    print(f"Exported {len(elements)} kernel elements for p={prime} to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kernel element database utilities")
    parser.add_argument("command", choices=["summary", "list", "export"],
                        help="Command to run")
    parser.add_argument("--prime", "-p", type=int, help="Prime to filter by")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH,
                        help="Path to database file")
    parser.add_argument("--output", "-o", type=str, help="Output path for export")
    
    args = parser.parse_args()
    
    if args.command == "summary":
        print_summary(args.db)
    
    elif args.command == "list":
        if args.prime is None:
            print("Error: --prime required for list command")
        else:
            elements = get_kernel_elements(args.prime, args.db)
            print(f"Kernel elements for p={args.prime}: {len(elements)} total")
            for i, word in enumerate(elements):
                print(f"  {i+1}. {word} (length {len(word)})")
    
    elif args.command == "export":
        if args.prime is None:
            print("Error: --prime required for export command")
        else:
            export_for_prime(args.prime, args.output, args.db)
