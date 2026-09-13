#!/usr/bin/env python3
"""Gate cargo-mutants results on a minimum viable kill rate.

Kill rate = caught / (caught + missed + timeout). Unviable mutants are
excluded because they cannot be killed by any test.

Exit codes:
  0 - kill rate meets the threshold, or no viable mutants were generated
  1 - kill rate is below the threshold
  2 - mutants output is missing or malformed
"""

import argparse
import json
import sys
from pathlib import Path


def load_summary(out_dir: Path):
    outcomes = out_dir / "outcomes.json"
    mutants = out_dir / "mutants.json"

    if outcomes.exists():
        data = json.loads(outcomes.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
        raise ValueError("outcomes.json has an unexpected shape")

    if mutants.exists():
        data = json.loads(mutants.read_text(encoding="utf-8"))
        if data == []:
            return None  # no mutants generated (e.g. doc-only PR)
        raise ValueError("mutants.json exists but outcomes.json is missing")

    raise FileNotFoundError("neither outcomes.json nor mutants.json was found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("mutants.out"))
    parser.add_argument("--threshold", type=float, default=90.0)
    args = parser.parse_args()

    try:
        summary = load_summary(args.out_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read mutants results: {exc}", file=sys.stderr)
        return 2

    if summary is None:
        print("No mutants generated; nothing to gate. PASS")
        return 0

    total = int(summary.get("total_mutants", 0))
    caught = int(summary.get("caught", 0))
    missed = int(summary.get("missed", 0))
    timeout = int(summary.get("timeout", 0))
    unviable = int(summary.get("unviable", 0))

    viable = caught + missed + timeout
    if viable == 0:
        print(f"No viable mutants (total={total}, unviable={unviable}). PASS")
        return 0

    rate = caught / viable * 100.0
    print(
        f"Mutants: total={total} caught={caught} missed={missed} "
        f"timeout={timeout} unviable={unviable} viable={viable} "
        f"kill_rate={rate:.1f}%"
    )
    if rate + 1e-9 >= args.threshold:
        print(f"Kill rate {rate:.1f}% >= {args.threshold:g}%. PASS")
        return 0

    print(f"Kill rate {rate:.1f}% < {args.threshold:g}%. FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
