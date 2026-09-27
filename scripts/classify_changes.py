#!/usr/bin/env python3
"""Decide whether a change set needs the expensive CI matrix.

This repository runs `cargo build --all-targets` and `cargo test --workspace` on
three platforms for every pull request. A change that only touches documentation
cannot affect compilation or tests, yet it pays the full ~25 minutes -- the wall
clock is set by the slowest platform, so three platforms run in parallel and the
cost is not additive but it is still large.

Output contract (consumed by the `changes` job in `.github/workflows/ci.yml`):

    stdout: exactly one line, `code=true` or `code=false`
    stderr: the human-readable reason (ends up in the job log)

`code=false` means "documentation only": the caller may skip the test matrix.
It is emitted **only** when every changed path matches DOCS_PATTERNS. Everything
else -- source, data, scripts, workflows -- yields `code=true`, and so does an
empty range or a range that cannot be resolved. The default is deliberately the
expensive one: a wrong `code=false` lets a regression through a green pull
request, while a wrong `code=true` only costs time.

Why the caller must gate *jobs* and never the `pull_request` trigger: filtering
the workflow itself with `paths:` / `paths-ignore:` means the required checks are
never created at all, so the pull request sits at "Expected -- Waiting for
status" forever. A job that a job-level `if` skips still reports its status as
success, which is what branch protection accepts.

Usage:
    python3 scripts/classify_changes.py <rev-range>   # e.g. origin/dev..HEAD
    python3 scripts/classify_changes.py --list        # print the allowlist

Exit codes:
    0 - classification produced on stdout
    1 - usage error
"""

import argparse
import os
import re
import subprocess
import sys

# A changed path counts as documentation only if it matches at least one pattern.
# Everything else is code. Keep this list narrow: adding `fixtures/` or
# `scripts/` here would let a real behavioural change skip the test matrix.
DOCS_PATTERNS = (
    r"^docs/",  # the docs/ tree, whatever the file type (specs, reports, images)
    r"\.md$",   # markdown anywhere: README, CONTRIBUTING, AGENTS.md, crate docs
)

# A push event with no previous head reports this as the "before" revision.
ZERO_SHA = "0" * 40


def classify_paths(paths: list[str]) -> tuple[bool, str]:
    """(needs_code_ci, reason) for an explicit list of changed paths."""
    if not paths:
        return True, "no changed paths in range"

    code_paths = [
        path
        for path in paths
        if not any(re.search(pattern, path) for pattern in DOCS_PATTERNS)
    ]
    if not code_paths:
        return False, f"all {len(paths)} changed path(s) are documentation"

    preview = ", ".join(code_paths[:5])
    if len(code_paths) > 5:
        preview += ", ..."
    return True, (
        f"{len(code_paths)}/{len(paths)} changed path(s) are not documentation: {preview}"
    )


def changed_paths(range_spec: str) -> list[str]:
    """Paths changed in `range_spec`, via `git diff --name-only`."""
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.quotepath=false",
            "diff",
            # --no-renames keeps a rename visible as delete+add, so both sides are
            # classified. Rename detection would hide the removed path.
            "--no-renames",
            "--name-only",
            range_spec,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def emit(needs_code_ci: bool, reason: str) -> int:
    print(f"code={'true' if needs_code_ci else 'false'}")
    print(f"Classify: {reason}", file=sys.stderr)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        verdict = (
            "code changes -> run the full test matrix"
            if needs_code_ci
            else "documentation only -> skip the test matrix"
        )
        try:
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write(f"### Change classification\n\n`{verdict}`\n\n{reason}\n")
        except OSError:
            pass  # a missing summary file must not change the classification
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "range",
        nargs="?",
        help="git revision range to classify, e.g. `origin/dev..HEAD`",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the documentation allowlist, then exit",
    )
    args = parser.parse_args()

    if args.list:
        print("documentation allowlist (docs-only requires every path to match):")
        for pattern in DOCS_PATTERNS:
            print(f"  {pattern}")
        print("anything else, an empty range, or an unresolvable range -> code=true")
        return 0

    if not args.range:
        print("usage: classify_changes.py <rev-range>", file=sys.stderr)
        return 1

    left, _, _ = args.range.partition("..")
    if left and set(left) == {"0"}:
        return emit(True, f"range {args.range} has no usable left side")

    try:
        paths = changed_paths(args.range)
    except FileNotFoundError as exc:
        return emit(True, f"git not found: {exc}")
    except subprocess.CalledProcessError as exc:
        # git prints the useful message first and usage text after it.
        detail = (exc.stderr or "").strip().splitlines()
        return emit(
            True,
            f"cannot read range {args.range!r}: {detail[0] if detail else exc}",
        )

    return emit(*classify_paths(paths))


if __name__ == "__main__":
    sys.exit(main())
