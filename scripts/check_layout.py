#!/usr/bin/env python3
"""Fail on Rust module layout that violates this repository's conventions.

**Rule: no `mod.rs`.** SoulMem uses the `xxx.rs` + `xxx/` layout everywhere --
27 module directories, and 0 `mod.rs` before this check existed. `xxx.rs`'s
`mod foo;` resolves to `xxx/foo.rs`, so a submodule directory never needs a
`mod.rs`; adding one silently introduces a second, inconsistent style.

Why a gate and not just a documentation line: the convention was written down
only in the harness-global instructions, and a single module-split change then
introduced 5 `mod.rs` files. A cheap CI check beats re-litigating layout in
review.

Scope: every Rust file in the working tree -- tracked files plus untracked ones
that are not gitignored (so a freshly created `mod.rs` is caught locally before
it is ever staged), minus the vendored third-party trees listed in
EXCLUDED_PREFIXES. Anything outside those prefixes is checked even if it is new,
so a newly vendored dependency fails loudly and gets a deliberate decision rather
than being silently skipped.

Usage:
    python3 scripts/check_layout.py           # check every Rust file
    python3 scripts/check_layout.py --list    # show rules and exclusions, then exit

Exit codes:
    0 - clean
    1 - findings reported
    2 - usage or IO error
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Prefix (as reported by `git ls-files`, forward slashes) -> why it is exempt.
EXCLUDED_PREFIXES = {
    "patches/": "vendored third-party crate (cargo vendor copy); its layout is not ours to change",
}

FORBIDDEN_BASENAME = "mod.rs"


def _ls_files(*extra_args: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", *extra_args, "*.rs"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"ERROR: cannot list files: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    return [line for line in result.stdout.splitlines() if line]


def rust_files() -> list[Path]:
    """Tracked plus untracked-but-not-ignored Rust files, deduplicated and sorted."""
    names = _ls_files() + _ls_files("--others", "--exclude-standard")
    return [Path(name) for name in sorted(set(names))]


def is_excluded(path: Path) -> bool:
    posix = path.as_posix()
    return any(posix.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def suggested_path(path: Path) -> Path:
    """`a/b/mod.rs` -> `a/b.rs` (the layout this repository uses)."""
    return path.parent.with_suffix(".rs") if path.parent.name else path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the enforced rule and the exclusion list, then exit",
    )
    args = parser.parse_args()

    if args.list:
        print(f"rule: no `{FORBIDDEN_BASENAME}` in any Rust file (tracked, or untracked but not gitignored)")
        print("excluded prefixes:")
        for prefix, why in EXCLUDED_PREFIXES.items():
            print(f"  {prefix}  -- {why}")
        return 0

    files = rust_files()
    checked = [p for p in files if not is_excluded(p)]
    offenders = [p for p in checked if p.name == FORBIDDEN_BASENAME]
    skipped = len(files) - len(checked)

    if not offenders:
        detail = f"; {skipped} excluded as vendored" if skipped else ""
        print(
            f"Layout check PASS (no `{FORBIDDEN_BASENAME}` in {len(checked)} Rust files{detail})"
        )
        return 0

    print(
        f"Layout check FAILED: {len(offenders)} `{FORBIDDEN_BASENAME}` file(s) found.",
        file=sys.stderr,
    )
    for path in offenders:
        print(f"  {path}  ->  rename to {suggested_path(path)}", file=sys.stderr)
    print(
        "\nFix: move the file next to its directory and name it after the directory\n"
        "(`xxx/mod.rs` -> `xxx.rs`). Submodule resolution keeps working unchanged:\n"
        "`mod foo;` inside `xxx.rs` already resolves to `xxx/foo.rs`, so no code edits\n"
        "are needed beyond the rename.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
