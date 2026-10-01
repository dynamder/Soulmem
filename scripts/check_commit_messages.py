#!/usr/bin/env python3
"""Fail on commit subjects that do not follow Conventional Commits.

**Rule** (authoritative text: `docs/dev-convention.md` §2): every non-merge
commit subject in the range under review must look like one of

    <type>(<scope>): <description>
    <type>: <description>
    <type>(<scope>)!: <description>

where `<type>` is one of ALLOWED_TYPES in lower case, the separator is exactly
`: ` (colon + one space), and `<description>` is non-empty.

Why a gate and not just a documentation line: the convention existed in
CONTRIBUTING.md only as "建议遵循" -- no type whitelist, no enforcement. The
result is measurable: 111 of the last 300 commits on `dev` carry no `type`
prefix at all, alongside typos like `fear(retrieve_algo)` and free-form subjects
like `wip`. A written rule without a gate decays into a suggestion.

Scope is deliberately narrow -- this is the "cheap check" half of the split in
`docs/dev-convention.md` §7. The check reads **the subject line only** and
validates **only the four hard rules above**. It does not look at the body, the
footer, subject length, scope spelling, or the language of the description.
Those are review concerns; gating them would make contributors push repeatedly
to satisfy a linter, which is the exact cost this repository's gates avoid.

Two classes of subject are skipped because the author does not control them:
merge commits (`Merge ...`) and GitHub-generated reverts (`Revert "..."`).

Only **new** commits are checked, never history: see "只对新提交生效，不追溯
历史". Callers pass a range; the CI job passes the pull request's range.

Usage:
    python3 scripts/check_commit_messages.py origin/dev..HEAD   # a range
    python3 scripts/check_commit_messages.py                    # default: HEAD only
    python3 scripts/check_commit_messages.py --list             # show the rule, then exit

Exit codes:
    0 - clean
    1 - findings reported
    2 - usage or IO error
"""

import argparse
import re
import subprocess
import sys

# Keep in sync with docs/dev-convention.md §2.
ALLOWED_TYPES = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)

# `docs(规范): 描述` / `fix: 描述` / `feat(api)!: 描述`
SUBJECT_RE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^()]*)\))?(?P<breaking>!)?: (?P<description>.+)$"
)

SKIPPED_PREFIXES = ("Merge ", 'Revert "')

# A push event with no previous head reports this as the "before" revision.
ZERO_SHA = "0" * 40


def _log_args(range_spec: str | None) -> list[str]:
    """Translate a user-supplied range into `git log` arguments.

    A bare revision means "that single commit", not "everything reachable from
    it" -- the default (`HEAD`) must not walk the whole history.
    """
    if not range_spec:
        return ["-1", "HEAD"]
    if ".." in range_spec:
        left, _, right = range_spec.partition("..")
        left, right = left.strip(), right.strip() or "HEAD"
        if not left or set(left) == {"0"}:
            return ["-1", right]
        return [f"{left}..{right}"]
    return ["-1", range_spec]


def subjects(range_spec: str | None) -> list[tuple[str, str]]:
    """(abbreviated hash, subject) for every non-merge commit in the range."""
    cmd = [
        "git",
        "-c",
        "core.quotepath=false",
        "log",
        "--no-merges",
        "--format=%H%x1f%s",
        *_log_args(range_spec),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: git not found: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: cannot read commits: {exc.stderr.strip()}", file=sys.stderr)
        print(
            "Hint: pass a range whose left side exists locally, e.g. `origin/dev..HEAD`.\n"
            "      In CI, the checkout must fetch enough history (fetch-depth: 0).",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    commits = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        sha, _, subject = line.partition("\x1f")
        commits.append((sha[:7], subject))
    return commits


def violations(commits: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """(hash, subject, reason) for each subject that breaks a hard rule."""
    findings = []
    allowed = ", ".join(ALLOWED_TYPES)
    for sha, subject in commits:
        if subject.startswith(SKIPPED_PREFIXES):
            continue
        match = SUBJECT_RE.match(subject)
        if not match:
            findings.append(
                (
                    sha,
                    subject,
                    "does not match `<type>(<scope>): <description>` "
                    "(the separator must be `: ` -- colon + exactly one space)",
                )
            )
            continue
        type_ = match.group("type")
        scope = match.group("scope")
        if type_ not in ALLOWED_TYPES:
            findings.append((sha, subject, f"type {type_!r} is not allowed; allowed: {allowed}"))
        elif scope is not None and not scope.strip():
            findings.append((sha, subject, "scope is empty; drop the parentheses or name a scope"))
        elif not match.group("description").strip():
            findings.append((sha, subject, "description is empty"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "range",
        nargs="?",
        help="git revision range to check, e.g. `origin/dev..HEAD` (default: HEAD only)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the enforced rule, then exit",
    )
    args = parser.parse_args()

    if args.list:
        print("rule: <type>(<scope>): <description>, subject line only")
        print(f"allowed types: {', '.join(ALLOWED_TYPES)}")
        skipped = " | ".join(repr(prefix) for prefix in SKIPPED_PREFIXES)
        print(f"skipped subjects: {skipped}")
        print("not checked (deliberately): body, footer, length, scope spelling, language")
        return 0

    commits = subjects(args.range)
    if not commits:
        print("Commit message check PASS (no commits in range)")
        return 0

    findings = violations(commits)
    if not findings:
        print(f"Commit message check PASS ({len(commits)} commit(s))")
        return 0

    print(
        f"Commit message check FAILED: {len(findings)} of {len(commits)} commit(s) "
        "do not follow Conventional Commits.",
        file=sys.stderr,
    )
    for sha, subject, reason in findings:
        print(f"  {sha}  {subject!r}", file=sys.stderr)
        print(f"          -> {reason}", file=sys.stderr)
    print(
        "\nFix: rewrite the subject as `<type>(<scope>): <description>`, keeping the type\n"
        "inside the whitelist. Amend the tip commit (`git commit --amend`) or reword\n"
        "with `git rebase -i`. Only new commits are checked -- do not rewrite history\n"
        "that is already on `dev` or `main`.\n"
        f"Rule and rationale: docs/dev-convention.md §2. Run `--list` for the short form.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
