# scripts

Gates and CI helpers.

- `check_encoding.py` — enforces UTF-8 source files.
- `check_layout.py` — no `mod.rs`.
- `check_commit_messages.py` — Conventional Commits subject.
- `classify_changes.py` — classifies doc-only changes to skip the Test matrix.
- `mutants_gate.py` — mutation kill rate ≥90%.

## Constraints

- No gate may exist here without being wired into CI: a new gate must be added to a workflow in the same commit, or marked "not wired" in its header.
- Exit codes: 0 = clean, 1 = findings, 2 = usage/IO error.
