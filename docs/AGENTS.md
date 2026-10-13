# docs

Process specs and archived reports. Design decisions and "why" live in the book (`../book/src/architecture/choices.md`), not here.

- `dev-convention.md` is the single authority for collaboration rules; `CONTRIBUTING.md` is the short entry.
- `测试数据规范.md` is the test-data authority.

## Constraints

- Adding, renaming, or deleting a file here must update `README.md` in the same commit.
- A superseded spec must be marked as such in the same commit; never keep two contradictory specs side by side.
- One-off reports must state their measurement point (or commit), dataset, environment, and that they're not current state.
- The book (`book/src/`) is manually maintained by the maintainer — agents must not modify it, unless the user explicitly let you do. Even the user let you modify the book, you should be careful and frequently ask user to align with their intent and expectation.
