# soul-mem-core

Data model: `MemoryNote` / `MemoryLink` shells plus their type-specific payloads. No IO, LLM, or algorithms.

- `MemoryNote::links()` is the complete out-edge set, not a patch: `runtime::upsert_notes` rebuilds all out-edges in one transaction.
- New code uses bare-noun accessors (`id()`, `tags()`), not `get_xxx()`.
- Public output shape is owned by `render`; never serialize the internal types directly.

## Invariants

- Keep `#[serde(default)]` on `missing_degree` / `last_forget_time`: it is what makes already-persisted data deserialize.
- `MemoryNote::missing_degree` stays private and writable only through `set_missing_degree`, which clamps to `[0,1]`.
- `MemoryLink::intensity` / `MemoryLink::missing_degree` are `pub` fields, so out-of-range values are writable there; the node side clamps, the link side does not.
- `MemoryNoteBuilder::build()` returns `Result`; `MemoryLinkBuilder::build()` returns a bare value — don't copy one pattern to the other.
