# soul-mem-runtime

Working memory + SurrealDB storage.

- The crate root module doc lists the subdomains; the notes below are the ones that bite.
- `working_memory`: sliding window, memory cluster, active records. `storage`: `MemoryRepository` + the SurrealDB implementation.

## Traps

- Sliding-window eviction is two steps: `push` only enqueues and reports overflow, `pop` only removes the head, and `summarize` is a separate call by the caller; a failing summary does not drop messages.
- `upsert_notes` rebuilds out-edges wholesale; a node with only some out-edges filled silently loses the rest. Read-modify-write must `fetch_notes` first.
- The schema is all `IF NOT EXISTS` with no migration; all DB tests run in-memory, so an on-disk library never upgrades. Consider existing libraries before changing the schema.
- `cluster::memory_cluster` has a large commented-out block referencing `qdrant_client`, which isn't a dependency — uncommenting won't compile.

## Constraints

- Don't hold `parking_lot` locks across `await`; the async lock used for summary updates is the only exception.
- `WorkingMemory` has no `&mut self` API on purpose (`Arc::get_mut` silently returns `None` when shared).
- The per-field locks don't cover cross-field invariants: `add_node` writes the cluster and then `records` without holding both locks.
- No `MERGE` on purpose: deep merging leaves stale keys from old variants of the external tag enum.
- A new HNSW slot must update both `EmbeddingSlot::ALL` and the schema.
