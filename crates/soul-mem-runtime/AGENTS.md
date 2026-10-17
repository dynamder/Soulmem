# soul-mem-runtime

Working memory + SurrealDB storage.

- Read the crate root module doc first ("must-know conventions").
- `working_memory`: sliding window, memory cluster, active records. `storage`: `MemoryRepository` + the SurrealDB implementation.

## Traps

- `upsert_notes` rebuilds out-edges wholesale; a node with only some out-edges filled silently loses the rest. Read-modify-write must `fetch_notes` first.
- The schema is all `IF NOT EXISTS` with no migration; all DB tests run in-memory, so an on-disk library never upgrades. Consider existing libraries before changing the schema.
- `cluster::memory_cluster` has a large commented-out block referencing `qdrant_client`, which isn't a dependency — uncommenting won't compile.

## Constraints

- Don't hold `parking_lot` locks across `await`.
- `WorkingMemory` has no `&mut self` API on purpose (`Arc::get_mut` silently returns `None` when shared).
- A new HNSW slot must update both `EmbeddingSlot::ALL` and the schema.
