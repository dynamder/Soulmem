# soul-mem-algo

Retrieval, forgetting, consolidation.

- Read the crate root module doc first (three subdomains + "before you change this layer").
- The pipeline `RetrDefaultPipeline` (`retrieve::complex`) runs ShortOnly → Similarity → AssociateWithAction.
- Algorithm entries that need an LLM take `&LlmEngine`; there is no retry here (retry lives in soul-mem-llm).

## Traps

- `weighted_ppr_fp` panics on `damping_factor = 1.0` (assert, not Err); `naive_ppr` accepts `[0,1]` but is reference-only.
- Scoring/merging is split across three sites with inconsistent NaN handling (`total_cmp` vs `partial_cmp().unwrap_or(Equal)`).
- The action result is one seat per ActionType, not top-k; the Skill seat is always empty.
- Mutant exclusions live only in `.cargo/mutants.toml`; `#[mutants::skip]` is unused, and `exclude_re` doesn't match struct-literal field mutations.
