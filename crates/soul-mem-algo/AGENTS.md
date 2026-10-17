# soul-mem-algo

Retrieval, forgetting, consolidation.

- Read the crate root module doc first (three subdomains); the notes below are the ones that bite.
- The pipeline `RetrDefaultPipeline` (`retrieve::complex`) runs ShortOnly → Similarity → AssociateWithAction.
- Algorithm entries that need an LLM take `&LlmEngine`; there is no retry here (retry lives in soul-mem-llm).
- `retrieve::prefetch_db` does DB IO and writes working memory, so this crate depends on `runtime`; the "algo layer does no IO" line does not hold on the retrieval path.

## Traps

- `weighted_ppr_fp` panics on `damping_factor = 1.0` (assert, not Err); `naive_ppr` accepts `[0,1]` but is reference-only.
- Scoring/merging is split across `default_pipeline::merge_note_scores`, `assoc_with_action::merge_situation_sources` and the sort/truncate in `association`; NaN handling differs (`total_cmp` vs `partial_cmp().unwrap_or(Equal)`), so changing "how scores merge" means touching all three.
- The action result is one seat per ActionType, not top-k; ties break on `MemoryId`, and the skill seat is always empty (`SkillRecord` is an empty placeholder).
- Mutant exclusions live only in `.cargo/mutants.toml`; `#[mutants::skip]` is unused, and `exclude_re` doesn't match struct-literal field mutations.
