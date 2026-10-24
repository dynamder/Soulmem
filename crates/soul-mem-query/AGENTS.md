# soul-mem-query

Embedding + scoring: turns memory and queries into vectors and computes similarity.

- The crate root module doc lists the two subdomains; the notes below are the ones that bite.
- Two subdomains: `embedding` (domain objects → vectors) and `query` (query types + scoring).
- The only scoring entry point is `query::compute::compute_fused`; `MemoryEmbedding` has no distance methods, so don't build a second same-named API.
- Many tests call `BgeSmallZh::default_cpu()` and download a model; an offline failure is an environment problem, not a code failure.

## Traps

- `time_span` is accepted and stored but scoring was never implemented; the embedding side has no such field. No test asserts time filtering works.
- The score contract claims [0,1], but raw cosine can be negative; the Situation path folds negatives to 0, so "negatively correlated" is indistinguishable from "no signal".
- `into_parts`/`mean_pooling` silently reset `blend_weights` to default.
- `string_blend_alpha` is stored on the query but `compute_fused` takes it as a parameter; only the algo-side caller wires it up.
- `is_zero()` is true for an empty vector; `normalize()` returns NaN (not Err) on a zero vector.
- Weight comments in `query::compute` are stale — the real defaults are in `embedding::blend_weights`.

## Constraints

- Query/passage asymmetry: the `infer_query_*` defaults on `EmbeddingModel` equal the non-query versions. Models needing an instruction prefix must override them; forgetting doesn't error, it just degrades retrieval.
- Empty tag → zero vector is a cross-crate contract; runtime skips writing such columns.
- `time_span` intent is undecided — don't implement scoring or delete the field without asking the maintainer.
