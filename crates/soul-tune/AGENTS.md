# soul-tune

Tuning and playtest binary: runs the forget and retrieve suites against real graphs and emits observations.

- Suites live under `src/engine/`; `forget` has three stages: `Mask` (deterministic masking), `Revise` (LLM completion), `Pipeline` (full pipeline).
- `forget` file map: `mask.rs` / `revise.rs` / `pipeline.rs` (+ `pipeline/{multistep,activation,excitation}.rs`) / `tests.rs`. The pipeline suite pairs a control and an experimental graph to check that excitation delays forgetting, and reads no algorithm constants.
- LLM access goes through `engine::llm::resolver`; when no engine is available the suite degrades to masking only.
- `engine/llm/llama_server.rs` owns process lifecycle (probe, spawn, shutdown, startup timeout); HTTP calls go through `soul-mem-llm`. Wire shape lives in `soul-mem-llm/src/oai_comp/wire.rs`, retries and timeouts in `transport.rs` + `engine.rs`, and `SOULMEM_LLM_TRACE=<path>` enables tracing.

## Traps

- `LlmBackend::chat` is a synchronous shell; call it from a blocking context.
- `DEFAULT_TEMPERATURE` is fixed on purpose: existing playtest observations depend on it.
- The task-local PAW / tokio bridge runtime is global and never dropped, because dropping a runtime inside async context panics.
