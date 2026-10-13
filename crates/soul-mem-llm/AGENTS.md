# soul-mem-llm

The only LLM call layer in the workspace, and the only true leaf crate.

- Read the crate root module doc first (reading index, layering, the two retry levels).
- Callers only need `LlmEngine` (in `engine`); the top-level contract (`Task`/`Completion`/`StreamEvent`/`ChatBackend`) is in `backend`.
- OpenAI-compatible wire details live in `oai_comp`; `oai_comp::wire` is the only place provider field names may appear.
- Design decisions and "why" live in the book (`../../book/src/architecture/choices.md`).

## Invariants

- Exactly one `on_end` per semantic call, including streams dropped early.
- Never replay a stream after content has been emitted.
- Never send both `max_tokens` and `max_completion_tokens`.
- Keep prompt text and keys out of logs and observability events.
- Don't fabricate data: leave `Usage` as `None`, don't invent `reasoning`, use `Unknown` for an uncertain `stop`.

## Traps

- The request is actually sent by async-openai's `ReqwestService::call`, not in this crate — grep won't find it.
- `transport` has no tower timeout layer on purpose; the backend wraps body reads in `tokio::time::timeout`.
