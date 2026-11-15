# soul-mem-llm

The only LLM call layer in the workspace, and the only true leaf crate.

## Where things live

| You want | Go to |
|---|---|
| semantic types | `backend` |
| orchestration, whole-call retry, observability | `engine` |
| wire encode/decode and error mapping | `oai_comp` |
| trace | `observer` |
| lenient JSON extraction | `json` |
| where the request is actually sent | not this crate: `async_openai::middleware::ReqwestService::call` |

- Callers only need `LlmEngine` (in `engine`); the top-level contract (`Task`/`Completion`/`StreamEvent`/`ChatBackend`) is in `backend`.
- OpenAI-compatible wire details live in `oai_comp`; `oai_comp::wire` is the only place provider field names may appear.
- Provider-specific extensions go through `OaiCompatConfig::extra_body`; never add them to the semantic types.
- Adding a backend usually doesn't touch `backend.rs`; the checklist and the full call chain are in `../../docs/architecture/llm-layer.md`.
- Design decisions and "why" live in the book (`../../book/src/architecture/choices.md`).

## Retry

Two levels; the maximum number of requests is `(1 + inner) × (1 + outer)`.

| Level | Covers | Where |
|---|---|---|
| transport (inner) | 429 / 5xx / connect failures | `oai_comp::transport` |
| whole call (outer) | overall and first-byte timeouts, body read failures, stream died before any output | `LlmEngine` |

- `transport` is the only place that can read `Retry-After`.
- `transport` uses `SimpleRetryPolicy`, not `OpenAIRetryLayer`, so that retries reach the trace; the cost is that a 429 quota exhaustion is retried like rate limiting.
- No tower timeout layer on purpose: the backend wraps body reads in `tokio::time::timeout`, because the transport never sees the body phase.

## Invariants

- Exactly one `on_end` per semantic call, including streams dropped early.
- Never replay a stream after content has been emitted.
- Never send both `max_tokens` and `max_completion_tokens`.
- Keep prompt text and keys out of logs and observability events.
- Don't fabricate data: leave `Usage` as `None`, don't invent `reasoning`, use `Unknown` for an uncertain `stop`.
