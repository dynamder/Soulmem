# soul-tune-api

FRB interface crate: JSON over FRB for the Dart GUI.

- Every entry point returns a JSON string or pushes JSON strings through `StreamSink<String>`; events are the internal tagged enums `RunEvent` / `BatchEvent`.
- Cancellation is the global `CANCEL` flag, checked between cases; the Dart side calls `reset_cancel` before each run.
- Every run goes through the one spawn-and-report helper: an error must never be dropped, since the Dart side implements the `error` branch.

## Traps

- The batch entries ignore `params_json`: the engine batch path passes `None` per dataset, so batch runs always use `RetrieveSuite::load` defaults. The single-dataset entries do honor it — don't infer batch behavior from their signature.
