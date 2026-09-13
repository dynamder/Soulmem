# SoulMem

[简体中文](README.md) | [English](README_en.md)

[![Project Status: WIP](https://img.shields.io/badge/Status-Active%20Development-orange)](https://github.com/dynamder/Soulmem)

[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

[![CI](https://github.com/dynamder/Soulmem/actions/workflows/ci.yml/badge.svg)](https://github.com/dynamder/Soulmem/actions/workflows/ci.yml)

[![Mutants](https://github.com/dynamder/Soulmem/actions/workflows/mutants.yml/badge.svg)](https://github.com/dynamder/Soulmem/actions/workflows/mutants.yml)

SoulMem is a memory system specifically designed for role-playing tasks. It **aims to** enable more anthropomorphic outputs from LLMs, allowing simulated characters to remember important, emotionally relevant, and behavior-driving events like humans, and to establish connections between them. **It does not aim to** remember event details or factual knowledge with precise accuracy.

**Please Note!**: SoulMem is a memory system intended for **individual users** running on **personal computers**, and is not an enterprise-level solution.

## ✨ Core Features

-   ***Memory Consolidation & Evolution***: Memories are consolidated and generalized over time, forming higher-level cognitions.
-   ***Active Forgetting Mechanism***: Simulates the human forgetting curve, retaining important memories while fading trivial details.
-   ***Graph-based Association***: Achieves active memory association through a working memory subgraph.
-   ***Dynamic Memory Updates***: Supports real-time addition and updating of memories during interaction.
-   ***Short-term Memory Abstraction***: Processes short-term context through summarization mechanisms to prevent information overload.

## 🏗️ Design Philosophy

The core design philosophy of SoulMem is: ***"All characteristics and events belong to memory."***

Unlike traditional role-playing systems that rely on static "character cards," SoulMem posits that a character's personality, speech habits, behavioral patterns, etc., are all results of the interactive evolution of long-term memories. This design aims to better support the dynamic evolution of character personality and maintain high character consistency.

## 📁 Project Status & Architecture

***Current Status: Active Development***

*> 🚧 SoulMem is under active development and a stable version has not yet been released. We welcome interested developers to follow, discuss, and even contribute! Please refer to the* *`docs`* *directory for the latest architectural designs and development progress.*

- ***Important Notice***: The project has undergone architectural refactoring. The `main` branch contains the latest version. The old alpha version code can be found on the [`alpha_deprecated`](https://github.com/dynamder/SoulMem/tree/alpha_deprecated) branch.

- ***Current Architecture***: See [`docs/architecture/orchestration.md`](docs/architecture/orchestration.md) — it marks every element ✅/🔲 to separate **implemented** from **planned**, and is the description kept in sync with the code.

- ***Design History***: [`docs/architecture/beta_ver.md`](docs/architecture/beta_ver.md) is a beta-stage **design proposal** (open questions and `- [ ]` TODOs included). Parts of it have been superseded by the implementation — do not read it as current state.

- ***For AI agents and contributors***: start with [`AGENTS.md`](AGENTS.md). It carries the repository map, a "where to start reading" index, the invariants that must hold, and the known traps that fail silently.

### Repository layout

| Path | Description |
|---|---|
| `crates/soul-mem-core` | Pure data model (`MemoryNote` / `MemoryLink`), no internal dependencies |
| `crates/soul-mem-query` | Text→vector embedding, query types, similarity and scoring |
| `crates/soul-mem-llm` | The single LLM call layer (contract / transport / retry / streaming / trace) |
| `crates/soul-mem-runtime` | Working memory (sliding window, cluster, activation records) and the SurrealDB repository |
| `crates/soul-mem-algo` | Retrieval strategies, forgetting, consolidation |
| `crates/soul-tune` | Test and benchmark framework (headless CLI + library), **not a runtime component** |
| `crates/soul-tune-api` | Flutter Rust Bridge layer |
| `soul-tune-ui/` | Flutter GUI |
| `benches/` | criterion benchmarks (cosine / SIMD / PPR) |
| `fixtures/` | Test datasets (character graphs and dialogues) |
| `docs/` | Architecture and specification documents |

## 🚀 Quick Start

No stable release yet, but the test framework is usable today:

```bash
# GUI (recommended; requires a Flutter toolchain)
cd soul-tune-ui && flutter run -d windows

# Or the headless CLI
cargo run -p soul-tune -- inspect fixtures/graphs/rust_small_zh.json
cargo run -p soul-tune -- run retrieve/full fixtures/example_data --batch
cargo run -p soul-tune -- playtest <graph_dir> <dialogue_file>
```

Full command reference, the `algo` value table and the dataset format live in [`crates/soul-tune/docs/user-guide.md`](crates/soul-tune/docs/user-guide.md).

> **Note**: some tests and all playtest runs need a downloaded embedding model (BGE)
> or a local GGUF model. Failing offline is an environment problem, not a broken
> build — see [`AGENTS.md`](AGENTS.md) §5.

## 🔧 Development & CI

- Build: `cargo build --workspace --all-targets`
- Test: `cargo test --workspace`
- Format & lint: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`
- Mutation testing: `cargo mutants --workspace` (kill rate ≥90%, gated by `scripts/mutants_gate.py`)
- Pushes and PRs trigger build + tests on Windows, Ubuntu, and macOS; PRs also run diff-scoped mutants; a full mutants run happens every other week.

## 🤝 Contributing

We warmly welcome contributions of any kind! Whether it's code, documentation, ideas, or testing, all help SoulMem grow.

1.  Fork the repository
2.  Create your feature branch
3.  Commit your changes
4.  Push to the branch
5.  Open a Pull Request

Please ensure your code follows the existing project style.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the detailed contribution guide.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Thanks to all contributors who have provided ideas and assistance for this project.
