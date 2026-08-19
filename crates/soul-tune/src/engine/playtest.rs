pub mod runner;
pub mod trace;
pub mod repair;

// 仅 re-export 被 crate 内部实际引用的类型；其余经模块全路径访问（runner:: / trace::）。
pub use runner::{PlayTestRunner, PlayTurnResult, DialogueFile};
