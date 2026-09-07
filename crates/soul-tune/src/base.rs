use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use crate::engine::suite::SuiteReport;

/// 检索算法测试的"管线"维度：决定具体跑哪条检索算法。
///
/// 与记忆来源正交（见 [`RetrieveMode`] 的 `*Db` 变体）：同一条管线既可以
/// 直接跑在"全量载入工作记忆"上，也可以跑在"mem 数据库召回子集"上。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetrieveFlavor {
    /// 纯相似度（Embedding 检索）
    Embedding,
    /// 相似度 + PPR 联想
    Association,
    /// DefaultPipeline（短期记忆 + 相似度 + 联想 + 动作），即生产默认检索
    FullPipeline,
}

impl std::fmt::Display for RetrieveFlavor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetrieveFlavor::Embedding => write!(f, "embedding"),
            RetrieveFlavor::Association => write!(f, "association"),
            RetrieveFlavor::FullPipeline => write!(f, "full"),
        }
    }
}

/// 检索测试运行模式 = 管线（[`RetrieveFlavor`]）× 记忆来源（直接全量 vs mem 数据库）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetrieveMode {
    /// 直接：example_data 全量载入工作记忆，跑纯相似度检索
    Embedding,
    /// 直接：全量工作记忆上跑相似度 + PPR
    Association,
    /// 直接：全量工作记忆上跑 DefaultPipeline
    FullPipeline,
    /// 数据库：example_data 先写入 mem 数据库，经 DB 召回后跑纯相似度检索
    EmbeddingDb,
    /// 数据库：DB 召回子图上跑相似度 + PPR
    AssociationDb,
    /// 数据库：DB 召回子图上跑 DefaultPipeline
    FullPipelineDb,
}

impl RetrieveMode {
    /// 该模式是否以 mem 数据库为记忆来源（查询时 DB 召回，而非全量载入工作记忆）。
    pub fn uses_db(self) -> bool {
        matches!(
            self,
            RetrieveMode::EmbeddingDb | RetrieveMode::AssociationDb | RetrieveMode::FullPipelineDb
        )
    }

    /// 该模式下实际执行的检索管线。
    pub fn flavor(self) -> RetrieveFlavor {
        match self {
            RetrieveMode::Embedding | RetrieveMode::EmbeddingDb => RetrieveFlavor::Embedding,
            RetrieveMode::Association | RetrieveMode::AssociationDb => RetrieveFlavor::Association,
            RetrieveMode::FullPipeline | RetrieveMode::FullPipelineDb => {
                RetrieveFlavor::FullPipeline
            }
        }
    }

    /// 同管线的数据库模式（直接模式返回 None）。
    pub fn db_mode(self) -> Option<RetrieveMode> {
        match self {
            RetrieveMode::Embedding => Some(RetrieveMode::EmbeddingDb),
            RetrieveMode::Association => Some(RetrieveMode::AssociationDb),
            RetrieveMode::FullPipeline => Some(RetrieveMode::FullPipelineDb),
            db @ (RetrieveMode::EmbeddingDb
            | RetrieveMode::AssociationDb
            | RetrieveMode::FullPipelineDb) => Some(db),
        }
    }
}

impl std::fmt::Display for RetrieveMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetrieveMode::Embedding => write!(f, "embedding"),
            RetrieveMode::Association => write!(f, "association"),
            RetrieveMode::FullPipeline => write!(f, "full"),
            RetrieveMode::EmbeddingDb => write!(f, "db/embedding"),
            RetrieveMode::AssociationDb => write!(f, "db/association"),
            RetrieveMode::FullPipelineDb => write!(f, "db/full"),
        }
    }
}

/// 遗忘算法测试模式：三阶段独立验证
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForgetMode {
    /// 阶段 1：只验证遮罩（纯算法、无 LLM、确定性）
    Mask,
    /// 阶段 2：只验证遮罩补全（直接驱动 llama-server，贴 LLM 原始回复）
    Revise,
    /// 阶段 3：全管线（衰减 → 遮罩 → LLM 补全 → 边衰减）
    Pipeline,
}

impl std::fmt::Display for ForgetMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForgetMode::Mask => write!(f, "mask"),
            ForgetMode::Revise => write!(f, "revise"),
            ForgetMode::Pipeline => write!(f, "full"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AlgoType {
    Retrieve(RetrieveMode),
    Compare,
    /// 同管线「直接（全量工作记忆）vs 数据库召回」逐用例对比
    CompareDb(RetrieveFlavor),
    PlayTest,
    Consolidate,
    Forget(ForgetMode),
}

impl std::fmt::Display for AlgoType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgoType::Retrieve(mode) => write!(f, "retrieve/{}", mode),
            AlgoType::Compare => write!(f, "compare"),
            AlgoType::CompareDb(RetrieveFlavor::FullPipeline) => write!(f, "compare/db"),
            AlgoType::CompareDb(flavor) => write!(f, "compare/db/{}", flavor),
            AlgoType::PlayTest => write!(f, "playtest"),
            AlgoType::Consolidate => write!(f, "consolidate"),
            AlgoType::Forget(mode) => write!(f, "forget/{}", mode),
        }
    }
}

#[derive(Clone)]
pub struct TestConfig {
    pub algo: AlgoType,
    pub dataset_path: PathBuf,
    pub params: HashMap<String, String>,
}

pub struct TestReport {
    pub config: TestConfig,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub elapsed: Duration,
    pub suite_report: SuiteReport,
    pub error: Option<String>,
}
