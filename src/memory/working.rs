///工作记忆
mod task;
mod diffuser;

use std::collections::{HashMap, HashSet};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;

use crate::memory::{GraphMemoryLink, MemoryCluster, MemoryNote, NodeRefId};
use anyhow::Result;
use chrono::DateTime;
use formatx::formatx;
use petgraph::Direction;
use petgraph::prelude::EdgeRef;
use petgraph::visit::{Dfs, VisitMap, Visitable};
use crate::llm_driver::{LLMConfig, Llm};
use crate::memory::share::NoteRecord;
use crate::memory::temporary::{MemorySource, TemporaryMemory, TemporaryNoteRecord};
use crate::memory::working::diffuser::{DiffuserConfig, MemoryDiffuser};
use crate::memory::working::task::{SoulTask, SoulTaskSet};
use super::default_prompts;

const DEFAULT_FOCUS: &str = "发呆";
///工作记忆记录，记录记忆节点被提取，激活的情况，用于记忆进化和巩固
#[derive(Debug,Clone)]
pub struct WorkingNoteRecord {
    #[allow(unused)]
    pub note_id: NodeRefId, // 节点索引

    activation_count: u32, // 激活次数
    activation_history: HashMap<NodeRefId, u32>, // 激活历史信息（共激活节点索引，次数）
    last_accessed: DateTime<chrono::Utc>,// 最后访问时间
}
#[allow(dead_code)]
impl WorkingNoteRecord {
    pub fn new(note_id: NodeRefId) -> Self {
        Self {
            note_id,
            activation_count: 1,
            activation_history: HashMap::new(),
            last_accessed: chrono::Utc::now(),
        }
    }
    pub fn last_accessed_at(&self) -> &DateTime<chrono::Utc> {
        &self.last_accessed
    }
}
impl NoteRecord for WorkingNoteRecord {
    fn activation_count(&self) -> u32 {
        self.activation_count
    }
    ///记录共激活信息，接受同时激活的节点列表
    fn record_activation(&mut self, act_ids: &[NodeRefId]) {
        act_ids.iter().for_each(|id| {
            if let Some(activations) = self.activation_history.get_mut(id) {
                *activations += 1;
            } else {
                self.activation_history.insert(id.clone(), 1);
            }
        });
        self.activation_count += 1;
        self.last_accessed = chrono::Utc::now();
    }
    fn activation_history(&self) -> &HashMap<NodeRefId, u32> {
        &self.activation_history
    }
}

#[allow(dead_code)]
#[derive(Debug,Clone)]
pub struct WorkingMemory {
    cluster: MemoryCluster, //记忆图
    working_record_map: HashMap<NodeRefId, WorkingNoteRecord>, //工作记忆激活记录映射
    temporary: TemporaryMemory, //临时记忆
    diffuser: MemoryDiffuser,
    task_set: SoulTaskSet, //任务集
}
#[allow(unused)]
impl WorkingMemory {
    pub fn new(task_inertia: f32, diffuser_config: DiffuserConfig) -> Result<WorkingMemory> {
        Ok(
            WorkingMemory {
                cluster: MemoryCluster::new(),
                temporary: TemporaryMemory::new(),
                working_record_map: HashMap::new(),
                task_set: SoulTaskSet::new(task_inertia)?,
                diffuser: MemoryDiffuser::from_config(diffuser_config),
            }
        )
    }
    /// 添加任务
    pub fn add_task(&mut self, task_summary: impl Into<String>, related_notes: &[NodeRefId] ) {
        self.task_set.add_task(
            SoulTask::new(
                task_summary.into(),
                related_notes.to_vec()
            )
        );
    }

    /// 删除一个节点
    pub fn remove_note(&mut self, node_id: &str) -> Option<MemoryNote> {
       todo!()
    }
    /// 获取一个节点记忆
    pub fn get_note(&self, node_id: &NodeRefId) -> Option<&MemoryNote> {
        self.cluster.get_node(node_id)
    }
    pub fn get_note_mut(&mut self, node_id: &NodeRefId) -> Option<&mut MemoryNote> {
        self.cluster.get_node_mut(node_id)
    }
    ///将一系列记忆合并到工作记忆图中
    pub fn merge_mem_graph(&mut self, mem: Vec<MemoryNote>) {
        self.cluster.merge(mem)
    }

    /// 添加临时记忆
    pub fn add_temp_mem(&mut self, mem: MemoryNote,source: MemorySource) {
        self.temporary.add_temp_memory(
            TemporaryNoteRecord::new(mem.id().to_owned(),source)
        );
        self.cluster.add_single_node(mem);
    }
    ///获取当前焦点的LLM上下文,并维护激活信息
    pub fn focus_context(&mut self, sliding_window_size: usize, depth: usize, rng: &mut impl rand::RngCore) -> Vec<String> {
       todo!()
    }
    ///使用LLM找到关联的任务
    pub async fn find_related_task(&self, query: impl AsRef<str>, role: impl AsRef<str>, llm_driver: &impl Llm, llm_config: &LLMConfig) -> Result<Vec<String>> {
        let response = llm_driver.get_completion(
            formatx!(
                default_prompts::FIND_RELATION_PROMPT,
                role.as_ref(),
                query.as_ref(),
                self.task_set.tasks()
                    .enumerate()
                    .fold(String::new(), |acc, (index,task)| {
                    acc + &format!("task_id: {}, task_index:{}, description: {}\n",task.id, index, task.summary)
                })
            )?,
            llm_config
        ).await?;
        if let serde_json::Value::Array(tasks) = response["related_task"].to_owned() {
            Ok(
                tasks.into_iter()
                    .filter_map(|task_id| task_id.as_str().map(|task_id| task_id.to_string()))
                    .collect()
            )
        }else {
            Err(anyhow::anyhow!("Invalid response Json from Llm"))
        }

    }
    ///焦点迭代，根据当前相关任务选择新焦点
    pub fn shift_focus(&mut self, sorted_related_task: &[impl AsRef<str>]) -> String {
        self.task_set.shift_focus(sorted_related_task)
    }

    ///根据共激活历史信息，选择需要巩固，进化的记忆
    pub fn filter_need_consolidate<Fw,Ft>(&self, filter_working: Fw, filter_temp: Ft) -> Vec<(MemoryNote,Vec<MemoryNote>)> //(记忆，可能建立的联系)
    where
        Fw: Fn(&WorkingNoteRecord) -> bool,
        Ft: Fn(&TemporaryNoteRecord) -> bool,{
        let plastic_temporary =
            self.temporary.filter_temp_memory(|record| filter_temp(record))
                .into_iter()
                .filter(|record| {
                    self.cluster.contains_node(&record.note_id) &&
                    record.activation_history()
                        .keys()
                        .all(|id| self.cluster.contains_node(id)) //TODO: clean dirty id instead
                })
                .map(|record| {
                    (
                        self.cluster.get_node(&record.note_id).unwrap().clone(), //For now, we ensured all notes are actually exists
                        record.activation_history()
                            .keys()
                            .map(|id| {
                                self.cluster.get_node(id).unwrap().clone()
                            })
                            .collect::<Vec<_>>()
                        )
                });



        let plastic_working =
            self.working_record_map.values()
                .filter(|&record| {
                    filter_working(record) &&
                    self.cluster.contains_node(&record.note_id) &&
                    record.activation_history()
                        .keys()
                        .all(|id| self.cluster.contains_node(id))
                })
                .map(|record| {
                    (
                        self.cluster.get_node(&record.note_id).unwrap().clone(),
                        record.activation_history()
                            .keys()
                            .map(|id| {
                                self.cluster.get_node(id).unwrap().clone()
                            })
                            .collect::<Vec<_>>()
                    )
                });

        plastic_temporary.chain(plastic_working).collect()
    }
}

mod test {
    use crate::memory::{MemoryLink, MemoryNoteBuilder};
    use super::*;
    fn prepare_working_memory(init: Vec<MemoryNote>) -> WorkingMemory {
        let mut mem = WorkingMemory::new(0.5,DiffuserConfig::default()).unwrap();
        mem.merge_mem_graph(init);
        mem
    }
    #[test]
    fn test_merge_mem_graph() {
        let mut mem = prepare_working_memory(
            vec![
                MemoryNoteBuilder::new("test1")
                    .id("test1")
                    .links(vec![
                        MemoryLink::new("test2", None::<String>,"test".to_string(),1.0),
                        MemoryLink::new("test3", None::<String>,"test".to_string(),2.0),
                    ])
                    .build(),
                MemoryNoteBuilder::new("test2")
                    .id("test2")
                    .links(vec![MemoryLink::new("test1", None::<String>,"test".to_string(),1.0)])
                    .build(),
                MemoryNoteBuilder::new("test3")
                    .id("test3")
                    .links(vec![MemoryLink::new("test2", None::<String>,"test".to_string(),1.0)])
                    .build(),
            ]
        );
        let new_mem_note = vec![
            MemoryNoteBuilder::new("test4")
                .id("test4")
                .links(vec![MemoryLink::new("test3", None::<String>,"test".to_string(),1.0)])
                .build(),
            MemoryNoteBuilder::new("test5")
                .id("test5")
                .links(vec![MemoryLink::new("test4", None::<String>,"test".to_string(),1.0)])
                .build(),
        ];
        mem.merge_mem_graph(new_mem_note);
        assert_eq!(mem.cluster.graph.node_count(), 5);
        assert_eq!(mem.cluster.graph.edge_count(), 6);
    }
    #[test]
    fn test_add_temp_mem() {
        let mut mem = prepare_working_memory(vec![]);
        mem.add_temp_mem(
            MemoryNoteBuilder::new("test")
                .id("test")
                .links(vec![MemoryLink::new("test2", None::<String>,"test".to_string(),1.0)])
                .build(),
            MemorySource::Dialogue("test".to_string())
        );
        assert_eq!(mem.cluster.graph.node_count(), 1);
        assert_eq!(mem.cluster.graph.edge_count(), 0);
        assert_eq!(mem.temporary.get_all().len(), 1);
    }
    #[test]
    fn test_focus_context() {
        let mut mem = prepare_working_memory(
            vec![
                MemoryNoteBuilder::new("test1")
                    .id("test1")
                    .links(
                        vec![
                            MemoryLink::new("test2", None::<String>,"test".to_string(), 1.0)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test2")
                    .id("test2")
                    .links(
                        vec![
                            MemoryLink::new("test3", None::<String>,"test".to_string(), 1.0)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test3")
                    .id("test3")
                    .links(
                        vec![
                            MemoryLink::new("test4", None::<String>,"test".to_string(), 1.0)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test4")
                    .id("test4")
                    .links(
                        vec![
                            MemoryLink::new("test5", None::<String>,"test".to_string(), 1.0)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test5")
                    .id("test5")
                    .links(
                        vec![
                            MemoryLink::new("test6", None::<String>,"test".to_string(), 1.0)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test6")
                    .id("test6")
                    .links(
                        vec![
                            MemoryLink::new("test7", None::<String>,"test".to_string(), 1.0)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test7")
                    .id("test7")
                    .links(
                        vec![
                            MemoryLink::new("test8", None::<String>,"test".to_string(), 1.0)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test8")
                    .id("test8")
                    .build(),
            ]
        );
        mem.add_task("task1",&vec![NodeRefId::from("test1")]);
        mem.add_task("task2",&vec![NodeRefId::from("test2")]);
        mem.add_task("task3",&vec![NodeRefId::from("test3")]);
        let context = mem.focus_context(6,1,&mut rand::rng());
        assert_eq!(context.len(), 4,"context is: {context:?}");
    }
    #[test]
    fn test_filter_need_consolidate() {
        let mut mem = prepare_working_memory(
            vec![
                MemoryNoteBuilder::new("test1").id("test1").build(),
                MemoryNoteBuilder::new("test2").id("test2").build(),
                MemoryNoteBuilder::new("test3").id("test3").build(),
            ]
        );
        mem.add_temp_mem(
            MemoryNoteBuilder::new("test4").id("test4").build(),
            MemorySource::Dialogue("test".to_string())
        );
        let con = mem.filter_need_consolidate(
            |note| note.activation_count() > 1,
            |note| note.activation_count() > 0
        );
        assert_eq!(con.len(), 1);
        assert_eq!(con[0].0.mem_id, NodeRefId::from("test4"));

    }

}

