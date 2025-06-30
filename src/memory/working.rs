///工作记忆
mod task;

use std::collections::{HashMap, HashSet};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;

use crate::memory::{GraphMemoryLink,MemoryNote};
use anyhow::Result;
use chrono::DateTime;
use formatx::formatx;
use petgraph::Direction;
use petgraph::prelude::EdgeRef;
use petgraph::visit::{Dfs, VisitMap, Visitable};
use crate::llm_driver::{LLMConfig, Llm};
use crate::memory::share::NoteRecord;
use crate::memory::temporary::{MemorySource, TemporaryMemory, TemporaryNoteRecord};
use crate::memory::working::task::{SoulTask, SoulTaskSet};
use super::default_prompts;

const DEFAULT_FOCUS: &str = "发呆";
///工作记忆记录，记录记忆节点被提取，激活的情况，用于记忆进化和巩固
#[derive(Debug,Clone)]
pub struct WorkingNoteRecord {
    #[allow(unused)]
    pub index: NodeIndex, // 节点索引

    activation_count: u32, // 激活次数
    activation_history: HashMap<NodeIndex, u32>, // 激活历史信息（共激活节点索引，次数）
    last_accessed: DateTime<chrono::Utc>,// 最后访问时间
}
#[allow(dead_code)]
impl WorkingNoteRecord {
    pub fn new(index: NodeIndex) -> Self {
        Self {
            index,
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
    fn record_activation(&mut self, indexes: &[NodeIndex]) {
        indexes.iter().for_each(|index| {
            if let Some(activations) = self.activation_history.get_mut(index) {
                *activations += 1;
            } else {
                self.activation_history.insert(*index, 1);
            }
        });
        self.activation_count += 1;
        self.last_accessed = chrono::Utc::now();
    }
    fn activation_history(&self) -> &HashMap<NodeIndex, u32> {
        &self.activation_history
    }
}

#[allow(dead_code)]
#[derive(Debug,Clone)]
pub struct WorkingMemory {
    graph: StableGraph<MemoryNote, GraphMemoryLink>, //记忆图
    working_record_map: HashMap<NodeIndex, WorkingNoteRecord>, //工作记忆激活记录映射
    uuid_to_index: HashMap<String,NodeIndex>, //uuid到节点索引的映射
    temporary: TemporaryMemory, //临时记忆
    task_set: SoulTaskSet, //任务集
}
#[allow(unused)]
impl WorkingMemory {
    pub fn new(task_inertia: f32) -> Result<WorkingMemory> {
        Ok(
            WorkingMemory {
                graph: StableGraph::new(),
                uuid_to_index: HashMap::new(),
                temporary: TemporaryMemory::new(),
                working_record_map: HashMap::new(),
                task_set: SoulTaskSet::new(task_inertia)?,
            }
        )
    }
    /// 添加任务
    pub fn add_task(&mut self, task_summary: impl Into<String>, related_notes: &[String] ) {
        self.task_set.add_task(
            SoulTask::new(
                task_summary.into(),
                related_notes.to_vec()
            )
        );
    }
    /// 清理无效索引
    pub fn clean_index(&mut self) {
        // 使用更高效的存在性检查
        let valid_nodes: HashSet<NodeIndex> = self.graph.node_indices().collect();

        // 步骤1：清理id_to_index映射
        self.uuid_to_index.retain(|_, idx| valid_nodes.contains(idx));

        // 步骤2：清理临时节点记录
        self.temporary.get_map_mut().retain(|idx,_| valid_nodes.contains(idx));

        self.working_record_map.retain(|idx,_| valid_nodes.contains(idx));
    }

    /// 删除一个节点
    pub fn remove_note(&mut self, node_id: &str) -> Option<MemoryNote> {
        if let Some(idx) = self.uuid_to_index.remove(node_id) {
            self.temporary.remove_temp_memory(idx);
            let node = self.graph.remove_node(idx);
            self.clean_index(); // 删除后立即清理
            node
        } else {
            None
        }
    }
    /// 获取一个节点记忆
    pub fn get_note(&self, node_id: &str) -> Option<&MemoryNote> {
        self.uuid_to_index.get(node_id).and_then(|&idx| self.graph.node_weight(idx))
    }
    pub fn get_note_mut(&mut self, node_id: &str) -> Option<&mut MemoryNote> {
        self.uuid_to_index.get(node_id)
            .and_then(|&idx| self.graph.node_weight_mut(idx))
    }
    ///将一系列记忆合并到工作记忆图中
    pub fn merge_mem_graph(&mut self, mem: Vec<MemoryNote>) {

        // 节点更新/添加队列
        let mut updated_indices = Vec::new();

        // 合并新记忆节点
        for note in mem {
            let id = note.id().to_owned();

            match self.uuid_to_index.get(&id) {
                // 更新已有节点（包括临时节点）
                Some(&idx) => {
                    self.graph[idx] = note;
                    updated_indices.push(idx);
                }
                // 添加新节点
                None => {
                    let idx = self.graph.add_node(note);
                    self.uuid_to_index.insert(id, idx);
                    updated_indices.push(idx);
                }
            }
        }

        // 收集所有需要添加的边
        let mut edges_to_add = Vec::new();

        for idx in &updated_indices {
            // 清理旧出边
            let edges: Vec<_> = self.graph
                .edges_directed(*idx, petgraph::Direction::Outgoing)
                .map(|e| e.id())
                .collect();

            for edge in edges {
                self.graph.remove_edge(edge);
            }

            // 收集需要添加的新边
            let links = self.graph[*idx].links.clone(); // 克隆链接数据避免借用问题
            for link in links {
                if let Some(&target) = self.uuid_to_index.get(&link.id) {
                    edges_to_add.push((*idx, target, GraphMemoryLink::from(link)));
                } else {
                    eprintln!("Linked node {} not found in graph", link.id);
                }
            }
        }

        // 批量添加所有边
        for (source, target, link) in edges_to_add {
            self.graph.add_edge(source, target, link);
        }
        self.clean_index();
    }

    /// 添加临时记忆
    pub fn add_temp_mem(&mut self, mem: MemoryNote,source: MemorySource) {
        let id = mem.id().to_owned();
        let links = mem.links().to_owned();
        let idx = self.graph.add_node(mem);
        self.uuid_to_index.insert(id, idx);
        self.temporary.add_temp_memory(
            TemporaryNoteRecord::new(idx,source)
        );
        for link in links {
            if let Some(link_idx) = self.uuid_to_index.get(&link.id){
                self.graph.add_edge(idx,*link_idx,GraphMemoryLink::from(link));
            }
        }
    }
    ///获取当前焦点的LLM上下文,并维护激活信息
    pub fn focus_context(&mut self, sliding_window_size: usize, depth: usize, rng: &mut impl rand::RngCore) -> Vec<String> {
        // 1. 从任务集中采样初始节点
        let sampled_indices = self.task_set.focus_sample(sliding_window_size, rng);

        //查找索引
        let sampled_indexes= sampled_indices
            .iter()
            .filter_map(|uuid| self.uuid_to_index.get(uuid).copied())
            .collect::<Vec<_>>();


        // 2. 创建访问映射和结果集合
        let mut visited = self.graph.visit_map(); // 使用petgraph的高效访问映射
        let mut result_contents = Vec::new();
        let mut refered_notes = Vec::new();

        let mut neighbor_stack = sampled_indexes;
        let mut depth_count = 0;
        let mut temp_stack = Vec::new();

        while !neighbor_stack.is_empty() {
            for &index in neighbor_stack.iter() {
               if visited.is_visited(&index) {
                   continue;
               }
                visited.visit(index);
                if let Some(note) = self.graph.node_weight(index) {
                    result_contents.push(note.content.clone());
                    refered_notes.push(index);
                    if depth_count < depth {
                        temp_stack.extend(self.graph.neighbors(index));
                    }
                }
            }
            neighbor_stack.clear();
            if depth_count < depth {
                neighbor_stack.append(&mut temp_stack);
            }
            println!("{neighbor_stack:?}");
            depth_count += 1;
        }

        // 4. 记录临时记忆, 激活的长期记忆（除临时记忆以外的工作记忆）的激活历史
        for index in &refered_notes {
            if let Some(record) = self.temporary.get_temp_memory_mut(*index) {
                record.record_activation(&refered_notes)
            }else if let Some(record) = self.working_record_map.get_mut(index) {
                record.record_activation(&refered_notes);
            }
        }

        result_contents
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
                    self.graph.contains_node(record.index) &&
                    record.activation_history()
                        .keys()
                        .all(|&index| self.graph.contains_node(index))
                })
                .map(|record| {
                    (
                        self.graph[record.index].clone(),
                        record.activation_history()
                            .keys()
                            .map(|&index| {
                                self.graph[index].clone()
                            })
                            .collect::<Vec<_>>()
                        )
                });



        let plastic_working =
            self.working_record_map.values()
                .filter(|&record| {
                    filter_working(record) &&
                    self.graph.contains_node(record.index) &&
                    record.activation_history()
                        .keys()
                        .all(|&index| self.graph.contains_node(index))
                })
                .map(|record| {
                    (
                        self.graph[record.index].clone(),
                        record.activation_history()
                            .keys()
                            .map(|&index| {
                                self.graph[index].clone()
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
        let mut mem = WorkingMemory::new(0.5).unwrap();
        mem.merge_mem_graph(init);
        mem
    }
    #[test]
    fn test_clean_index() {
        let mut mem = prepare_working_memory(vec![]);
        mem.working_record_map.insert(NodeIndex::new(0), WorkingNoteRecord::new(NodeIndex::new(0)));
        mem.working_record_map.insert(NodeIndex::new(1), WorkingNoteRecord::new(NodeIndex::new(1)));
        mem.working_record_map.insert(NodeIndex::new(2), WorkingNoteRecord::new(NodeIndex::new(2)));
        mem.clean_index();
        assert_eq!(mem.working_record_map.len(), 0)
    }
    #[test]
    fn test_merge_mem_graph() {
        let mut mem = prepare_working_memory(
            vec![
                MemoryNoteBuilder::new("test1")
                    .id("test1")
                    .links(vec![
                        MemoryLink::new("test2", None::<String>,1),
                        MemoryLink::new("test3", None::<String>,2),
                    ])
                    .build(),
                MemoryNoteBuilder::new("test2")
                    .id("test2")
                    .links(vec![MemoryLink::new("test1", None::<String>,1)])
                    .build(),
                MemoryNoteBuilder::new("test3")
                    .id("test3")
                    .links(vec![MemoryLink::new("test2", None::<String>,1)])
                    .build(),
            ]
        );
        let new_mem_note = vec![
            MemoryNoteBuilder::new("test4")
                .id("test4")
                .links(vec![MemoryLink::new("test3", None::<String>,1)])
                .build(),
            MemoryNoteBuilder::new("test5")
                .id("test5")
                .links(vec![MemoryLink::new("test4", None::<String>,1)])
                .build(),
        ];
        mem.merge_mem_graph(new_mem_note);
        assert_eq!(mem.graph.node_count(), 5);
        assert_eq!(mem.graph.edge_count(), 6);
    }
    #[test]
    fn test_add_temp_mem() {
        let mut mem = prepare_working_memory(vec![]);
        mem.add_temp_mem(
            MemoryNoteBuilder::new("test")
                .id("test")
                .links(vec![MemoryLink::new("test2", None::<String>,1)])
                .build(),
            MemorySource::Dialogue("test".to_string())
        );
        assert_eq!(mem.graph.node_count(), 1);
        assert_eq!(mem.graph.edge_count(), 0);
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
                            MemoryLink::new("test2", None::<String>, 1)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test2")
                    .id("test2")
                    .links(
                        vec![
                            MemoryLink::new("test3", None::<String>, 1)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test3")
                    .id("test3")
                    .links(
                        vec![
                            MemoryLink::new("test4", None::<String>, 1)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test4")
                    .id("test4")
                    .links(
                        vec![
                            MemoryLink::new("test5", None::<String>, 1)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test5")
                    .id("test5")
                    .links(
                        vec![
                            MemoryLink::new("test6", None::<String>, 1)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test6")
                    .id("test6")
                    .links(
                        vec![
                            MemoryLink::new("test7", None::<String>, 1)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test7")
                    .id("test7")
                    .links(
                        vec![
                            MemoryLink::new("test8", None::<String>, 1)
                        ]
                    )
                    .build(),
                MemoryNoteBuilder::new("test8")
                    .id("test8")
                    .build(),
            ]
        );
        mem.add_task("task1",&vec!["test1".to_string()]);
        mem.add_task("task2",&vec!["test2".to_string()]);
        mem.add_task("task3",&vec!["test3".to_string()]);
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
        assert_eq!(con[0].0.id, "test4");
        
    }

}

