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
#[derive(Debug,Clone)]
pub struct WorkingNoteRecord {
    #[allow(unused)]
    pub index: NodeIndex,

    activation_count: u32,
    activation_history: HashMap<NodeIndex, u32>,
    last_accessed: DateTime<chrono::Utc>,
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
}
impl NoteRecord for WorkingNoteRecord {
    fn activation_count(&self) -> u32 {
        self.activation_count
    }
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
    graph: StableGraph<MemoryNote, GraphMemoryLink>,
    working_record_map: HashMap<NodeIndex, WorkingNoteRecord>,
    uuid_to_index: HashMap<String,NodeIndex>,
    temporary: TemporaryMemory,
    task_set: SoulTaskSet,
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
    pub fn add_task(&mut self, task_summary: impl Into<String>, related_notes: &[String] ) {
        self.task_set.add_task(
            SoulTask::new(
                task_summary.into(),
                related_notes.to_vec()
            )
        );
    }
    pub fn clean_index(&mut self) {
        // 使用更高效的存在性检查
        let valid_nodes: HashSet<NodeIndex> = self.graph.node_indices().collect();

        // 步骤1：清理id_to_index映射
        self.uuid_to_index.retain(|_, idx| valid_nodes.contains(idx));

        // 步骤2：清理临时节点记录
        self.temporary.get_map_mut().retain(|idx,_| valid_nodes.contains(idx));

    }

    // 在删除节点的方法中调用清理函数
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
    pub fn get_note(&self, node_id: &str) -> Option<&MemoryNote> {
        self.uuid_to_index.get(node_id).and_then(|&idx| self.graph.node_weight(idx))
    }
    pub fn get_note_mut(&mut self, node_id: &str) -> Option<&mut MemoryNote> {
        self.uuid_to_index.get(node_id)
            .and_then(|&idx| self.graph.node_weight_mut(idx))
    }
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
    //获取当前焦点的LLM上下文
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

        // 3. 对每个采样节点执行深度受限的DFS
        for &start_idx in &sampled_indexes {

            // 使用petgraph的DFS遍历，限制深度
            let mut dfs = Dfs::new(&self.graph, start_idx);
            let mut depth_map = vec![None; self.graph.node_count()]; // 使用Vec存储深度

            // 设置起始节点深度为0
            let start_index = start_idx.index();
            depth_map[start_index] = Some(0);

            while let Some(node) = dfs.next(&self.graph) {
                let node_index = node.index(); // 将NodeIndex转换为usize
                let current_depth = depth_map[node_index].unwrap_or(0);

                // 如果达到深度限制，不再继续深入
                if current_depth >= depth {
                    continue;
                }

                // 记录节点内容（如果首次访问）
                if visited.visit(node) {
                    result_contents.push(self.graph[node].content.clone());
                    refered_notes.push(node);
                }

                // 遍历出边邻居
                for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                    let neighbor = edge.target();
                    let neighbor_index = neighbor.index();

                    // 如果邻居节点还未被分配深度
                    if depth_map[neighbor_index].is_none() {
                        // 设置邻居深度为当前深度+1
                        depth_map[neighbor_index] = Some(current_depth + 1);
                        dfs.stack.push(neighbor);
                    }
                }
            }
        }

        // 4. 记录临时记忆, 激活的长期记忆（除临时记忆以外的工作记忆）的激活历史
        for index in &refered_notes {
            if let Some(record) = self.temporary.get_temp_memory_mut(*index) {
                record.record_activation(&sampled_indexes)
            }else if let Some(record) = self.working_record_map.get_mut(index) {
                record.record_activation(&sampled_indexes);
            }
        }

        result_contents
    }
    ///return task_ids
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
    pub fn shift_focus(&mut self, sorted_related_task: &[impl AsRef<str>]) -> String {
        self.task_set.shift_focus(sorted_related_task)
    }
    pub fn filter_need_consolidate<F>(&self, filter: F) -> Vec<(MemoryNote,Vec<MemoryNote>)> //(记忆，可能建立的联系)
    where
        F: Fn(&dyn NoteRecord) -> bool{
        let plastic_temporary =
            self.temporary.filter_temp_memory(|record| filter(record))
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
                    filter(record) &&
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
