use std::collections::{HashMap, HashSet};

// ========================================================================
// 知识图谱数据结构
// ========================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct KGVertex {
    pub id: usize,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KGEdge {
    pub from: usize,
    pub to: usize,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct KnowledgeGraph {
    pub vertices: Vec<KGVertex>,
    pub edges: Vec<KGEdge>,
}

// ========================================================================
// 图编辑操作定义（六种）
// ========================================================================

#[derive(Clone, Debug)]
pub enum GEditOp {
    AddVertex { label: String },
    DeleteVertex { id: usize, label: String },
    ModifyVertex { id: usize, old_label: String, new_label: String },
    AddEdge { from: usize, to: usize, label: String },
    DeleteEdge { from: usize, to: usize, label: String },
    ModifyEdge { from: usize, to: usize, old_label: String, new_label: String },
}

// ========================================================================
// 字符串相似度（trigram Jaccard，适合中文）
// ========================================================================

fn trigrams(s: &str) -> HashSet<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= 3 {
        return [s.to_string()].into_iter().collect();
    }
    chars.windows(3).map(|w| w.iter().collect::<String>()).collect()
}

fn trigram_similarity(a: &str, b: &str) -> f64 {
    let ta = trigrams(a);
    let tb = trigrams(b);
    let intersection = ta.intersection(&tb).count();
    let union = ta.union(&tb).count();
    if union == 0 { 1.0 } else { intersection as f64 / union as f64 }
}

// ========================================================================
// 顶点匹配（贪心）
// ========================================================================

struct VertexMatch {
    old_id: usize,
    new_id: usize,
    #[allow(dead_code)]
    similarity: f64,
}

fn match_vertices_greedy(old: &[KGVertex], new: &[KGVertex], threshold: f64) -> Vec<VertexMatch> {
    let mut used_old = HashSet::new();
    let mut used_new = HashSet::new();
    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for (i, ov) in old.iter().enumerate() {
        for (j, nv) in new.iter().enumerate() {
            pairs.push((i, j, trigram_similarity(&ov.label, &nv.label)));
        }
    }
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut matches = Vec::new();
    for (oi, nj, sim) in &pairs {
        if *sim < threshold { break; }
        if !used_old.contains(oi) && !used_new.contains(nj) {
            used_old.insert(*oi);
            used_new.insert(*nj);
            matches.push(VertexMatch { old_id: *oi, new_id: *nj, similarity: *sim });
        }
    }
    matches
}

// ========================================================================
// 新图谱中相似顶点合并（cos >= 0.9 → trigram 替代）
// ========================================================================

fn merge_similar_vertices(kg: &KnowledgeGraph, threshold: f64) -> KnowledgeGraph {
    let n = kg.vertices.len();
    if n <= 1 { return kg.clone(); }
    let mut sim = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s = trigram_similarity(&kg.vertices[i].label, &kg.vertices[j].label);
            sim[i][j] = s; sim[j][i] = s;
        }
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut Vec<usize>, x: usize) -> usize {
        if p[x] != x { p[x] = find(p, p[x]); } p[x]
    }
    fn union(p: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(p, a); let rb = find(p, b); if ra != rb { p[rb] = ra; }
    }
    for i in 0..n { for j in (i + 1)..n { if sim[i][j] >= threshold { union(&mut parent, i, j); } } }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n { groups.entry(find(&mut parent, i)).or_default().push(i); }
    let mut merged = KnowledgeGraph { vertices: Vec::new(), edges: Vec::new() };
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    for (_, members) in &groups {
        let label = kg.vertices[members[0]].label.clone();
        let new_id = merged.vertices.len();
        merged.vertices.push(KGVertex { id: new_id, label });
        for &oid in members { old_to_new.insert(oid, new_id); }
    }
    for e in &kg.edges {
        if let (Some(&nf), Some(&nt)) = (old_to_new.get(&e.from), old_to_new.get(&e.to)) {
            if !merged.edges.iter().any(|ee| ee.from == nf && ee.to == nt && ee.label == e.label) {
                merged.edges.push(KGEdge { from: nf, to: nt, label: e.label.clone() });
            }
        }
    }
    merged
}

// ========================================================================
// 图编辑距离计算
// ========================================================================

fn compute_edit_distance(old: &KnowledgeGraph, new: &KnowledgeGraph) -> Vec<GEditOp> {
    let merged_new = merge_similar_vertices(new, 0.9);
    let matches = match_vertices_greedy(&old.vertices, &merged_new.vertices, 0.4);
    let mut old_matched = HashSet::new();
    let mut new_matched = HashSet::new();
    let mut new_to_old: HashMap<usize, usize> = HashMap::new();
    let mut old_to_new: HashMap<usize, usize> = HashMap::new();
    for m in &matches {
        old_matched.insert(m.old_id);
        new_matched.insert(m.new_id);
        new_to_old.insert(m.new_id, m.old_id);
        old_to_new.insert(m.old_id, m.new_id);
    }
    let mut ops: Vec<GEditOp> = Vec::new();
    for (i, v) in old.vertices.iter().enumerate() {
        if !old_matched.contains(&i) {
            ops.push(GEditOp::DeleteVertex { id: i, label: v.label.clone() });
        } else if let Some(&nj) = old_to_new.get(&i) {
            if old.vertices[i].label != merged_new.vertices[nj].label {
                ops.push(GEditOp::ModifyVertex {
                    id: i,
                    old_label: old.vertices[i].label.clone(),
                    new_label: merged_new.vertices[nj].label.clone(),
                });
            }
        }
    }
    for (j, v) in merged_new.vertices.iter().enumerate() {
        if !new_matched.contains(&j) {
            ops.push(GEditOp::AddVertex { label: v.label.clone() });
        }
    }
    let mut new_edge_map: HashMap<(usize, usize), &str> = HashMap::new();
    for e in &merged_new.edges {
        if let (Some(&of), Some(&ot)) = (new_to_old.get(&e.from), new_to_old.get(&e.to)) {
            new_edge_map.insert((of, ot), &e.label);
        }
    }
    let mut old_edge_checked = HashSet::new();
    for e in &old.edges {
        if !old_matched.contains(&e.from) || !old_matched.contains(&e.to) { continue; }
        let key = (e.from, e.to);
        old_edge_checked.insert(key);
        if let Some(&nl) = new_edge_map.get(&key) {
            if nl != e.label {
                ops.push(GEditOp::ModifyEdge { from: e.from, to: e.to, old_label: e.label.clone(), new_label: nl.to_string() });
            }
        } else {
            ops.push(GEditOp::DeleteEdge { from: e.from, to: e.to, label: e.label.clone() });
        }
    }
    for e in &merged_new.edges {
        if let (Some(&of), Some(&ot)) = (new_to_old.get(&e.from), new_to_old.get(&e.to)) {
            if !old_edge_checked.contains(&(of, ot)) {
                ops.push(GEditOp::AddEdge { from: of, to: ot, label: e.label.clone() });
            }
        }
    }
    ops
}

// ========================================================================
// 变换评分
// ========================================================================

pub fn op_cost(op: &GEditOp) -> f64 {
    match op {
        GEditOp::AddVertex { .. } | GEditOp::DeleteVertex { .. } => 0.5,
        GEditOp::ModifyVertex { .. } => 0.3,
        GEditOp::AddEdge { .. } | GEditOp::DeleteEdge { .. } => 0.5,
        GEditOp::ModifyEdge { .. } => 0.3,
    }
}

pub fn compute_transform_score(ops: &[GEditOp]) -> f64 {
    let len = ops.len() as f64;
    let sum_costs: f64 = ops.iter().map(op_cost).sum();
    if len == 0.0 { 0.0 } else { len * sum_costs }
}

// ========================================================================
// LLM 提取知识图谱
// ========================================================================

async fn extract_kg_from_text<F, Fut>(
    text: &str,
    llm_call: F,
) -> Result<KnowledgeGraph, Box<dyn std::error::Error + Send + Sync>>
where
    F: FnOnce(&str, &str) -> Fut,
    Fut: std::future::Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>>,
{
    let system = "You are a knowledge graph extractor. Given a text, extract entities (vertices) and relationships (edges).\n\
        Respond ONLY in this format:\n\
        Vertices:\n- entity1\n- entity2\n...\n\
        Edges:\n- entity1 -- relation --> entity2\n...\n\
        Each vertex must be a noun phrase. Each edge must connect two existing vertices.";

    let user = format!("Text: {}", text);
    let response = llm_call(system, &user).await?;

    let mut kg = KnowledgeGraph { vertices: Vec::new(), edges: Vec::new() };
    let mut vertex_map: HashMap<String, usize> = HashMap::new();
    let mut in_vertices = false;
    let mut in_edges = false;

    for line in response.lines() {
        let line = line.trim();
        if line.eq_ignore_ascii_case("Vertices:") || line.eq_ignore_ascii_case("Vertices") { in_vertices = true; in_edges = false; continue; }
        if line.eq_ignore_ascii_case("Edges:") || line.eq_ignore_ascii_case("Edges") { in_vertices = false; in_edges = true; continue; }
        if in_vertices && (line.starts_with("- ") || line.starts_with("* ")) {
            let label = line[2..].trim().trim_matches('"').to_string();
            if !label.is_empty() && !vertex_map.contains_key(&label) {
                let id = kg.vertices.len();
                vertex_map.insert(label.clone(), id);
                kg.vertices.push(KGVertex { id, label });
            }
        }
        if in_edges && (line.starts_with("- ") || line.starts_with("* ")) {
            let content = &line[2..];
            if let Some(ap) = content.rfind(" --> ") {
                let before = &content[..ap];
                let target = content[ap + 5..].trim().trim_matches('"');
                if let Some(rp) = before.rfind(" -- ") {
                    let source = before[..rp].trim().trim_matches('"');
                    let rel = before[rp + 4..].trim().trim_matches('"');
                    if let (Some(&sid), Some(&tid)) = (vertex_map.get(source), vertex_map.get(target)) {
                        kg.edges.push(KGEdge { from: sid, to: tid, label: rel.to_string() });
                    }
                }
            }
        }
    }
    Ok(kg)
}

// ========================================================================
// Context 序列化 + 嵌入 + 余弦相似度
// ========================================================================

use soul_mem_core::memory_note::situation_mem::Context;
use soul_mem_query::embedding::Embeddable;

fn context_to_text(ctx: &Context) -> String {
    let mut parts = Vec::new();
    if let Some(loc) = ctx.get_location() {
        parts.push(format!("地点:{}({})", loc.name, loc.coordinates));
    }
    for p in ctx.get_participants() {
        parts.push(format!("参与者:{}({})", p.name, p.role));
    }
    for e in ctx.get_emotions() {
        parts.push(format!("情绪:{}(强度{})", e.name, e.intensity));
    }
    for s in ctx.get_sensory_data() {
        parts.push(format!("感官:{}(强度{})", s.name, s.intensity));
    }
    let env = ctx.get_environment();
    parts.push(format!("氛围:{},色调:{}", env.atmosphere, env.tone));
    for e in ctx.get_event() {
        parts.push(format!("事件:{}由{}对{}发起(强度{})", e.action, e.initiator, e.target, e.action_intensity));
    }
    parts.join("; ")
}

fn embed_context(
    ctx: &Context,
    model: &dyn soul_mem_query::embedding::EmbeddingModel,
) -> Option<soul_mem_query::embedding::situation::context::ContextEmbedding> {
    ctx.embed(model).ok()
}

fn compare_context_embeddings(
    old: &soul_mem_query::embedding::situation::context::ContextEmbedding,
    new: &soul_mem_query::embedding::situation::context::ContextEmbedding,
) -> f32 {
    let mut scores = Vec::new();

    if let (Some(ol), Some(nl)) = (old.location(), new.location()) {
        let ns = ol.name().cosine_similarity(nl.name()).unwrap_or(0.0);
        let cs = ol.coordinates().cosine_similarity(nl.coordinates()).unwrap_or(0.0);
        scores.push(ns * 0.6 + cs * 0.4);
    }

    if let (Some(op), Some(np)) = (old.fused_participant(), new.fused_participant()) {
        let ns = op.name().cosine_similarity(np.name()).unwrap_or(0.0);
        let rs = op.role().cosine_similarity(np.role()).unwrap_or(0.0);
        let fs = op.fused().cosine_similarity(np.fused()).unwrap_or(0.0);
        scores.push(ns * 0.4 + rs * 0.3 + fs * 0.3);
    }

    let oe = old.environment();
    let ne = new.environment();
    let atmo = oe.atmosphere().cosine_similarity(ne.atmosphere()).unwrap_or(0.0);
    let tone = oe.tone().cosine_similarity(ne.tone()).unwrap_or(0.0);
    scores.push(atmo * 0.5 + tone * 0.5);

    if let (Some(oev), Some(nev)) = (old.fused_event(), new.fused_event()) {
        let a = oev.action().cosine_similarity(nev.action()).unwrap_or(0.0);
        let i = oev.initiator().cosine_similarity(nev.initiator()).unwrap_or(0.0);
        let t = oev.target().cosine_similarity(nev.target()).unwrap_or(0.0);
        scores.push(a * 0.4 + i * 0.3 + t * 0.3);
    }

    // EmotionEmbedding uses .emotion() (not .name())
    if let (Some(oe), Some(ne)) = (old.fused_emotion(), new.fused_emotion()) {
        let e = oe.emotion().cosine_similarity(ne.emotion()).unwrap_or(0.0);
        scores.push(e);
    }

    // SensoryDataEmbedding uses .sensory() (not .name())
    if let (Some(os), Some(ns)) = (old.fused_sensory_data(), new.fused_sensory_data()) {
        let s = os.sensory().cosine_similarity(ns.sensory()).unwrap_or(0.0);
        scores.push(s);
    }

    if scores.is_empty() { 1.0 } else { scores.iter().sum::<f32>() / scores.len() as f32 }
}

// ========================================================================
// 构建节点辅助
// ========================================================================

use chrono::{DateTime, Utc};
use soul_mem_core::memory_note::{
    MemoryNote, MemoryNoteBuilder, MemoryType,
    sem_mem::{ConceptType, SemMemory},
    situation_mem::{
        SpecificSituation, SituationType,
        Location, Participant, Emotion, SensoryData, Environment, Event,
    },
};

fn build_situation_node(narrative: &str, created: DateTime<Utc>, ctx: Context) -> MemoryNote {
    let sit = SpecificSituation::new(narrative.to_string(), created, ctx);
    MemoryNoteBuilder::new(MemoryType::Situation(SituationType::SpecificSituation(sit)))
        .create_time(created)
        .last_accessed_time(created)
        .build().unwrap()
}

fn build_sem_node(content: &str, aliases: Vec<&str>, desc: &str, ct: ConceptType, created: DateTime<Utc>) -> MemoryNote {
    let sem = SemMemory::new(content.to_string(), ct, desc.to_string());
    let mut node = MemoryNoteBuilder::new(MemoryType::Semantic(sem))
        .create_time(created)
        .last_accessed_time(created)
        .build().unwrap();
    if let MemoryType::Semantic(s) = node.mem_type_mut() {
        s.aliases = aliases.into_iter().map(|a| a.to_string()).collect();
    }
    node
}

fn print_node_state(label: &str, node: &MemoryNote) {
    println!("【{}】", label);
    println!("  id:              {}", node.id());
    println!("  create_time:     {:?}", node.creation_time());
    println!("  retrieval_count: {}", node.retrieval_count());
    match node.mem_type() {
        MemoryType::Semantic(s) => {
            println!("  type:            SemMemory");
            println!("  content:         {}", s.content);
            println!("  aliases:         {:?}", s.aliases);
            println!("  description:     {}", s.description);
            println!("  concept_type:    {:?}", s.concept_type);
        }
        MemoryType::Situation(SituationType::SpecificSituation(s)) => {
            println!("  type:            SpecificSituation");
            println!("  narrative:       {}", s.get_narrative());
            println!("  time_span:       {:?}", s.get_time_span());
            println!("  context:         {}", context_to_text(s.get_context()));
        }
        _ => {}
    }
    println!();
}

// ========================================================================
// 集成测试主函数
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use soul_mem_runtime::working_memory::llm::client::LlmClient;
    use soul_mem_runtime::working_memory::llm::config::LLMConfig;
    use crate::algo::forget::decay_revise;

    fn try_create_llm_client() -> Option<LlmClient> {
        let key = std::env::var("API_KEY").ok()?;
        let base = std::env::var("API_BASE").ok()?;
        let model = std::env::var("MODEL").ok()?;
        Some(LlmClient::new(LLMConfig::new(&key, &base, &model)))
    }

    fn make_llm_closure(client: Arc<LlmClient>) -> impl FnOnce(&str, &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, Box<dyn std::error::Error + Send + Sync>>> + Send>> {
        move |sys: &str, user: &str| {
            let c = client.clone();
            let s = sys.to_string();
            let u = user.to_string();
            Box::pin(async move {
                use async_openai::types::chat::{ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage};
                let messages = vec![
                    ChatCompletionRequestSystemMessage::from(s).into(),
                    ChatCompletionRequestUserMessage::from(u).into(),
                ];
                let mut resp = c.call_llm(messages).await
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
                Ok(resp.remove(0))
            })
        }
    }

    /// 对单个节点执行遗忘，返回遗忘前后的概要
    async fn run_forget(
        node: &mut MemoryNote,
        now: DateTime<Utc>,
        jieba: &jieba_rs::Jieba,
        client: Arc<LlmClient>,
    ) -> (String, String, decay_revise::ForgetAction) {
        let before = get_summary(node).unwrap_or_default();
        let cl = client.clone();
        let result = decay_revise::lazy_forget(node, now, jieba, make_llm_closure(cl)).await;
        let after = get_summary(node).unwrap_or_default();
        (before, after, result)
    }

    use decay_revise::get_summary;

    #[tokio::test]
    #[ignore]
    async fn test_integration_forget_full() {
        dotenvy::dotenv().ok();
        let client = try_create_llm_client().expect("请设置 API_KEY, API_BASE, MODEL");
        let client = Arc::new(client);

        println!();
        println!("========================================================================");
        println!("  集成遗忘测试：知识图谱变换 + Context 相似度 + 变换评分");
        println!("========================================================================");
        println!();

        let embedding_model = match soul_mem_query::embedding::embedding_model::qwen3::Qwen3Embedding600M::default_cpu() {
            Ok(m) => Some(m),
            Err(e) => {
                println!("(Qwen3 模型加载失败: {}，余弦相似度将显示 -1)", e);
                None
            }
        };
        let jieba = jieba_rs::Jieba::new();
        let now = Utc::now();
        let t1 = now - chrono::Duration::hours(48);

        // ---- 创建 4 个节点 ----
        let ctx1 = Context::new(
            Some(Location { name: "中山公园".to_string(), coordinates: "39.9,116.4".to_string() }),
            vec![Participant { name: "我".to_string(), role: "本人".to_string() }],
            vec![Emotion { name: "放松".to_string(), intensity: 0.7 }],
            vec![SensoryData { name: "鸟鸣".to_string(), intensity: 0.5 }],
            Environment { atmosphere: "宁静".to_string(), tone: "温暖".to_string() },
            vec![Event { action: "散步".to_string(), action_intensity: 0.3, initiator: "我".to_string(), target: "公园".to_string() }],
        );
        let narrative1 = "傍晚我在中山公园散步听到鸟鸣和风吹树叶的声音感受着温暖的阳光和宁静的氛围";

        let ctx2 = Context::new(
            Some(Location { name: "星巴克".to_string(), coordinates: "120.1,30.2".to_string() }),
            vec![Participant { name: "张三".to_string(), role: "同事".to_string() }, Participant { name: "李四".to_string(), role: "客户".to_string() }],
            vec![Emotion { name: "愉悦".to_string(), intensity: 0.8 }],
            vec![SensoryData { name: "咖啡香".to_string(), intensity: 0.6 }],
            Environment { atmosphere: "轻松".to_string(), tone: "明亮".to_string() },
            vec![Event { action: "讨论".to_string(), action_intensity: 0.5, initiator: "张三".to_string(), target: "项目方案".to_string() }],
        );
        let narrative2 = "下午我和张三李四在星巴克讨论项目方案闻着咖啡香气氛轻松愉快";

        // --- 打印遗忘前状态 ---
        let mut n1 = build_situation_node(narrative1, t1, ctx1.clone());
        let mut n2 = build_situation_node(narrative2, t1, ctx2.clone());
        let mut n3 = build_sem_node(
            "Rust是一门由Mozilla主导研发的系统级编程语言以其内存安全和高并发零成本抽象著称",
            vec!["Rust语言", "Rust-lang"], "系统级编程语言", ConceptType::Entity, t1,
        );
        let mut n4 = build_sem_node(
            "机器学习是人工智能的核心分支通过数据训练模型实现预测和决策广泛应用于图像识别自然语言处理等领域",
            vec!["ML", "Machine Learning"], "AI核心技术", ConceptType::Abstract, t1,
        );

        println!("=== 遗忘前节点完整状态 ===");
        println!();
        print_node_state("节点1 - SpecificSituation(公园散步)", &n1);
        print_node_state("节点2 - SpecificSituation(咖啡馆会面)", &n2);
        print_node_state("节点3 - SemMemory(Rust)", &n3);
        print_node_state("节点4 - SemMemory(机器学习)", &n4);
        println!("==========================================");
        println!();

        // ---- 保存原始摘要用于后续对比 ----
        let _orig_n1_summary = get_summary(&n1).unwrap_or_default();
        let _orig_n2_summary = get_summary(&n2).unwrap_or_default();
        let orig_n3_summary = get_summary(&n3).unwrap_or_default();
        let orig_n4_summary = get_summary(&n4).unwrap_or_default();

        // ---- 执行遗忘 ----
        let r1 = run_forget(&mut n1, now, &jieba, client.clone()).await;
        let r2 = run_forget(&mut n2, now, &jieba, client.clone()).await;
        let r3 = run_forget(&mut n3, now, &jieba, client.clone()).await;
        let r4 = run_forget(&mut n4, now, &jieba, client.clone()).await;
        let nodes = [&n1, &n2, &n3, &n4];
        let actions = [&r1, &r2, &r3, &r4];
        let names = [
            "节点1 - SpecificSituation(公园散步)",
            "节点2 - SpecificSituation(咖啡馆会面)",
            "节点3 - SemMemory(Rust)",
            "节点4 - SemMemory(机器学习)",
        ];

        // ---- 打印遗忘过程 ----
        println!("=== 遗忘过程 + 新节点内容 ===");
        println!();
        for (i, node) in nodes.iter().enumerate() {
            let (before, after, action) = actions[i];
            println!("【{}】", names[i]);
            match action {
                decay_revise::ForgetAction::Revised { masked_text, .. } => {
                    println!("  操作: Revised");
                    println!("  遗忘前: {}", before);
                    println!("  遮罩:   {}", masked_text);
                    println!("  遗忘后: {}", after);
                }
                decay_revise::ForgetAction::MaskOnly { missing_degree, masked_count: _, masked_text } => {
                    println!("  操作: MaskOnly (缺失度:{:.1}%)", missing_degree * 100.0);
                    println!("  遗忘前: {}", before);
                    println!("  遮罩:   {}", masked_text);
                    println!("  遗忘后: {}", after);
                }
                decay_revise::ForgetAction::NoAction => {
                    println!("  操作: NoAction");
                    println!("  内容: {}", get_summary(*node).unwrap_or_default());
                }
            }
            println!();
        }

        // ---- SpecificSituation 分析 ----
        println!("=== SpecificSituation 节点分析 ===");
        println!();
        for (i, node) in nodes.iter().enumerate() {
            if i >= 2 { break; }
            let (orig_ctx, _orig_narr) = match node.mem_type() {
                MemoryType::Situation(SituationType::SpecificSituation(s)) => {
                    let ctx = if i == 0 { &ctx1 } else { &ctx2 };
                    (ctx, s.get_narrative().clone())
                }
                _ => continue,
            };
            let label = names[i];
            let new_ctx = match node.mem_type() {
                MemoryType::Situation(SituationType::SpecificSituation(s)) => s.get_context(),
                _ => continue,
            };
            let orig_text = context_to_text(orig_ctx);
            let new_text = context_to_text(new_ctx);

            let cos = match embedding_model.as_ref() {
                Some(m) => {
                    let old_emb = embed_context(orig_ctx, m);
                    let new_emb = embed_context(new_ctx, m);
                    match (old_emb, new_emb) {
                        (Some(o), Some(n)) => compare_context_embeddings(&o, &n),
                        _ => -2.0,
                    }
                }
                None => -2.0,
            };

            println!("【{} Context 相似度】", label);
            println!("  原始 context: {}", orig_text);
            println!("  新 context:   {}", new_text);
            println!("  余弦相似度:   {:.6}", cos);
            println!();

            let new_narr = get_summary(node).unwrap_or_default();
            let cl = client.clone();
            match extract_kg_from_text(&new_narr, make_llm_closure(cl)).await {
                Ok(kg) => {
                    println!("  新概要知识图谱:");
                    println!("    顶点: {:?}", kg.vertices.iter().map(|v| &v.label).collect::<Vec<_>>());
                    println!("    边数: {}", kg.edges.len());
                }
                Err(e) => println!("  KG提取失败: {}", e),
            }
            println!();
        }

        // ---- SemMemory 分析 ----
        println!("=== SemMemory 节点知识图谱变换分析 ===");
        println!();
        for (i, node) in nodes.iter().enumerate() {
            if i < 2 { continue; }
            let orig_content = if i == 2 { &orig_n3_summary } else { &orig_n4_summary };
            let new_content = get_summary(node).unwrap_or_default();
            let label = names[i];

            let cl1 = client.clone();
            let old_kg = extract_kg_from_text(orig_content, make_llm_closure(cl1)).await
                .unwrap_or_else(|_| KnowledgeGraph { vertices: Vec::new(), edges: Vec::new() });
            let cl2 = client.clone();
            let new_kg = extract_kg_from_text(&new_content, make_llm_closure(cl2)).await
                .unwrap_or_else(|_| KnowledgeGraph { vertices: Vec::new(), edges: Vec::new() });

            let ops = compute_edit_distance(&old_kg, &new_kg);
            let score = compute_transform_score(&ops);

            println!("【{} 知识图谱变换】", label);
            println!("  原始 content: {}", orig_content);
            println!("  遗忘后 content: {}", new_content);
            println!();
            println!("  原 KG: {} 顶点, {} 边", old_kg.vertices.len(), old_kg.edges.len());
            if !old_kg.vertices.is_empty() {
                println!("    顶点: {:?}", old_kg.vertices.iter().map(|v| &v.label).collect::<Vec<_>>());
            }
            println!("  新 KG: {} 顶点, {} 边", new_kg.vertices.len(), new_kg.edges.len());
            if !new_kg.vertices.is_empty() {
                println!("    顶点: {:?}", new_kg.vertices.iter().map(|v| &v.label).collect::<Vec<_>>());
            }
            println!();
            println!("  编辑操作序列 ({} 步):", ops.len());
            for (oi, op) in ops.iter().enumerate() {
                println!("    {}. {:?} (成本: {})", oi + 1, op, op_cost(op));
            }
            let sum_c: f64 = ops.iter().map(op_cost).sum();
            println!("  变换评分: S = {} × {:.4} = {:.4}", ops.len(), sum_c, score);
            println!();
        }

        // ---- 最终汇总 ----
        println!("========================================================================");
        println!("  测试结果汇总");
        println!("========================================================================");
        println!();
        println!("【节点遗忘触发情况】");
        for (i, node) in nodes.iter().enumerate() {
            let (_before, after, action) = actions[i];
            match action {
                decay_revise::ForgetAction::Revised { .. } => {
                    println!("  {}: Revised → {}", names[i], after);
                }
                decay_revise::ForgetAction::MaskOnly { .. } => {
                    println!("  {}: MaskOnly → {}", names[i], after);
                }
                decay_revise::ForgetAction::NoAction => {
                    println!("  {}: NoAction (内容: {})", names[i], get_summary(*node).unwrap_or_default());
                }
            }
        }
        println!();
        println!("【SpecificSituation Context 余弦相似度】");
        for (i, node) in nodes.iter().enumerate() {
            if i >= 2 { break; }
            let (orig_ctx, _) = if i == 0 { (&ctx1, "..") } else { (&ctx2, "..") };
            let new_ctx = match node.mem_type() {
                MemoryType::Situation(SituationType::SpecificSituation(s)) => s.get_context(),
                _ => continue,
            };
            let cos = match embedding_model.as_ref() {
                Some(m) => {
                    match (embed_context(orig_ctx, m), embed_context(new_ctx, m)) {
                        (Some(o), Some(n)) => compare_context_embeddings(&o, &n),
                        _ => -2.0,
                    }
                }
                None => -2.0,
            };
            println!("  {}: {:.6}", names[i], cos);
        }
        println!();
        println!("【SemMemory KG 变换评分】");
        for (i, node) in nodes.iter().enumerate() {
            if i < 2 { continue; }
            let orig_c = if i == 2 { &orig_n3_summary } else { &orig_n4_summary };
            let new_c = get_summary(node).unwrap_or_default();
            let cl1 = client.clone();
            let ok = extract_kg_from_text(orig_c, make_llm_closure(cl1)).await.unwrap_or_else(|_| KnowledgeGraph { vertices: Vec::new(), edges: Vec::new() });
            let cl2 = client.clone();
            let nk = extract_kg_from_text(&new_c, make_llm_closure(cl2)).await.unwrap_or_else(|_| KnowledgeGraph { vertices: Vec::new(), edges: Vec::new() });
            let ops = compute_edit_distance(&ok, &nk);
            let score = compute_transform_score(&ops);
            println!("  {}: {} 步操作, 评分 {:.4}", names[i], ops.len(), score);
        }
        println!();
        println!("========================================================================");
    }
}
