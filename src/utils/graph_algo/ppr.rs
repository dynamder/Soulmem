use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    fmt::Debug,
    hash::Hash,
};

use petgraph::{
    Direction::Incoming,
    algo::UnitMeasure,
    visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};

///PPR: ppr_s = dampling_factor * P * ppr_s + (1-damping_factor) * personalized_vec, P为转移矩阵
/// 对无出度的节点，采取与source_bias中的节点建立连接
/// 必须保证source_bias的key是有效的NodeId, 否则会得到不正确的结果
// 由于NodeId会由MemoryCluster提供，这不会造成额外的检查负担
#[track_caller]
pub fn naive_ppr<G, D>(
    graph: G,
    damping_factor: D,
    personalized_vec: HashMap<G::NodeId, D>,
    nb_iter: usize,
) -> HashMap<G::NodeId, D>
where
    G: NodeCount + IntoEdges + NodeIndexable + IntoNodeIdentifiers,
    D: UnitMeasure + Copy,
    G::NodeId: Hash + Eq,
{
    let node_count = graph.node_count();
    if node_count == 0 {
        return HashMap::new();
    }

    //检查阻尼系数
    assert!(
        D::zero() <= damping_factor && damping_factor <= D::one(),
        "Damping factor should be between 0 et 1."
    );

    //检查个性化分布是不是一个概率分布
    let personalized_sum: D = personalized_vec.values().copied().sum();
    assert!(
        personalized_sum > D::zero(),
        "Personalized Source bias sum must be positive"
    );

    //归一化个性化向量（初始向量）
    let normalized_personalized_vec: HashMap<G::NodeId, D> = if personalized_sum != D::one() {
        personalized_vec
            .into_iter()
            .map(|(node_id, bias)| (node_id, bias / personalized_sum))
            .collect()
    } else {
        personalized_vec
    };

    //图中有效的索引值，适配StableGraph(索引可能不连续)
    let valid_index = graph
        .node_identifiers()
        .map(|node_id| graph.to_index(node_id))
        .collect::<Vec<_>>();

    //ppr值的存储
    //此处可能有大量内存浪费（无效的索引值占位），考虑到工作记忆子图不会过于频繁释放和加载，这个内存开销应该是可以接受的
    let mut ppr_ranks = vec![D::zero(); graph.node_bound()];
    let mut out_degrees = vec![D::zero(); graph.node_bound()];

    //使用个性化向量，初始化PPR值向量，由于源节点有向量相似性取top-k提供（k通常不大），这样初始化通常可以加快收敛速度
    normalized_personalized_vec
        .iter()
        .for_each(|(&node_id, &bias)| {
            ppr_ranks[graph.to_index(node_id)] = bias; //SAFEUNWRAP: 已经预先分配了索引上限大小的内存，不会越界访问。
        });
    let normalized_bias_len = normalized_personalized_vec.len();
    //println!("normalized_bias: {:?}", normalized_bias);

    //预计算每个节点的出度
    graph.node_identifiers().for_each(|node_id| {
        out_degrees[graph.to_index(node_id)] = graph.edges(node_id).map(|_| D::one()).sum();
    });
    //println!("out_degrees: {:?}", out_degrees);

    for i in 0..nb_iter {
        let ppr_vec_i = valid_index
            .iter()
            .map(|&computing_idx| {
                let iter_ppr = valid_index
                    .iter()
                    .map(|&idx| {
                        //找到每个节点的出边
                        let mut out_edges = graph.edges(graph.from_index(idx));

                        //游走部分的计算，对于无出度节点，默认其连接至所有个性化向量中不为0的节点
                        if out_edges.any(|e| e.target() == graph.from_index(computing_idx)) {
                            damping_factor * ppr_ranks[idx] / out_degrees[idx]
                        } else if out_degrees[idx] == D::zero() {
                            normalized_personalized_vec
                                .get(&graph.from_index(computing_idx))
                                .map(|_| {
                                    damping_factor * ppr_ranks[idx]
                                        / D::from_usize(normalized_bias_len)
                                })
                                .unwrap_or(D::zero())
                        } else {
                            D::zero()
                        }
                    })
                    .sum::<D>();

                //随机重启部分计算
                let random_back_part = if let Some(per_i) =
                    normalized_personalized_vec.get(&graph.from_index(computing_idx))
                {
                    (D::one() - damping_factor) * *per_i
                } else {
                    D::zero()
                };

                (computing_idx, iter_ppr + random_back_part)
            })
            .collect::<Vec<_>>();

        // 归一化PPR值，确保数值稳定，总和为1
        let sum = ppr_vec_i.iter().map(|(_, ppr)| *ppr).sum::<D>();

        ppr_vec_i.iter().for_each(|&(idx, ppr)| {
            ppr_ranks[idx] = ppr / sum;
        });
        //println!("iteration {i}: PPR values: {:?}", ppr_ranks);
    }

    //最终归一化
    let sum = ppr_ranks.iter().map(|ppr| *ppr).sum::<D>();

    //返回PPR向量，HashMap形式
    graph
        .node_identifiers()
        .map(|node_id| (node_id, ppr_ranks[graph.to_index(node_id)] / sum))
        .collect()
}

//用于BinaryHeap的残差单元表示
struct ResidueUnit<Index, D_R: UnitMeasure + Copy> {
    pub idx: Index,
    pub value: D_R,
}
impl<Index, D_R> PartialOrd for ResidueUnit<Index, D_R>
where
    D_R: UnitMeasure + Copy + PartialOrd,
{
    fn ge(&self, other: &Self) -> bool {
        self.value.ge(&other.value)
    }
    fn gt(&self, other: &Self) -> bool {
        self.value.gt(&other.value)
    }
    fn le(&self, other: &Self) -> bool {
        self.value.le(&other.value)
    }
    fn lt(&self, other: &Self) -> bool {
        self.value.lt(&other.value)
    }
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}
impl<Index, D_R> PartialEq for ResidueUnit<Index, D_R>
where
    D_R: UnitMeasure + Copy + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
    fn ne(&self, other: &Self) -> bool {
        self.value.ne(&other.value)
    }
}
impl<Index, D_R> Eq for ResidueUnit<Index, D_R> where D_R: UnitMeasure + Copy + Eq {}
impl<Index, D_R> Ord for ResidueUnit<Index, D_R>
where
    D_R: UnitMeasure + Copy + Ord + PartialOrd + Eq,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
    fn clamp(self, min: Self, max: Self) -> Self
    where
        Self: Sized,
    {
        Self {
            idx: self.idx,
            value: self.value.clamp(min.value, max.value),
        }
    }
    fn max(self, other: Self) -> Self
    where
        Self: Sized,
    {
        match self.cmp(&other) {
            Ordering::Less => other,
            Ordering::Equal => other,
            Ordering::Greater => self,
        }
    }
    fn min(self, other: Self) -> Self
    where
        Self: Sized,
    {
        match self.cmp(&other) {
            Ordering::Less => self,
            Ordering::Equal => self,
            Ordering::Greater => other,
        }
    }
}
type EdgeWeightUnit<Index, D> = ResidueUnit<Index, D>;

#[track_caller]
pub fn weighted_ppr_fp<G, D, Q>(
    graph: G,
    damping_factor: D,
    personalized_vec: HashMap<G::NodeId, D>,
    residue_threshold: D,
    weight_calc: impl Fn(&G::EdgeRef, &Q) -> D,
    dynamic_query: &Q,
) -> HashMap<G::NodeId, D>
where
    G: NodeCount + IntoEdges + NodeIndexable + IntoNodeIdentifiers,
    D: UnitMeasure + Copy + Ord,
    G::NodeId: Hash + Eq,
    G::EdgeId: Hash + Eq,
{
    let personalized_sum = personalized_vec.values().copied().sum::<D>();
    assert!(
        personalized_sum > D::zero(),
        "Personalized Source bias sum must be positive"
    );

    let normalized_personalized_vec: HashMap<G::NodeId, D> = if personalized_sum != D::one() {
        personalized_vec
            .into_iter()
            .map(|(node_id, bias)| (node_id, bias / personalized_sum))
            .collect()
    } else {
        personalized_vec
    };
    let source_node_count = D::from_usize(normalized_personalized_vec.len());

    let mut reserve_vec = vec![D::zero(); graph.node_bound()];
    let mut residue_vec = (0..graph.node_bound())
        .map(|i| {
            let residue_i = normalized_personalized_vec
                .get(&graph.from_index(i))
                .copied()
                .unwrap_or(D::zero());
            ResidueUnit {
                idx: i,
                value: residue_i,
            }
        })
        .collect::<BinaryHeap<_>>();

    let mut ppr_edge_weight_cache: HashMap<G::NodeId, Vec<EdgeWeightUnit<G::EdgeId, D>>> =
        HashMap::with_capacity(graph.node_count());

    while let Some(residue_i) = residue_vec.pop() {
        let out_edges = graph.edges(graph.from_index(residue_i.idx));
        //动态归一化的边权计算
        if !ppr_edge_weight_cache.contains_key(&graph.from_index(residue_i.idx)) {
            let weights = out_edges
                .map(|edge| {
                    let weight = weight_calc(&edge, dynamic_query);
                    EdgeWeightUnit {
                        idx: edge.id(),
                        value: weight,
                    }
                })
                .collect::<Vec<_>>();
            let sum = weights.iter().map(|v| v.value).sum::<D>();
            let weights = weights
                .into_iter()
                .map(|w| EdgeWeightUnit {
                    idx: w.id,
                    value: w.value / sum,
                })
                .collect::<Vec<_>>();
            ppr_edge_weight_cache.insert(graph.from_index(residue_i.idx), weights);
        }

        let edge_weights = &ppr_edge_weight_cache[&graph.from_index(residue_i.idx)];

        //残差push
        if let Some(edge_weight_max) = edge_weights.iter().max() {
            if residue_i.value * edge_weight_max.value > residue_threshold {
                todo!("push residue by weights")
            }
        } else {
            if residue_i.value / source_node_count > residue_threshold {
                todo!("push residue by weights")
            }
        }
    }
    todo!()
}

#[cfg(test)]
mod test {

    use mockall::predicate::float;
    use petgraph::{matrix_graph::NodeIndex, prelude::StableDiGraph};

    use super::*;
    fn diff(actual: f64, expected: f64) -> f64 {
        if expected.abs() < f64::EPSILON && actual.abs() < f64::EPSILON {
            0.0
        } else {
            let diff = (actual - expected).abs();
            diff
        }
    }
    fn pressure_large_graph() -> (StableDiGraph<String, f64>, Vec<NodeIndex<u32>>) {
        let mut graph = StableDiGraph::new();
        let mut nodes = Vec::new();
        for i in 0..500 {
            let mut node = graph.add_node("".to_string());
            if i % 2 == 0 || i % 7 == 0 {
                graph.remove_node(node);
                node = graph.add_node("".to_string());
            }
            nodes.push(node);
            graph.add_edge(node, node, 1.0);
            nodes.iter().for_each(|idx| {
                graph.add_edge(node, *idx, 1.0);
            });
        }
        (graph, nodes)
    }

    fn test_toy_graph() -> (StableDiGraph<String, f64>, Vec<NodeIndex<u32>>) {
        let mut graph = StableDiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        //制造索引空洞
        graph.remove_node(b);
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        let d = graph.add_node("D".to_string());

        graph.add_edge(a, b, 1.0);
        graph.add_edge(a, c, 1.0);
        graph.add_edge(b, c, 1.0);
        graph.add_edge(c, d, 1.0);

        (graph, vec![a, b, c, d])
    }
    fn toy_graph_with_init_a() -> (
        StableDiGraph<String, f64>,
        HashMap<NodeIndex<u32>, f64>,
        Vec<NodeIndex<u32>>,
    ) {
        let (graph, indexes) = test_toy_graph();
        let ans_vec: Vec<f64> = vec![0.851652742, 0.06387396045, 0.07345504972, 0.01101824785];
        let ans = indexes.iter().copied().zip(ans_vec).collect();
        (graph, ans, indexes)
    }
    fn toy_graph_with_init_b() -> (
        StableDiGraph<String, f64>,
        HashMap<NodeIndex<u32>, f64>,
        Vec<NodeIndex<u32>>,
    ) {
        let (graph, indexes) = test_toy_graph();
        let ans_vec: Vec<f64> = vec![0.0, 0.852878432, 0.1279320211, 0.01918954688];
        let ans = indexes.iter().copied().zip(ans_vec).collect();
        (graph, ans, indexes)
    }
    fn toy_graph_with_init_ab() -> (
        StableDiGraph<String, f64>,
        HashMap<NodeIndex<u32>, f64>,
        Vec<NodeIndex<u32>>,
    ) {
        let (graph, indexes) = test_toy_graph();
        let ans_vec: Vec<f64> = vec![0.4261326137, 0.4580925718, 0.1006738318, 0.00510098267];
        let ans = indexes.iter().copied().zip(ans_vec).collect();
        (graph, ans, indexes)
    }
    #[test]
    fn ppr_toy_graph_init_a() {
        let (graph, true_ans, indexes) = toy_graph_with_init_a();
        let mut source_bias = HashMap::new();
        source_bias.insert(indexes[0], 1.0);

        let ppr_ans = naive_ppr(&graph, 0.15_f64, source_bias, 15);
        let ans_sum = ppr_ans.values().copied().sum::<f64>();
        assert!(ans_sum - 1.0 < f64::EPSILON);

        let avg_diff = 0.25
            * indexes
                .iter()
                .map(|idx| {
                    let actual = ppr_ans[idx];
                    let expected = true_ans[idx];
                    diff(actual, expected)
                })
                .sum::<f64>();

        assert!(
            avg_diff < 0.005,
            "failed with avg_diff {}, whole ppr_vec is : {:?}, but it should be : {:?}",
            avg_diff,
            ppr_ans,
            true_ans
        )
    }
    #[test]
    fn ppr_toy_graph_init_b() {
        let (graph, true_ans, indexes) = toy_graph_with_init_b();
        let mut source_bias = HashMap::new();
        source_bias.insert(indexes[1], 1.0);

        let ppr_ans = naive_ppr(&graph, 0.15_f64, source_bias, 15);
        let ans_sum = ppr_ans.values().copied().sum::<f64>();
        assert!(ans_sum - 1.0 < f64::EPSILON);

        let avg_diff = 0.25
            * indexes
                .iter()
                .map(|idx| {
                    let actual = ppr_ans[idx];
                    let expected = true_ans[idx];
                    diff(actual, expected)
                })
                .sum::<f64>();

        assert!(
            avg_diff < 0.005,
            "failed with avg_diff {}, whole ppr_vec is : {:?}, but it should be : {:?}",
            avg_diff,
            ppr_ans,
            true_ans
        )
    }
    #[test]
    fn ppr_toy_graph_init_ab() {
        let (graph, true_ans, indexes) = toy_graph_with_init_ab();
        let mut source_bias = HashMap::new();
        source_bias.insert(indexes[0], 1.0);
        source_bias.insert(indexes[1], 1.0);

        let ppr_ans = naive_ppr(&graph, 0.15_f64, source_bias, 15);
        let ans_sum = ppr_ans.values().copied().sum::<f64>();
        assert!(ans_sum - 1.0 < f64::EPSILON);

        let avg_diff = 0.25
            * indexes
                .iter()
                .map(|idx| {
                    let actual = ppr_ans[idx];
                    let expected = true_ans[idx];
                    diff(actual, expected)
                })
                .sum::<f64>();

        assert!(
            avg_diff < 0.005,
            "failed with avg_diff {}, whole ppr_vec is : {:?}, but it should be : {:?}",
            avg_diff,
            ppr_ans,
            true_ans
        )
    }
    #[test]
    fn pressure_large_graph_test() {
        let (graph, nodes) = pressure_large_graph();
        let mut source_bias = HashMap::new();
        nodes.iter().take(10).for_each(|idx| {
            source_bias.insert(*idx, graph.to_index(*idx) as f64);
        });

        let ppr_ans = naive_ppr(&graph, 0.15_f64, source_bias, 15);
        let ans_sum = ppr_ans.values().copied().sum::<f64>();
        assert!(ans_sum - 1.0 < f64::EPSILON);
    }
}
