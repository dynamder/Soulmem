use std::{collections::HashMap, fmt::Debug, hash::Hash};

use petgraph::{
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
        for i in 0..5000 {
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
