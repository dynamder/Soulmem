use std::{collections::HashMap, hash::Hash};

use petgraph::{
    algo::UnitMeasure,
    visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};

#[track_caller]
pub fn naive_ppr<G, D>(
    graph: G,
    damping_factor: D,
    source_bias: HashMap<G::NodeId, D>,
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
    let d_node_count = D::from_usize(node_count);

    //检查个性化分布是不是一个概率分布
    let bias_sum: D = source_bias.values().copied().sum();
    assert!(
        bias_sum > D::zero(),
        "Personalized Source bias sum must be positive"
    );
    //归一化个性化向量（初始向量）
    let normalized_bias: HashMap<G::NodeId, D> = if bias_sum != D::one() {
        source_bias
            .into_iter()
            .map(|(node_id, bias)| (node_id, bias / bias_sum))
            .collect()
    } else {
        source_bias
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

    //初始化PPR向量
    valid_index.iter().for_each(|&valid_index| {
        ppr_ranks[valid_index] = D::one() / d_node_count //SAFEUNWRAP: 已经预先分配了索引上限大小的内存，不会越界访问。
    });

    //预计算每个节点的出度
    graph.node_identifiers().for_each(|node_id| {
        out_degrees[graph.to_index(node_id)] = graph.edges(node_id).map(|_| D::one()).sum();
    });

    for _ in 0..nb_iter {
        todo!("Implement the PPR algorithm")
    }

    //返回PPR向量，HashMap形式
    graph
        .node_identifiers()
        .map(|node_id| (node_id, ppr_ranks[graph.to_index(node_id)]))
        .collect()
}
