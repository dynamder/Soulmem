use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    fmt::Debug,
    hash::Hash,
    ops::AddAssign,
};

use petgraph::{
    algo::UnitMeasure,
    visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable},
};

///PPR: ppr_s = dampling_factor * P * ppr_s + (1-damping_factor) * personalized_vec, P为转移矩阵
/// 对无出度的节点，采取与source_bias中的节点建立连接
/// 必须保证source_bias的key是有效的NodeId, 否则会得到不正确的结果
// 由于NodeId会由MemoryCluster提供，这不会造成额外的检查负担
#[track_caller]
#[hotpath::measure]
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

    for _ in 0..nb_iter {
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

        //sum为0时跳过归一化，避免0/0产生NaN
        if sum != D::zero() {
            ppr_vec_i.iter().for_each(|&(idx, ppr)| {
                ppr_ranks[idx] = ppr / sum;
            });
        }
        //println!("iteration {i}: PPR values: {:?}", ppr_ranks);
    }

    //最终归一化
    let sum = ppr_ranks.iter().copied().sum::<D>();
    //sum为0时返回全零分布，避免0/0产生NaN
    if sum == D::zero() {
        return HashMap::new();
    }

    //返回PPR向量，HashMap形式
    graph
        .node_identifiers()
        .map(|node_id| (node_id, ppr_ranks[graph.to_index(node_id)] / sum))
        .collect()
}

//残差单元表示
#[derive(Debug, Clone, Copy)]
struct ResidueUnit<Index: Copy, DR: UnitMeasure + Copy> {
    pub idx: Index,
    pub value: DR,
}
impl<Index, DR> PartialOrd for ResidueUnit<Index, DR>
where
    Index: Copy,
    DR: UnitMeasure + Copy + PartialOrd,
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
impl<Index, DR> PartialEq for ResidueUnit<Index, DR>
where
    Index: Copy,
    DR: UnitMeasure + Copy + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
    #[allow(clippy::partialeq_ne_impl)] // 保持与内部浮点值一致的 NaN 语义：float 的 ne 不等价于 !eq
    fn ne(&self, other: &Self) -> bool {
        self.value.ne(&other.value)
    }
}
impl<Index, DR> Eq for ResidueUnit<Index, DR>
where
    Index: Copy,
    DR: UnitMeasure + Copy + Eq,
{
}
impl<Index, DR> Ord for ResidueUnit<Index, DR>
where
    Index: Copy,
    DR: UnitMeasure + Copy + Ord + PartialOrd + Eq,
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

//边权的单元表示
#[derive(Debug)]
pub struct EdgeWeightUnit<NodeIdx, EdgeIdx, D>
where
    D: UnitMeasure + Copy,
{
    pub target_node: NodeIdx,
    pub idx: EdgeIdx,
    pub value: D,
}
impl<NodeIdx, EdgeIdx, D> PartialOrd for EdgeWeightUnit<NodeIdx, EdgeIdx, D>
where
    D: UnitMeasure + Copy + PartialOrd,
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
impl<NodeIdx, EdgeIdx, D> PartialEq for EdgeWeightUnit<NodeIdx, EdgeIdx, D>
where
    D: UnitMeasure + Copy + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
    #[allow(clippy::partialeq_ne_impl)] // 同上：保留 NaN 下与 !eq 不同的 ne 语义
    fn ne(&self, other: &Self) -> bool {
        self.value.ne(&other.value)
    }
}
impl<NodeIdx, EdgeIdx, D> Eq for EdgeWeightUnit<NodeIdx, EdgeIdx, D> where D: UnitMeasure + Copy + Eq
{}

impl<NodeIdx, EdgeIdx, D> Ord for EdgeWeightUnit<NodeIdx, EdgeIdx, D>
where
    D: UnitMeasure + Copy + Ord + PartialOrd + Eq,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
    fn clamp(self, min: Self, max: Self) -> Self
    where
        Self: Sized,
    {
        Self {
            target_node: self.target_node,
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

type EdgeWeightCache<NodeId, EdgeId, D> = HashMap<NodeId, Vec<EdgeWeightUnit<NodeId, EdgeId, D>>>;

#[track_caller]
//TODO: make damping factor specific to each node
#[hotpath::measure]
pub fn weighted_ppr_fp<G, D, Q>(
    graph: G,
    damping_factor: D,
    personalized_vec: HashMap<G::NodeId, D>,
    residue_threshold: D,
    weight_calc: impl Fn(G, &G::EdgeRef, Option<&Q>) -> D,
    dynamic_query: Option<&Q>,
) -> Vec<(G::NodeId, D)>
where
    G: NodeCount + IntoEdges + NodeIndexable + IntoNodeIdentifiers,
    D: UnitMeasure + Copy + AddAssign + Ord,
    G::NodeId: Hash + Eq + Debug, //TODO: delete the Debug Trait bound
    G::EdgeId: Hash + Eq + Debug,
{
    //检查阻尼系数。damping == 1.0时残差无法转化为reserve，残差会被无限循环传播
    //（尤其是单节点无出度的情况），必须在进入循环前拒绝，防止DoS。
    assert!(
        D::zero() <= damping_factor && damping_factor < D::one(),
        "Damping factor should be in [0, 1)."
    );

    //归一化个性化向量
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
    //println!("source_node_count: {:?}", source_node_count);

    //初始化残差和保留
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
        .collect::<Vec<_>>();

    let mut ppr_edge_weight_cache: EdgeWeightCache<G::NodeId, G::EdgeId, D> =
        HashMap::with_capacity(graph.node_count());

    // 惰性优先队列替代"每轮全量扫描取最大残差"（find_max 原为 O(N)/轮，
    // profile 实测占 PPR 时间 89%~94%）：push 时入堆、陈旧条目弹出即弃、
    // 堆顶 ≤ 阈值即整体收敛（等价于原逻辑的 max ≤ 阈值 break）。
    let mut residue_heap: BinaryHeap<ResidueUnit<usize, D>> = residue_vec
        .iter()
        .copied()
        .filter(|u| u.value > D::zero())
        .collect();

    //每次取残差最大的节点进行push，加速收敛
    //迭代上限作为安全网：残差会随damping<1几何衰减，正常在有限次内收敛；
    //极小的residue_threshold或病态图可能使迭代次数过大，用上限兜底防止无限循环。
    let max_iterations = graph.node_bound().max(1) * 1024;
    let mut iteration_count = 0usize;
    // 每次取残差最大的节点进行push（热点测量：find_max 现为堆顶弹出 O(log N)）
    loop {
        let residue_i = hotpath::measure_block!("ppr::find_max", residue_heap.pop());
        let Some(residue_i) = residue_i else { break };
        if residue_i.value <= residue_threshold {
            // 堆顶 ≤ 阈值：所有剩余残差均 ≤ 阈值，整体收敛
            break;
        }
        // 陈旧条目：弹出值与当前残差不一致（该节点之后被再次更新过），弃之
        if residue_i.value != residue_vec[residue_i.idx].value {
            continue;
        }
        iteration_count += 1;
        if iteration_count > max_iterations {
            break;
        }
        //println!("Processing node {}", residue_i.idx);
        let out_edges = graph.edges(graph.from_index(residue_i.idx));
        //动态归一化的边权计算（懒缓存：每节点首次访问时计算一次）
        ppr_edge_weight_cache
            .entry(graph.from_index(residue_i.idx))
            .or_insert_with(|| {
                hotpath::measure_block!("ppr::edge_weight_calc", {
                    //println!("Calculating edge weights for node {}", residue_i.idx);
                    let weights = out_edges
                        .map(|edge| {
                            let weight = weight_calc(graph, &edge, dynamic_query);
                            EdgeWeightUnit {
                                target_node: edge.target(),
                                idx: edge.id(),
                                value: weight,
                            }
                        })
                        .collect::<Vec<_>>();
                    let sum = weights.iter().map(|v| v.value).sum::<D>();

                    if sum != D::zero() {
                        //防止NaN
                        weights
                            .into_iter()
                            .map(|w| EdgeWeightUnit {
                                target_node: w.target_node,
                                idx: w.idx,
                                value: w.value / sum,
                            })
                            .collect::<Vec<_>>()
                    } else {
                        weights
                    }
                })
            });

        let edge_weights = &ppr_edge_weight_cache[&graph.from_index(residue_i.idx)];
        //println!("edge_weights: {:?}", edge_weights);
        hotpath::measure_block!("ppr::push_spread", {
            //清空当前节点残差
            residue_vec[residue_i.idx].value = D::zero();

            //将部分残差转为保留
            reserve_vec[residue_i.idx] += (D::one() - damping_factor) * residue_i.value;

            //残差push（目标残差超阈值才入堆；堆顶≤阈值即收敛，故低于阈值者无需入堆）
            if let Some(edge_weight_max) = edge_weights.iter().max() {
                //节点出度不为0的情况
                if residue_i.value * edge_weight_max.value > residue_threshold {
                    edge_weights.iter().for_each(|edge_w| {
                        let idx = graph.to_index(edge_w.target_node);
                        let new_value = residue_vec[idx].value
                            + damping_factor * edge_w.value * residue_i.value;
                        residue_vec[idx].value = new_value;
                        if new_value > residue_threshold {
                            residue_heap.push(ResidueUnit {
                                idx,
                                value: new_value,
                            });
                        }
                    });
                }
                // else：该节点最大边权过小，本次push的贡献低于阈值；
                // 其残差已退役为保留值，不扩散（原 continue 语义）
            } else {
                //节点出度为0的情况
                if residue_i.value / source_node_count > residue_threshold {
                    normalized_personalized_vec.keys().for_each(|node| {
                        let idx = graph.to_index(*node);
                        let new_value = residue_vec[idx].value
                            + damping_factor * residue_i.value / source_node_count;
                        residue_vec[idx].value = new_value;
                        if new_value > residue_threshold {
                            residue_heap.push(ResidueUnit {
                                idx,
                                value: new_value,
                            });
                        }
                    });
                }
                // else：扩散量低于阈值，不扩散（原 continue 语义）
            }
        });
    }
    let sum = reserve_vec.iter().copied().sum::<D>();
    //sum为0时（如damping=1或全零边权）返回全零分布，避免0/0产生NaN
    if sum == D::zero() {
        return Vec::new();
    }

    graph
        .node_identifiers()
        .map(|node| {
            let ppr_value = reserve_vec[graph.to_index(node)] / sum;
            (node, ppr_value)
        })
        .collect()
}

#[cfg(test)]
mod test;
