use crate::common::ord_float::OrdFloat;
use petgraph::{matrix_graph::NodeIndex, prelude::StableDiGraph};

use super::*;
fn diff(actual: f64, expected: f64) -> f64 {
    if expected.abs() < f64::EPSILON && actual.abs() < f64::EPSILON {
        0.0
    } else {
        (actual - expected).abs()
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
type ToyGraph = StableDiGraph<String, f64>;
type BiasMap = HashMap<NodeIndex<u32>, f64>;
type IndexList = Vec<NodeIndex<u32>>;

fn toy_graph_with_init_a() -> (ToyGraph, BiasMap, IndexList) {
    let (graph, indexes) = test_toy_graph();
    let ans_vec: Vec<f64> = vec![0.851652742, 0.06387396045, 0.07345504972, 0.01101824785];
    let ans = indexes.iter().copied().zip(ans_vec).collect();
    (graph, ans, indexes)
}
fn toy_graph_with_init_b() -> (ToyGraph, BiasMap, IndexList) {
    let (graph, indexes) = test_toy_graph();
    let ans_vec: Vec<f64> = vec![0.0, 0.852878432, 0.1279320211, 0.01918954688];
    let ans = indexes.iter().copied().zip(ans_vec).collect();
    (graph, ans, indexes)
}
fn toy_graph_with_init_ab() -> (ToyGraph, BiasMap, IndexList) {
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
    assert!(
        (ans_sum - 1.0).abs() < 1e-6,
        "ppr sum should be ~1, got {ans_sum}"
    );

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
    assert!(
        (ans_sum - 1.0).abs() < 1e-6,
        "ppr sum should be ~1, got {ans_sum}"
    );

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
    assert!(
        (ans_sum - 1.0).abs() < 1e-6,
        "ppr sum should be ~1, got {ans_sum}"
    );

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
fn ppr_forward_push_toy_graph_init_a() {
    let (graph, true_ans, indexes) = toy_graph_with_init_a();
    let mut source_bias = HashMap::new();
    source_bias.insert(indexes[0], OrdFloat::from_f64(1.0));

    let ppr_ans = weighted_ppr_fp(
        &graph,
        OrdFloat::from_f64(0.15),
        source_bias,
        OrdFloat::from_f64(0.002),
        |_, _, _| OrdFloat::from_f64(1.0),
        Some(&"1"),
    );
    let ans_sum: f64 = ppr_ans
        .iter()
        .map(|(_, score)| score)
        .copied()
        .sum::<OrdFloat<f64>>()
        .into_inner();
    assert!(
        (ans_sum - 1.0).abs() < 1e-6,
        "ppr sum should be ~1, got {ans_sum}"
    );

    let ppr_ans = ppr_ans
        .into_iter()
        .collect::<HashMap<NodeIndex<u32>, OrdFloat<f64>>>();

    let avg_diff = 0.25
        * indexes
            .iter()
            .map(|idx| {
                let actual: f64 = ppr_ans[idx].into_inner();
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
fn ppr_forward_push_toy_graph_init_b() {
    let (graph, true_ans, indexes) = toy_graph_with_init_b();
    let mut source_bias = HashMap::new();
    source_bias.insert(indexes[1], OrdFloat::from_f64(1.0));

    let ppr_ans = weighted_ppr_fp(
        &graph,
        OrdFloat::from_f64(0.15),
        source_bias,
        OrdFloat::from_f64(0.002),
        |_, _, _| OrdFloat::from_f64(1.0),
        Some(&"1"),
    );
    let ans_sum: f64 = ppr_ans
        .iter()
        .map(|(_, score)| score)
        .copied()
        .sum::<OrdFloat<f64>>()
        .into_inner();

    let ppr_ans = ppr_ans
        .into_iter()
        .collect::<HashMap<NodeIndex<u32>, OrdFloat<f64>>>();
    assert!(ans_sum - 1.0 < 1e-5, "the sum is: {ans_sum}");

    let avg_diff = 0.25
        * indexes
            .iter()
            .map(|idx| {
                let actual: f64 = ppr_ans[idx].into_inner();
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
fn ppr_forward_push_weighted_hub_does_not_truncate() {
    // 带权图传播验证：
    //   枢纽节点S出度很大（边权归一化后max_weight≈0.005），其push贡献低于阈值；
    //   低度节点T只有单条高权边（max_weight=1.0），仍应继续传播到远端U。
    // 旧实现中 S 触发 `break` 会终止整个循环，导致 U 永远得不到残差（ppr(U)=0）；
    // 修复后 S 被 `continue` 跳过，T 正常传播，U 应获得正的PPR分数。
    let mut graph = StableDiGraph::new();
    let t = graph.add_node("T".to_string()); // 低度节点，idx 0
    let s = graph.add_node("S".to_string()); // 枢纽节点，idx 1（更高的idx，保证max()在等值时先处理S）
    let mut leaves = Vec::new();
    for i in 0..200 {
        leaves.push(graph.add_node(format!("L{i}")));
    }
    let u = graph.add_node("U".to_string()); // 远端节点
    for leaf in &leaves {
        graph.add_edge(s, *leaf, 1.0);
    }
    graph.add_edge(t, u, 1.0);

    let mut source_bias = HashMap::new();
    source_bias.insert(s, OrdFloat::from_f64(1.0));
    source_bias.insert(t, OrdFloat::from_f64(1.0));

    let ppr_ans = weighted_ppr_fp(
        &graph,
        OrdFloat::from_f64(0.65),
        source_bias,
        OrdFloat::from_f64(0.02),
        |_, _, _| OrdFloat::from_f64(1.0),
        Some(&"1"),
    );

    let u_ppr: f64 = ppr_ans
        .iter()
        .find(|(node, _)| *node == u)
        .map(|(_, score)| score.into_inner())
        .unwrap_or(0.0);
    assert!(
        u_ppr > 0.0,
        "far node U should receive PPR mass via low-degree node T, got {u_ppr}"
    );

    // 实际数量级验证：所有PPR分数有限且落在[0,1]，且总和≈1（概率分布）
    let sum: f64 = ppr_ans.iter().map(|(_, score)| score.into_inner()).sum();
    for (_, score) in &ppr_ans {
        let v = score.into_inner();
        assert!(v.is_finite(), "non-finite PPR score: {v}");
        assert!((0.0..=1.0).contains(&v), "PPR score out of [0,1]: {v}");
    }
    assert!((sum - 1.0).abs() < 1e-5, "PPR distribution sum {sum} != 1");
}

#[test]
fn ppr_forward_push_toy_graph_init_ab() {
    let (graph, true_ans, indexes) = toy_graph_with_init_ab();
    let mut source_bias = HashMap::new();
    source_bias.insert(indexes[0], OrdFloat::from_f64(1.0));
    source_bias.insert(indexes[1], OrdFloat::from_f64(1.0));

    let ppr_ans = weighted_ppr_fp(
        &graph,
        OrdFloat::from_f64(0.15),
        source_bias,
        OrdFloat::from_f64(0.002),
        |_, _, _| OrdFloat::from_f64(1.0),
        Some(&"1"),
    );
    let ans_sum: f64 = ppr_ans
        .iter()
        .map(|(_, score)| score)
        .copied()
        .sum::<OrdFloat<f64>>()
        .into_inner();
    assert!(
        (ans_sum - 1.0).abs() < 1e-6,
        "ppr sum should be ~1, got {ans_sum}"
    );

    let ppr_ans = ppr_ans
        .into_iter()
        .collect::<HashMap<NodeIndex<u32>, OrdFloat<f64>>>();
    assert!(ans_sum - 1.0 < 1e-5, "the sum is: {ans_sum}");

    let avg_diff = 0.25
        * indexes
            .iter()
            .map(|idx| {
                let actual: f64 = ppr_ans[idx].into_inner();
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
    assert!(ans_sum - 1.0 < 1e-5, "the sum is: {ans_sum}");
}
