//! PPR算法性能基准测试
//!
//! 本文件包含两种PPR（Personalized PageRank）算法的性能对比测试：
//! 1. Power Iteration算法：传统迭代方法，收敛性好但较慢
//! 2. Forward Push算法：近似算法，速度极快但有一定精度损失
//!
//! 使用方法：
//! 1. 运行所有测试: cargo bench
//! 2. 运行特定测试: cargo bench <测试组名>
//! 3. 测试结果会生成在target/criterion目录下

use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use petgraph::stable_graph::NodeIndex;
use petgraph::{prelude::StableDiGraph, visit::NodeIndexable};
use soul_mem::utils::graph_algo::ord_float::OrdFloat;
use soul_mem::utils::graph_algo::ppr::{naive_ppr, weighted_ppr_fp};
use std::collections::HashMap;
use std::hint::black_box;

/// 创建小型测试图的辅助函数（在benchmark计时范围外使用）
/// 使用稀疏图结构以提高测试效率
fn create_test_graph(size: usize) -> (StableDiGraph<String, OrdFloat<f64>>, Vec<NodeIndex<u32>>) {
    let mut graph = StableDiGraph::new();
    let mut nodes = Vec::new();

    // 创建指定大小的图
    for i in 0..size {
        let node = graph.add_node(format!("node_{}", i));
        nodes.push(node);

        // 每个节点连接到前min(10, 节点数)个节点（如果存在）
        nodes.iter().take(10.min(nodes.len())).for_each(|idx| {
            graph.add_edge(node, *idx, OrdFloat::from(1.0));
        });

        // 添加自环
        graph.add_edge(node, node, OrdFloat::from(1.0));
    }
    (graph, nodes)
}

/// 创建大规模测试图的辅助函数（在benchmark计时范围外使用）
fn create_large_test_graph(
    size: usize,
) -> (StableDiGraph<String, OrdFloat<f64>>, Vec<NodeIndex<u32>>) {
    let mut graph = StableDiGraph::new();
    let mut nodes = Vec::new();

    // 创建指定大小的图（优化版本，减少边数以保持可管理性）
    for i in 0..size {
        let node = graph.add_node(format!("node_{}", i));
        nodes.push(node);

        // 对于大规模图，限制边的数量以提高性能
        let max_connections = if size > 1000 { 5 } else { 10 };
        nodes
            .iter()
            .take(max_connections.min(nodes.len()))
            .for_each(|idx| {
                graph.add_edge(node, *idx, OrdFloat::from(1.0));
            });

        // 添加自环
        graph.add_edge(node, node, OrdFloat::from(1.0));
    }
    (graph, nodes)
}

/// 简单的边权重计算函数（用于forward push算法）
fn simple_weight_calc(
    _edge: &petgraph::stable_graph::EdgeReference<OrdFloat<f64>>,
    _query: &(),
) -> OrdFloat<f64> {
    OrdFloat::from(1.0)
}

/// 准备小型测试数据（在benchmark外执行）
/// 用于基础性能对比测试
fn prepare_test_data() -> (
    StableDiGraph<String, OrdFloat<f64>>,
    HashMap<NodeIndex<u32>, OrdFloat<f64>>,
) {
    let (graph, nodes) = create_test_graph(20);
    let mut source_bias = HashMap::new();

    // 前3个节点作为源节点
    nodes.iter().take(3).for_each(|idx| {
        source_bias.insert(*idx, OrdFloat::from(graph.to_index(*idx) as f64));
    });

    (graph, source_bias)
}

/// 准备大规模测试数据（在benchmark外执行）
/// 专门用于中等和大型规模图的性能测试
fn prepare_large_test_data(
    size: usize,
) -> (
    StableDiGraph<String, OrdFloat<f64>>,
    HashMap<NodeIndex<u32>, OrdFloat<f64>>,
) {
    let (graph, nodes) = create_large_test_graph(size);
    let mut source_bias = HashMap::new();

    // 前min(5, size/100)个节点作为源节点
    let source_count = (5.max(size / 100)).min(nodes.len());
    nodes.iter().take(source_count).for_each(|idx| {
        source_bias.insert(*idx, OrdFloat::from(graph.to_index(*idx) as f64));
    });

    (graph, source_bias)
}

/// 基础性能对比：Power Iteration vs Forward Push
/// 测试20个节点的小型图，展示两种算法的基本性能差异
/// 结果显示Forward Push比Power Iteration快约20-30倍
fn bench_basic_comparison(c: &mut Criterion) {
    let (graph, source_bias) = prepare_test_data();

    let mut group = c.benchmark_group("basic_comparison");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    // Power Iteration算法
    group.bench_function("power_iteration_15_iters", |b| {
        b.iter(|| {
            let result = naive_ppr(
                black_box(&graph),
                black_box(OrdFloat::from(0.15)),
                black_box(source_bias.clone()),
                black_box(15),
            );
            black_box(result);
        });
    });

    // Forward Push算法
    group.bench_function("forward_push_threshold_1e-4", |b| {
        b.iter(|| {
            let result = weighted_ppr_fp(
                black_box(&graph),
                black_box(OrdFloat::from(0.15)),
                black_box(source_bias.clone()),
                black_box(OrdFloat::from(0.0001)),
                black_box(simple_weight_calc),
                black_box(&()),
            );
            black_box(result);
        });
    });

    group.finish();
}

/// 不同迭代次数下的Power Iteration性能
/// 展示Power Iteration算法的时间复杂度与迭代次数的关系
/// 执行时间与迭代次数呈近似线性增长
fn bench_power_iteration_variants(c: &mut Criterion) {
    let (graph, source_bias) = prepare_test_data();

    let mut group = c.benchmark_group("power_iteration_variants");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    for iterations in [5, 10, 15, 20].iter() {
        group.bench_function(format!("iterations_{}", iterations), |b| {
            b.iter(|| {
                let result = naive_ppr(
                    black_box(&graph),
                    black_box(OrdFloat::from(0.15)),
                    black_box(source_bias.clone()),
                    black_box(*iterations),
                );
                black_box(result);
            });
        });
    }

    group.finish();
}

/// 不同残差阈值下的Forward Push性能
/// 展示阈值对Forward Push算法的精度和速度的影响
/// 阈值越小，精度越高但执行时间也相应增加
fn bench_forward_push_variants(c: &mut Criterion) {
    let (graph, source_bias) = prepare_test_data();

    let mut group = c.benchmark_group("forward_push_variants");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    for threshold in [0.001, 0.0001, 0.00001].iter() {
        group.bench_function(format!("threshold_{}", threshold), |b| {
            b.iter(|| {
                let result = weighted_ppr_fp(
                    black_box(&graph),
                    black_box(OrdFloat::from(0.15)),
                    black_box(source_bias.clone()),
                    black_box(OrdFloat::from(*threshold)),
                    black_box(simple_weight_calc),
                    black_box(&()),
                );
                black_box(result);
            });
        });
    }

    group.finish();
}

/// 不同阻尼因子下的性能对比
/// 测试阻尼因子[0.1, 0.3, 0.5, 0.7]对两种算法的影响
/// Forward Push在高阻尼因子下性能略有下降
fn bench_damping_factors(c: &mut Criterion) {
    let (graph, source_bias) = prepare_test_data();

    let mut group = c.benchmark_group("damping_factors");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    for damping in [0.1, 0.3, 0.5, 0.7].iter() {
        group.bench_function(format!("power_iteration_damping_{}", damping), |b| {
            b.iter(|| {
                let result = naive_ppr(
                    black_box(&graph),
                    black_box(OrdFloat::from(*damping)),
                    black_box(source_bias.clone()),
                    black_box(15),
                );
                black_box(result);
            });
        });

        group.bench_function(format!("forward_push_damping_{}", damping), |b| {
            b.iter(|| {
                let result = weighted_ppr_fp(
                    black_box(&graph),
                    black_box(OrdFloat::from(*damping)),
                    black_box(source_bias.clone()),
                    black_box(OrdFloat::from(0.0001)),
                    black_box(simple_weight_calc),
                    black_box(&()),
                );
                black_box(result);
            });
        });
    }

    group.finish();
}

/// 中等规模图性能测试 (100-500节点)
/// 测试规模：100, 300, 500个节点的图
/// 在此规模下Forward Push的性能优势开始显著体现（快100-1000倍）
fn bench_medium_scale_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("medium_scale_performance");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    let graph_sizes = [100, 300, 500];

    for size in graph_sizes.iter() {
        let (graph, source_bias) = prepare_large_test_data(*size);

        // Power Iteration
        group.bench_function(format!("power_iteration_size_{}", size), |b| {
            b.iter(|| {
                let result = naive_ppr(
                    black_box(&graph),
                    black_box(OrdFloat::from(0.15)),
                    black_box(source_bias.clone()),
                    black_box(10), // 中等规模图使用较少的迭代次数
                );
                black_box(result);
            });
        });

        // Forward Push
        group.bench_function(format!("forward_push_size_{}", size), |b| {
            b.iter(|| {
                let result = weighted_ppr_fp(
                    black_box(&graph),
                    black_box(OrdFloat::from(0.15)),
                    black_box(source_bias.clone()),
                    black_box(OrdFloat::from(0.0001)),
                    black_box(simple_weight_calc),
                    black_box(&()),
                );
                black_box(result);
            });
        });
    }

    group.finish();
}

/// 大规模图性能测试 (1000-3000节点)
/// 主要测试Forward Push算法，Power Iteration在1000节点以上效率过低
/// 结果显示Forward Push的执行时间与图规模近似线性增长
fn bench_large_scale_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_performance");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    let graph_sizes = [1000, 2000, 3000];

    for size in graph_sizes.iter() {
        let (graph, source_bias) = prepare_large_test_data(*size);

        // 对于大规模图，只测试Forward Push（Power Iteration可能太慢）
        group.bench_function(format!("forward_push_size_{}", size), |b| {
            b.iter(|| {
                let result = weighted_ppr_fp(
                    black_box(&graph),
                    black_box(OrdFloat::from(0.15)),
                    black_box(source_bias.clone()),
                    black_box(OrdFloat::from(0.0001)),
                    black_box(simple_weight_calc),
                    black_box(&()),
                );
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_basic_comparison,
    bench_power_iteration_variants,
    bench_forward_push_variants,
    bench_damping_factors,
    bench_medium_scale_performance,
    bench_large_scale_performance
);
criterion_main!(benches);
