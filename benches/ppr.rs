use criterion::{
    BenchmarkGroup, Criterion, SamplingMode, black_box, criterion_group, criterion_main,
};
use petgraph::{matrix_graph::NodeIndex, prelude::StableDiGraph, visit::NodeIndexable};
use soul_mem::utils::graph_algo::ppr::naive_ppr;
use std::collections::HashMap;

/// 创建压力测试图的辅助函数（在benchmark计时范围外使用）
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

/// 准备测试数据（在benchmark外执行）
fn prepare_test_data() -> (StableDiGraph<String, f64>, HashMap<NodeIndex<u32>, f64>) {
    let (graph, nodes) = pressure_large_graph();
    let mut source_bias = HashMap::new();
    nodes.iter().take(10).for_each(|idx| {
        source_bias.insert(*idx, graph.to_index(*idx) as f64);
    });
    (graph, source_bias)
}

/// 专门测试naive_ppr函数性能的benchmark
fn bench_naive_ppr_function(c: &mut Criterion) {
    // 准备测试数据（不计时）
    let (graph, source_bias) = prepare_test_data();

    // 设置采样次数为10
    let mut group = c.benchmark_group("naive_ppr_basic");
    group.sample_size(10); // 设置采样次数为10
    group.sampling_mode(SamplingMode::Flat);

    // Benchmark 1: 基础性能测试
    group.bench_function("basic", |b| {
        b.iter(|| {
            let result = naive_ppr(
                black_box(&graph),
                black_box(0.15_f64),
                black_box(source_bias.clone()),
                black_box(15),
            );
            black_box(result);
        });
    });

    // Benchmark 2: 测试不同阻尼因子
    group.bench_function("damping_high", |b| {
        b.iter(|| {
            let result = naive_ppr(
                black_box(&graph),
                black_box(0.85_f64), // 高阻尼因子
                black_box(source_bias.clone()),
                black_box(15),
            );
            black_box(result);
        });
    });

    // Benchmark 3: 测试不同迭代次数
    group.bench_function("iterations_50", |b| {
        b.iter(|| {
            let result = naive_ppr(
                black_box(&graph),
                black_box(0.15_f64),
                black_box(source_bias.clone()),
                black_box(50), // 更多迭代次数
            );
            black_box(result);
        });
    });

    // Benchmark 4: 测试更少的迭代次数
    group.bench_function("iterations_5", |b| {
        b.iter(|| {
            let result = naive_ppr(
                black_box(&graph),
                black_box(0.15_f64),
                black_box(source_bias.clone()),
                black_box(5), // 更少迭代次数
            );
            black_box(result);
        });
    });

    // Benchmark 5: 测试中等阻尼因子
    group.bench_function("damping_medium", |b| {
        b.iter(|| {
            let result = naive_ppr(
                black_box(&graph),
                black_box(0.5_f64), // 中等阻尼因子
                black_box(source_bias.clone()),
                black_box(15),
            );
            black_box(result);
        });
    });

    group.finish();
}

/// 参数化benchmark组，测试不同参数组合
fn bench_naive_ppr_parameterized(c: &mut Criterion) {
    let (graph, source_bias) = prepare_test_data();

    // 设置采样次数为10
    let mut group = c.benchmark_group("naive_ppr_parametrization");
    group.sample_size(10); // 设置采样次数为10
    group.sampling_mode(SamplingMode::Flat);

    // 测试不同迭代次数
    for iterations in [5, 10, 15, 20, 50].iter() {
        group.bench_function(format!("iterations_{}", iterations), |b| {
            b.iter(|| {
                let result = naive_ppr(
                    black_box(&graph),
                    black_box(0.15_f64),
                    black_box(source_bias.clone()),
                    black_box(*iterations),
                );
                black_box(result);
            });
        });
    }

    // 测试不同阻尼因子
    for damping in [0.1, 0.3, 0.5, 0.7, 0.9].iter() {
        group.bench_function(format!("damping_{}", damping), |b| {
            b.iter(|| {
                let result = naive_ppr(
                    black_box(&graph),
                    black_box(*damping as f64),
                    black_box(source_bias.clone()),
                    black_box(15),
                );
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_naive_ppr_function,
    bench_naive_ppr_parameterized
);
criterion_main!(benches);
