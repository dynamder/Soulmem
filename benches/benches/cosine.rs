//! cosine_similarity 微基准：旧 5 遍扫描 vs 标量融合 vs 公开 API（融合核心）。
//! 覆盖真实嵌入维度（BGE 512、qwen3 1024）与 128 对照。
//!
//! 旧实现逻辑（vec.rs 历史版本）：零检查×2 + dot + norm×2，语义照搬，仅去掉错误分支。
//! 标量融合为 fused_cosine_core 的切片内联副本（不含 API 分发/测量开销）；
//! 公开 API 走 `cosine_similarity`（同一融合核心）。结论见 bench 输出与 vec.rs 注释：
//! 真实维度下该循环为带宽受限，手动 AVX2 无额外收益，故实现保持纯标量可移植。
//! 运行：cargo bench -p soul-mem-benches --bench cosine

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use soul_mem_query::embedding::EmbeddingVec;

const DIMS: [usize; 3] = [128, 512, 1024];

fn make_vecs(dim: usize) -> (EmbeddingVec, EmbeddingVec, Vec<f32>, Vec<f32>) {
    // a 为块基（少量 1.0），b 为接近单位范数的确定性值
    let mut a = vec![0.0f32; dim];
    a[..3].fill(1.0);
    let b: Vec<f32> = (0..dim)
        .map(|i| ((i as f32 + 1.0) / dim as f32).sin())
        .collect();
    (
        EmbeddingVec::new(a.clone()),
        EmbeddingVec::new(b.clone()),
        a,
        b,
    )
}

/// 旧实现：零向量检查 ×2 + dot + norm×2（每次 norm 自带一次 dot 与 sqrt）。
fn cosine_old(a: &EmbeddingVec, b: &EmbeddingVec) -> f32 {
    if a.is_zero() || b.is_zero() {
        return 0.0;
    }
    let dot = a.dot(b).expect("shape ok");
    let norm_product = a.norm().expect("shape ok") * b.norm().expect("shape ok");
    dot / norm_product
}

/// 标量融合（fused_cosine_core 切片副本，避免测量走库内 hotpath guard）。
fn fused_slice(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut sa = 0.0f32;
    let mut sb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        sa += x * x;
        sb += y * y;
    }
    if sa == 0.0 || sb == 0.0 {
        0.0
    } else {
        dot / (sa * sb).sqrt()
    }
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine");
    group.sample_size(2000);
    for dim in DIMS {
        let (ea, eb, raw_a, raw_b) = make_vecs(dim);
        group.bench_function(BenchmarkId::new("old_5pass", dim), |bench| {
            bench.iter_batched(
                || (&ea, &eb),
                |(x, y)| std::hint::black_box(cosine_old(x, y)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function(BenchmarkId::new("scalar_fused", dim), |bench| {
            bench.iter_batched(
                || (&raw_a[..], &raw_b[..]),
                |(x, y)| std::hint::black_box(fused_slice(x, y)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function(BenchmarkId::new("fused_api", dim), |bench| {
            bench.iter_batched(
                || (&ea, &eb),
                |(x, y)| std::hint::black_box(x.cosine_similarity(y).expect("shape ok")),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cosine);
criterion_main!(benches);
