//! cosine_similarity 微基准：单遍融合实现 vs 旧 5 遍扫描实现（128 维）。
//!
//! 旧实现逻辑（vec.rs @HEAD）：零检查×2 + dot + norm×2，语义照搬，仅去掉错误分支。
//! 运行：cargo bench -p soul-mem-benches --bench cosine

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use soul_mem_query::embedding::EmbeddingVec;

fn make_vecs() -> (EmbeddingVec, EmbeddingVec) {
    // 128 维确定性向量：a 为块基（少量 1.0），b 为接近单位范数的随机值
    let mut a = vec![0.0f32; 128];
    for d in 0..3 {
        a[d] = 1.0;
    }
    let b: Vec<f32> = (0..128).map(|i| ((i as f32 + 1.0) / 128.0).sin()).collect();
    (EmbeddingVec::new(a), EmbeddingVec::new(b))
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

fn bench_cosine(c: &mut Criterion) {
    let (a, b) = make_vecs();
    let mut group = c.benchmark_group("cosine_128d");
    group.sample_size(2000);
    group.bench_function(BenchmarkId::new("old_5pass", 128), |bench| {
        bench.iter_batched(
            || (&a, &b),
            |(x, y)| std::hint::black_box(cosine_old(x, y)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function(BenchmarkId::new("new_fused", 128), |bench| {
        bench.iter_batched(
            || (&a, &b),
            |(x, y)| std::hint::black_box(x.cosine_similarity(y).expect("shape ok")),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, bench_cosine);
criterion_main!(benches);
