//! cosine 融合计算 SIMD 对照实验：验证"Rust 无 fast-math → 普通 += 归约不向量化"
//! 的推断，比较四种实现（真实嵌入维度 512/1024）：
//!   a) single      —— 现 fused_cosine_core 单累加器（标量，推测未被向量化）
//!   b) unroll8     —— 手动 8 路标量累加器（纯 ILP，零依赖）
//!   c) wide        —— wide crate f32x8 lane 累加（stable 可移植 SIMD）
//!   d) avx2_fma    —— std::arch AVX2+FMA 内联（x86_64 专用，运行时特性检测）
//!
//! 运行：cargo bench -p soul-mem-benches --bench cosine_simd

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

const DIMS: [usize; 2] = [512, 1024];

fn make_vecs(dim: usize) -> (Vec<f32>, Vec<f32>) {
    // 确定性数据：a 稀疏块基、b 连续值，保证数值非退化
    let mut a = vec![0.0f32; dim];
    for d in 0..3 {
        a[d] = 1.0;
    }
    let b: Vec<f32> = (0..dim)
        .map(|i| ((i as f32 + 1.0) / dim as f32).sin())
        .collect();
    (a, b)
}

/// a) 现实现：单累加器 fused 循环。
fn cosine_single(a: &[f32], b: &[f32]) -> f32 {
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

/// b) 手动 8 路标量累加器：ILP（不依赖任何 SIMD 指令/库）。
fn cosine_unroll8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut i = 0usize;
    let mut d = [0.0f32; 8];
    let mut sa = [0.0f32; 8];
    let mut sb = [0.0f32; 8];
    while i + 8 <= n {
        for k in 0..8 {
            let x = a[i + k];
            let y = b[i + k];
            d[k] += x * y;
            sa[k] += x * x;
            sb[k] += y * y;
        }
        i += 8;
    }
    let mut dv = 0.0f32;
    let mut sav = 0.0f32;
    let mut sbv = 0.0f32;
    for k in 0..8 {
        dv += d[k];
        sav += sa[k];
        sbv += sb[k];
    }
    for x in a[i..].iter().zip(&b[i..]) {
        dv += x.0 * x.1;
        sav += x.0 * x.0;
        sbv += x.1 * x.1;
    }
    if sav == 0.0 || sbv == 0.0 {
        0.0
    } else {
        dv / (sav * sbv).sqrt()
    }
}

/// c) wide crate：f32x8 lane 累加（stable 可移植 SIMD）。
fn cosine_wide(a: &[f32], b: &[f32]) -> f32 {
    use wide::f32x8;
    let n = a.len();
    let mut i = 0usize;
    let mut d = f32x8::splat(0.0);
    let mut sa = f32x8::splat(0.0);
    let mut sb = f32x8::splat(0.0);
    while i + 8 <= n {
        let mut aa = [0.0f32; 8];
        let mut bb = [0.0f32; 8];
        aa.copy_from_slice(&a[i..i + 8]);
        bb.copy_from_slice(&b[i..i + 8]);
        let va = f32x8::from(aa);
        let vb = f32x8::from(bb);
        d += va * vb;
        sa += va * va;
        sb += vb * vb;
        i += 8;
    }
    let mut dv = d.reduce_add();
    let mut sav = sa.reduce_add();
    let mut sbv = sb.reduce_add();
    for (x, y) in a[i..].iter().zip(&b[i..]) {
        dv += x * y;
        sav += x * x;
        sbv += y * y;
    }
    if sav == 0.0 || sbv == 0.0 {
        0.0
    } else {
        dv / (sav * sbv).sqrt()
    }
}

/// d) AVX2+FMA 内联（x86_64 专用；bench 外调用需运行时特性检测，此处由 bench 环境保证）。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut i = 0usize;
    let mut d = _mm256_setzero_ps();
    let mut sa = _mm256_setzero_ps();
    let mut sb = _mm256_setzero_ps();
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        d = _mm256_fmadd_ps(va, vb, d);
        sa = _mm256_fmadd_ps(va, va, sa);
        sb = _mm256_fmadd_ps(vb, vb, sb);
        i += 8;
    }
    let mut dv = 0.0f32;
    let mut sav = 0.0f32;
    let mut sbv = 0.0f32;
    let mut lane = [0.0f32; 8];
    _mm256_storeu_ps(lane.as_mut_ptr(), d);
    for v in lane {
        dv += v;
    }
    _mm256_storeu_ps(lane.as_mut_ptr(), sa);
    for v in lane {
        sav += v;
    }
    _mm256_storeu_ps(lane.as_mut_ptr(), sb);
    for v in lane {
        sbv += v;
    }
    for (x, y) in a[i..].iter().zip(&b[i..]) {
        dv += x * y;
        sav += x * x;
        sbv += y * y;
    }
    if sav == 0.0 || sbv == 0.0 {
        0.0
    } else {
        dv / (sav * sbv).sqrt()
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn cosine_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    cosine_single(a, b)
}

/// d) AVX2+FMA 的**安全包装**：仅当运行 CPU 支持时进入内联路径，否则回退单累加器。
/// （若集成进库，将以此形态做运行时派发。）
#[cfg(target_arch = "x86_64")]
fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: 已运行时确认 CPU 支持 avx2+fma
        unsafe { cosine_avx2_fma(a, b) }
    } else {
        cosine_single(a, b)
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    cosine_single(a, b)
}

/// 各实现一致性冒烟（数值容差内一致 + 零向量返回 0），bench 启动时自检一次。
fn sanity_check(a: &[f32], b: &[f32]) {
    let refs = cosine_single(a, b);
    for (name, got) in [
        ("unroll8", cosine_unroll8(a, b)),
        ("wide", cosine_wide(a, b)),
        ("avx2_fma", cosine_avx2(a, b)),
    ] {
        assert!(
            (got - refs).abs() <= 1e-4,
            "mismatch {name}: got {got}, single {refs}"
        );
    }
    let z = vec![0.0f32; a.len()];
    assert_eq!(cosine_wide(&z, a), 0.0);
    assert_eq!(cosine_unroll8(&z, a), 0.0);
    assert_eq!(cosine_avx2(&z, a), 0.0);
}

fn bench_cosine_simd(c: &mut Criterion) {
    for dim in DIMS {
        let (a, b) = make_vecs(dim);
        sanity_check(&a, &b);
        let mut group = c.benchmark_group(format!("cosine_fused_{dim}d"));
        group.sample_size(2000);
        group.bench_function(BenchmarkId::new("single", dim), |bench| {
            bench.iter_batched(
                || (&a[..], &b[..]),
                |(x, y)| std::hint::black_box(cosine_single(x, y)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function(BenchmarkId::new("unroll8", dim), |bench| {
            bench.iter_batched(
                || (&a[..], &b[..]),
                |(x, y)| std::hint::black_box(cosine_unroll8(x, y)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function(BenchmarkId::new("wide_f32x8", dim), |bench| {
            bench.iter_batched(
                || (&a[..], &b[..]),
                |(x, y)| std::hint::black_box(cosine_wide(x, y)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function(BenchmarkId::new("avx2_fma", dim), |bench| {
            bench.iter_batched(
                || (&a[..], &b[..]),
                |(x, y)| std::hint::black_box(cosine_avx2(x, y)),
                BatchSize::SmallInput,
            )
        });
        group.finish();
    }
}

criterion_group!(benches, bench_cosine_simd);
criterion_main!(benches);
