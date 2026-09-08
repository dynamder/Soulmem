use std::ops::{Add, Div, Mul, Sub};

use serde::{Deserialize, Serialize};

use crate::embedding::EmbeddingCalcResult;

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize, Default)]
pub struct EmbeddingVec(Vec<f32>);

impl EmbeddingVec {
    pub fn shape(&self) -> usize {
        self.0.len()
    }
    pub fn new(vec: Vec<f32>) -> Self {
        Self(vec)
    }
    pub fn from_slice(slice: &[f32]) -> Self {
        Self(slice.to_vec())
    }
    pub fn zero(shape: usize) -> Self {
        Self(vec![0.0; shape])
    }
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.0.iter()
    }
    /// 是否为零向量（无有效嵌入输入时用零向量占位）。
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&i| i == 0.0)
    }
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }
}
impl IntoIterator for EmbeddingVec {
    type IntoIter = <Vec<f32> as IntoIterator>::IntoIter;
    type Item = f32;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl<A> FromIterator<A> for EmbeddingVec
where
    A: Into<f32>,
{
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        Self(iter.into_iter().map(|x| x.into()).collect())
    }
}
///////////////////////////////////////////////////////////////
impl Add for EmbeddingVec {
    type Output = EmbeddingCalcResult<Self>;
    fn add(self, rhs: Self) -> Self::Output {
        if self.shape() != rhs.shape() {
            return Err(super::EmbeddingCalcError::ShapeMismatch);
        }
        Ok(Self(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(a, b)| a + b)
                .collect(),
        ))
    }
}

impl Sub for EmbeddingVec {
    type Output = EmbeddingCalcResult<Self>;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape() != rhs.shape() {
            return Err(super::EmbeddingCalcError::ShapeMismatch);
        }
        Ok(Self(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(a, b)| a - b)
                .collect(),
        ))
    }
}

impl Mul<f32> for EmbeddingVec {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0.iter().map(|x| x * rhs).collect())
    }
}

impl Div<f32> for EmbeddingVec {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0.iter().map(|x| x / rhs).collect())
    }
}

////////////////////////////////////////////////////////////////////
impl EmbeddingVec {
    pub fn dot(&self, other: &Self) -> EmbeddingCalcResult<f32> {
        if self.shape() != other.shape() {
            return Err(super::EmbeddingCalcError::ShapeMismatch);
        }
        Ok(self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum())
    }
    pub fn norm(&self) -> EmbeddingCalcResult<f32> {
        self.dot(self).map(|x| x.sqrt())
    }
    pub fn normalize(&self) -> EmbeddingCalcResult<Self> {
        self.norm()
            .map(|norm| self.0.iter().map(|x| x / norm).collect())
    }
    pub fn euclidean_distance(&self, other: &Self) -> EmbeddingCalcResult<f32> {
        if self.shape() != other.shape() {
            return Err(super::EmbeddingCalcError::ShapeMismatch);
        }
        Ok(self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt())
    }
    /// 单遍融合余弦：一次循环同时累计 dot、双方平方和并隐式完成零向量检测，
    /// 替代原先"零检查×2 + dot + norm×2（各带 sqrt）"共 5 遍扫描。
    /// 数值等价（零向量 ⇔ 平方和为 0；分母 sqrt(na·nb) 与 sqrt(na)·sqrt(nb) 仅 ~1ulp 舍入差）。
    ///
    /// SIMD 策略（库不可假设宿主指令集）：纯标量单遍循环，由 LLVM 按各架构
    /// 基线自动向量化（x86_64→SSE2 4 宽 / aarch64→NEON），无 target_feature、
    /// 无运行时检测，产物天然可移植。实测（criterion，512/1024 维）：该循环为
    /// 内存带宽受限而非 FLOP 受限，额外 AVX2+FMA 256-bit 路径无进一步收益
    /// （<2%），故不引入 unsafe 特化，保持零条件分支的最简实现。
    #[hotpath::measure]
    pub fn cosine_similarity(&self, other: &Self) -> EmbeddingCalcResult<f32> {
        match fused_cosine_core(&self.0, &other.0) {
            Some((v, _)) => Ok(v),
            None => Err(super::EmbeddingCalcError::ShapeMismatch),
        }
    }

    /// 余弦相似度 + 零向量标记，单遍完成：调用方（如融合打分）常需区分
    /// "相似度恰为 0" 与 "tag 通道缺失（任一侧为零向量）"，此前需额外
    /// `is_zero()` 再各扫一遍全向量；本方法一次循环同时给出两者。
    #[hotpath::measure]
    pub fn cosine_similarity_and_zero(&self, other: &Self) -> EmbeddingCalcResult<(f32, bool)> {
        fused_cosine_core(&self.0, &other.0).ok_or(super::EmbeddingCalcError::ShapeMismatch)
    }
}
///////////////////////////////////////////////////////////////
/// 融合余弦核心：`None` 表示形状不匹配；返回 `(cosine, either_zero)`，
/// 其中 either_zero=true 表示任一侧平方和为 0（零向量占位），此时 cosine 为 0.0。
/// 单一可移植实现：LLVM 按架构基线自动向量化（x86_64 SSE2 / aarch64 NEON）。
#[inline(always)]
fn fused_cosine_core(a: &[f32], b: &[f32]) -> Option<(f32, bool)> {
    if a.len() != b.len() {
        return None;
    }
    let mut dot = 0.0f32;
    let mut sum_sq_a = 0.0f32;
    let mut sum_sq_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        sum_sq_a += x * x;
        sum_sq_b += y * y;
    }
    let either_zero = sum_sq_a == 0.0 || sum_sq_b == 0.0;
    if either_zero {
        return Some((0.0, true));
    }
    Some((dot / (sum_sq_a * sum_sq_b).sqrt(), false))
}

pub fn raw_linear_blend(
    vec1: &EmbeddingVec,
    vec2: &EmbeddingVec,
    blend_factor: f32,
) -> EmbeddingCalcResult<EmbeddingVec> {
    if vec1.shape() != vec2.shape() {
        return Err(super::EmbeddingCalcError::ShapeMismatch);
    }
    Ok(vec1
        .0
        .iter()
        .zip(vec2.0.iter())
        .map(|(&a, &b)| a * blend_factor + b * (1.0 - blend_factor))
        .collect())
}
pub fn mean_pooling(vecs: &[&EmbeddingVec]) -> EmbeddingCalcResult<EmbeddingVec> {
    if vecs.is_empty() {
        return Ok(EmbeddingVec::default());
    }
    let len = vecs[0].shape();
    if !vecs.iter().all(|vec| vec.shape() == len) {
        return Err(crate::embedding::EmbeddingCalcError::ShapeMismatch);
    }
    Ok(vecs
        .iter()
        .fold(vec![0.0; len], |acc, vec| {
            acc.iter().zip(vec.0.iter()).map(|(&a, &b)| a + b).collect()
        })
        .iter()
        .map(|&sum| sum / vecs.len() as f32)
        .collect())
}

/// 融合余弦正确性测试：结果必须与独立 oracle（旧公式，不同舍入顺序）一致，
/// 覆盖任意维度与数值形态；实现无架构特化，在任何 CPU/架构上结果一致。
#[cfg(test)]
mod cosine_simd_tests {
    use super::*;

    /// 独立 oracle：旧公式 dot / (norm·norm)（与融合循环不同的计算顺序与舍入）。
    fn oracle(a: &EmbeddingVec, b: &EmbeddingVec) -> f32 {
        if a.is_zero() || b.is_zero() {
            return 0.0;
        }
        let dot = a.dot(b).expect("shape ok");
        let np = a.norm().expect("shape ok") * b.norm().expect("shape ok");
        dot / np
    }

    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f32 / (1u64 << 20) as f32) - 1.0
    }

    #[test]
    fn cosine_dispatch_matches_oracle_across_dims() {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        for dim in [
            0usize, 1, 3, 7, 8, 16, 31, 32, 63, 64, 127, 128, 129, 256, 512,
        ] {
            let raw_a: Vec<f32> = (0..dim).map(|_| lcg(&mut s)).collect();
            let raw_b: Vec<f32> = (0..dim).map(|_| lcg(&mut s)).collect();
            let a = EmbeddingVec::new(raw_a);
            let b = EmbeddingVec::new(raw_b);
            let got = a.cosine_similarity(&b).expect("same dim");
            let want = oracle(&a, &b);
            if want.is_nan() {
                assert!(got.is_nan(), "dim {dim}: expected NaN");
            } else {
                assert!(
                    (got - want).abs() <= 1e-5,
                    "dim {dim}: dispatch {got} vs oracle {want}"
                );
            }
        }
    }

    #[test]
    fn cosine_known_values() {
        let x = EmbeddingVec::new(vec![1.0, 0.0, 0.0]);
        let y = EmbeddingVec::new(vec![0.0, 1.0, 0.0]);
        let z = EmbeddingVec::new(vec![1.0, 1.0, 0.0]);
        assert_eq!(x.cosine_similarity(&x).unwrap(), 1.0);
        assert_eq!(x.cosine_similarity(&y).unwrap(), 0.0);
        let r2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((x.cosine_similarity(&z).unwrap() - r2).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vectors() {
        let z = EmbeddingVec::zero(128);
        let v = EmbeddingVec::new(vec![1.0; 128]);
        assert_eq!(z.cosine_similarity(&v).unwrap(), 0.0);
        assert_eq!(z.cosine_similarity(&z).unwrap(), 0.0);
        assert_eq!(v.cosine_similarity(&z).unwrap(), 0.0);
    }

    #[test]
    fn cosine_shape_mismatch_is_err() {
        let a = EmbeddingVec::zero(128);
        let b = EmbeddingVec::zero(129);
        assert!(a.cosine_similarity(&b).is_err());
    }

    #[test]
    fn cosine_and_zero_flag() {
        let z = EmbeddingVec::zero(128);
        let v = EmbeddingVec::new(vec![1.0; 128]);
        // 任一侧零向量：score=0 且 zero 标记=true
        assert_eq!(z.cosine_similarity_and_zero(&v).unwrap(), (0.0, true));
        assert_eq!(z.cosine_similarity_and_zero(&z).unwrap(), (0.0, true));
        assert_eq!(v.cosine_similarity_and_zero(&z).unwrap(), (0.0, true));
        // 非零两侧：zero=false 且分数与 cosine_similarity 一致
        let w = EmbeddingVec::new(vec![1.0f32; 128]);
        let (s, zero) = v.cosine_similarity_and_zero(&w).unwrap();
        assert!(!zero);
        assert!((s - v.cosine_similarity(&w).unwrap()).abs() < 1e-6);
        // 形状不匹配仍为 Err
        assert!(
            v.cosine_similarity_and_zero(&EmbeddingVec::zero(129))
                .is_err()
        );
    }

    #[test]
    fn cosine_nan_propagates() {
        let mut a = vec![0.0f32; 64];
        a[3] = f32::NAN;
        let b = vec![1.0f32; 64];
        let got = EmbeddingVec::new(a)
            .cosine_similarity(&EmbeddingVec::new(b))
            .unwrap();
        assert!(got.is_nan());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EmbeddingCalcError;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "expected {a} close to {b}");
    }

    #[test]
    fn test_shape_and_accessors() {
        let v = EmbeddingVec::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.shape(), 3);
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
        assert_eq!(EmbeddingVec::from_slice(&[1.0, 2.0]).shape(), 2);
        assert_eq!(EmbeddingVec::zero(4).shape(), 4);
        assert!(EmbeddingVec::zero(4).iter().all(|x| *x == 0.0));
        assert_eq!(EmbeddingVec::default().shape(), 0);
    }

    #[test]
    fn test_into_iterator() {
        let v = EmbeddingVec::new(vec![1.0, 2.0]);
        let collected: Vec<f32> = v.into_iter().collect();
        assert_eq!(collected, vec![1.0, 2.0]);
    }

    #[test]
    fn test_from_iterator() {
        let v: EmbeddingVec = vec![1.0f32, 2.0, 3.0].into_iter().collect();
        assert_eq!(v.shape(), 3);
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_into_inner_returns_data() {
        // into_inner 是仓储层绑定 SurrealDB KNN 查询向量的通道（零拷贝取出内部 Vec<f32>），
        // 返回值必须就是内部数据（变异为 vec![] 时无测试拦截 = 覆盖缺口）。
        let a = EmbeddingVec::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(a.into_inner(), vec![1.0, 2.0, 3.0]);
        let zero = EmbeddingVec::zero(2);
        assert_eq!(zero.into_inner(), vec![0.0, 0.0]);
    }

    #[test]
    fn test_add() {
        let a = EmbeddingVec::new(vec![1.0, 2.0, 3.0]);
        let b = EmbeddingVec::new(vec![4.0, 5.0, 6.0]);
        let sum = (a + b).unwrap();
        assert_eq!(sum.iter().copied().collect::<Vec<_>>(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_shape_mismatch() {
        let a = EmbeddingVec::new(vec![1.0]);
        let b = EmbeddingVec::new(vec![1.0, 2.0]);
        assert!(matches!(a + b, Err(EmbeddingCalcError::ShapeMismatch)));
    }

    #[test]
    fn test_sub() {
        let a = EmbeddingVec::new(vec![5.0, 7.0, 9.0]);
        let b = EmbeddingVec::new(vec![1.0, 2.0, 3.0]);
        let diff = (a - b).unwrap();
        assert_eq!(
            diff.iter().copied().collect::<Vec<_>>(),
            vec![4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_sub_shape_mismatch() {
        let a = EmbeddingVec::new(vec![1.0]);
        let b = EmbeddingVec::new(vec![1.0, 2.0]);
        assert!(matches!(a - b, Err(EmbeddingCalcError::ShapeMismatch)));
    }

    #[test]
    fn test_mul_scalar() {
        let a = EmbeddingVec::new(vec![1.0, 2.0, 3.0]);
        let scaled = a * 2.0;
        assert_eq!(
            scaled.iter().copied().collect::<Vec<_>>(),
            vec![2.0, 4.0, 6.0]
        );
        let scaled2 = EmbeddingVec::new(vec![1.0, 2.0]) * 0.5;
        assert_eq!(scaled2.iter().copied().collect::<Vec<_>>(), vec![0.5, 1.0]);
    }

    #[test]
    fn test_div_scalar() {
        let a = EmbeddingVec::new(vec![2.0, 4.0, 6.0]);
        let divided = a / 2.0;
        assert_eq!(
            divided.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
        let divided2 = EmbeddingVec::new(vec![1.0, 2.0]) / 0.5;
        assert_eq!(divided2.iter().copied().collect::<Vec<_>>(), vec![2.0, 4.0]);
    }

    #[test]
    fn test_dot() {
        let a = EmbeddingVec::new(vec![1.0, 2.0, 3.0]);
        let b = EmbeddingVec::new(vec![4.0, 5.0, 6.0]);
        assert_close(a.dot(&b).unwrap(), 32.0, 1e-6);
    }

    #[test]
    fn test_dot_shape_mismatch() {
        let a = EmbeddingVec::new(vec![1.0]);
        let b = EmbeddingVec::new(vec![1.0, 2.0]);
        assert!(matches!(a.dot(&b), Err(EmbeddingCalcError::ShapeMismatch)));
    }

    #[test]
    fn test_norm() {
        let a = EmbeddingVec::new(vec![3.0, 4.0]);
        assert_close(a.norm().unwrap(), 5.0, 1e-6);
    }

    #[test]
    fn test_normalize() {
        let a = EmbeddingVec::new(vec![3.0, 4.0]);
        let n = a.normalize().unwrap();
        assert_close(n.iter().copied().next().unwrap(), 0.6, 1e-6);
        assert_close(n.iter().copied().nth(1).unwrap(), 0.8, 1e-6);
        // normalized vector has unit norm
        assert_close(n.norm().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = EmbeddingVec::new(vec![1.0, 2.0]);
        let b = EmbeddingVec::new(vec![4.0, 6.0]);
        assert_close(a.euclidean_distance(&b).unwrap(), 5.0, 1e-6);
    }

    #[test]
    fn test_euclidean_distance_shape_mismatch() {
        let a = EmbeddingVec::new(vec![1.0]);
        let b = EmbeddingVec::new(vec![1.0, 2.0]);
        assert!(matches!(
            a.euclidean_distance(&b),
            Err(EmbeddingCalcError::ShapeMismatch)
        ));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = EmbeddingVec::new(vec![1.0, 2.0, 3.0]);
        assert_close(a.cosine_similarity(&a).unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = EmbeddingVec::new(vec![1.0, 0.0]);
        let b = EmbeddingVec::new(vec![0.0, 1.0]);
        assert_close(a.cosine_similarity(&b).unwrap(), 0.0, 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = EmbeddingVec::new(vec![1.0, 2.0]);
        let b = EmbeddingVec::new(vec![-1.0, -2.0]);
        assert_close(a.cosine_similarity(&b).unwrap(), -1.0, 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let zero = EmbeddingVec::zero(3);
        let a = EmbeddingVec::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(zero.cosine_similarity(&a).unwrap(), 0.0);
        assert_eq!(a.cosine_similarity(&zero).unwrap(), 0.0);
    }

    #[test]
    fn test_cosine_similarity_shape_mismatch() {
        let a = EmbeddingVec::new(vec![1.0]);
        let b = EmbeddingVec::new(vec![1.0, 2.0]);
        assert!(matches!(
            a.cosine_similarity(&b),
            Err(EmbeddingCalcError::ShapeMismatch)
        ));
    }

    #[test]
    fn test_raw_linear_blend() {
        let a = EmbeddingVec::new(vec![1.0, 2.0]);
        let b = EmbeddingVec::new(vec![3.0, 4.0]);
        // blend_factor=0.5: 0.5*a + 0.5*b
        let blended = raw_linear_blend(&a, &b, 0.5).unwrap();
        assert_close(blended.iter().copied().next().unwrap(), 2.0, 1e-6);
        assert_close(blended.iter().copied().nth(1).unwrap(), 3.0, 1e-6);
        // blend_factor=0: pure b
        let pure_b = raw_linear_blend(&a, &b, 0.0).unwrap();
        assert_eq!(pure_b.iter().copied().collect::<Vec<_>>(), vec![3.0, 4.0]);
        // blend_factor=1: pure a
        let pure_a = raw_linear_blend(&a, &b, 1.0).unwrap();
        assert_eq!(pure_a.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_raw_linear_blend_shape_mismatch() {
        let a = EmbeddingVec::new(vec![1.0]);
        let b = EmbeddingVec::new(vec![1.0, 2.0]);
        assert!(matches!(
            raw_linear_blend(&a, &b, 0.5),
            Err(EmbeddingCalcError::ShapeMismatch)
        ));
    }

    #[test]
    fn test_mean_pooling() {
        let a = EmbeddingVec::new(vec![1.0, 2.0]);
        let b = EmbeddingVec::new(vec![3.0, 4.0]);
        let pooled = mean_pooling(&[&a, &b]).unwrap();
        assert_eq!(pooled.iter().copied().collect::<Vec<_>>(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_mean_pooling_empty() {
        let pooled = mean_pooling(&[]).unwrap();
        assert_eq!(pooled.shape(), 0);
    }

    #[test]
    fn test_mean_pooling_shape_mismatch() {
        let a = EmbeddingVec::new(vec![1.0]);
        let b = EmbeddingVec::new(vec![1.0, 2.0]);
        assert!(matches!(
            mean_pooling(&[&a, &b]),
            Err(EmbeddingCalcError::ShapeMismatch)
        ));
    }

    #[test]
    fn test_mean_pooling_single() {
        let a = EmbeddingVec::new(vec![5.0, 6.0]);
        let pooled = mean_pooling(&[&a]).unwrap();
        assert_eq!(pooled.iter().copied().collect::<Vec<_>>(), vec![5.0, 6.0]);
    }
}
