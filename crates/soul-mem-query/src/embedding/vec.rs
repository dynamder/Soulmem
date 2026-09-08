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
    pub fn cosine_similarity(&self, other: &Self) -> EmbeddingCalcResult<f32> {
        if self.shape() != other.shape() {
            return Err(super::EmbeddingCalcError::ShapeMismatch);
        }
        if self.0.iter().all(|&i| i == 0.0) || other.0.iter().all(|&i| i == 0.0) {
            return Ok(0.0);
        }
        let dot_product = self.dot(other)?;
        let norm_product = self.norm()? * other.norm()?;
        Ok(dot_product / norm_product)
    }
}
////////////////////////////////////////////////////////////////
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
