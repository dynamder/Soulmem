use std::ops::{Add, Div, Mul, Sub};

use crate::memory::embedding::EmbeddingCalcResult;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
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
}
impl Default for EmbeddingVec {
    fn default() -> Self {
        Self(Vec::new())
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
        return Err(crate::memory::embedding::EmbeddingCalcError::ShapeMismatch);
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
