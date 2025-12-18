use core::iter::Sum;
use ordered_float::{FloatCore, OrderedFloat, PrimitiveFloat};
use petgraph::algo::UnitMeasure;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OrdFloat<F: ordered_float::FloatCore + PrimitiveFloat>(OrderedFloat<F>);
impl<F> Default for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + Default,
{
    fn default() -> Self {
        OrdFloat(OrderedFloat::default())
    }
}
impl<F> Sum for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + Sum,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        OrdFloat(OrderedFloat(iter.map(|f| f.0.0).sum()))
    }
}
impl<F> std::ops::Add for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + std::ops::Add,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        OrdFloat(self.0 + rhs.0)
    }
}
impl<F> std::ops::Sub for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + std::ops::Sub,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        OrdFloat(self.0 - rhs.0)
    }
}
impl<F> std::ops::Mul for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + std::ops::Mul,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        OrdFloat(self.0 * rhs.0)
    }
}
impl<F> std::ops::Div for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + std::ops::Div,
{
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        OrdFloat(self.0 / rhs.0)
    }
}

impl<F> UnitMeasure for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + Debug + Sum + Default,
{
    fn default_tol() -> Self {
        OrdFloat(OrderedFloat((F::epsilon())))
    }
    ///如果F不能从f32转化，则变为NaN而非panic
    fn from_f32(val: f32) -> Self {
        OrdFloat(OrderedFloat(F::from(val).unwrap_or(F::nan())))
    }
    ///如果F不能从f64转化，则变为NaN而非panic
    fn from_f64(val: f64) -> Self {
        OrdFloat(OrderedFloat(F::from(val).unwrap_or(F::nan())))
    }
    ///如果F不能从usize转化，则变为NaN而非panic
    fn from_usize(nb: usize) -> Self {
        OrdFloat(OrderedFloat(F::from(nb).unwrap_or(F::nan())))
    }
    fn one() -> Self {
        OrdFloat(OrderedFloat(F::one()))
    }
    fn zero() -> Self {
        OrdFloat(OrderedFloat(F::zero()))
    }
}
