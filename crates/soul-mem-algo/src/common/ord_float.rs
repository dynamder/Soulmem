use core::iter::Sum;
use ordered_float::{OrderedFloat, PrimitiveFloat};
use petgraph::algo::UnitMeasure;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct OrdFloat<F: ordered_float::FloatCore + PrimitiveFloat>(OrderedFloat<F>);
impl<F> OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat,
{
    pub fn into_inner(self) -> F {
        self.0.into_inner()
    }
}
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
impl<F> std::ops::AddAssign for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + std::ops::AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
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
impl<F> std::ops::SubAssign for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + std::ops::SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
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
impl<F> std::ops::MulAssign for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + std::ops::MulAssign,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
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
impl<F> std::ops::DivAssign for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + std::ops::DivAssign,
{
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}
impl<F> std::ops::Neg for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + std::ops::Neg,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        OrdFloat(-self.0)
    }
}
impl<F> UnitMeasure for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat + Debug + Sum + Default,
{
    fn default_tol() -> Self {
        OrdFloat(OrderedFloat(F::epsilon()))
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
impl<F> From<F> for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat,
{
    fn from(val: F) -> Self {
        OrdFloat(OrderedFloat(val))
    }
}
impl<F> Eq for OrdFloat<F> where F: ordered_float::FloatCore + PrimitiveFloat {}

impl<F> PartialOrd for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat,
{
    fn ge(&self, other: &Self) -> bool {
        self.0.ge(&other.0)
    }
    fn gt(&self, other: &Self) -> bool {
        self.0.gt(&other.0)
    }
    fn le(&self, other: &Self) -> bool {
        self.0.le(&other.0)
    }
    fn lt(&self, other: &Self) -> bool {
        self.0.lt(&other.0)
    }
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl<F> Ord for OrdFloat<F>
where
    F: ordered_float::FloatCore + PrimitiveFloat,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f64f(val: f64) -> OrdFloat<f64> {
        OrdFloat::from_f64(val)
    }

    #[test]
    fn test_ord_float_comparisons() {
        let a = f64f(1.0);
        let b = f64f(2.0);
        assert!(a < b);
        assert!(!(b < a));
        assert!(a <= b);
        assert!(!(b <= a));
        assert!(b > a);
        assert!(!(a > b));
        assert!(b >= a);
        assert!(!(a >= b));
        assert!(a != b);
        assert!(a == a);
    }

    #[test]
    fn test_ord_float_equal_values() {
        let a = f64f(0.5);
        let b = f64f(0.5);
        assert!(a == b);
        assert!(!(a < b));
        assert!(!(a > b));
        assert!(a <= b);
        assert!(a >= b);
    }

    #[test]
    fn test_ord_float_negative_and_zero() {
        let neg = f64f(-1.0);
        let zero = f64f(0.0);
        let pos = f64f(1.0);
        assert!(neg < zero);
        assert!(zero < pos);
        assert!(neg < pos);
        assert!(pos > neg);
    }

    #[test]
    fn test_ord_float_from_f64() {
        let v = f64f(3.14);
        assert_eq!(v.into_inner(), 3.14);
    }

    #[test]
    fn test_ord_float_arithmetic() {
        let a = f64f(10.0);
        let b = f64f(4.0);
        assert_eq!((a + b).into_inner(), 14.0);
        assert_eq!((a - b).into_inner(), 6.0);
        assert_eq!((a * b).into_inner(), 40.0);
        assert_eq!((a / b).into_inner(), 2.5);
        assert_eq!((-a).into_inner(), -10.0);
    }

    #[test]
    fn test_ord_float_assign_ops() {
        let mut a = f64f(10.0);
        let b = f64f(4.0);
        a += b;
        assert_eq!(a.into_inner(), 14.0);
        a -= b;
        assert_eq!(a.into_inner(), 10.0);
        a *= b;
        assert_eq!(a.into_inner(), 40.0);
        a /= b;
        assert_eq!(a.into_inner(), 10.0);
    }

    #[test]
    fn test_ord_float_unit_measure() {
        assert_eq!(OrdFloat::<f64>::zero().into_inner(), 0.0);
        assert_eq!(OrdFloat::<f64>::one().into_inner(), 1.0);
        assert_eq!(OrdFloat::<f64>::from_usize(7).into_inner(), 7.0);
        assert_eq!(OrdFloat::<f64>::from_f32(0.5).into_inner(), 0.5);
        // from_f64 遇到 NaN 输入应保持 NaN 而非 panic
        assert!(OrdFloat::<f64>::from_f64(f64::NAN).into_inner().is_nan());
        // default_tol 应为正的容差（不能为 0，否则除零/归一化失效）
        let tol = OrdFloat::<f64>::default_tol().into_inner();
        assert!(tol > 0.0 && tol.is_finite(), "default_tol should be positive and finite, got {tol}");
    }
}
