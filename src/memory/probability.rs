pub fn log_exp_sum2(a: f32, b: f32) -> f32 {
    //log(exp(a) + exp(b)) = max + log(exp(a-max) + exp(b-max))
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}
pub fn log_exp_sum(a: &[f32]) -> (f32,f32) {   //(sum, max)
    if a.is_empty() {
        return (f32::NEG_INFINITY, f32::NEG_INFINITY);
    }
    let max = a.iter().fold(f32::NEG_INFINITY, |acc, x| acc.max(*x));

    if max == f32::NEG_INFINITY {
        return (f32::NEG_INFINITY, f32::NEG_INFINITY);
    }

    (max + a.iter().map(|x| (x - max).exp()).sum::<f32>().ln(), max)
}