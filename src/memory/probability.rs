use anyhow::{anyhow, Result};
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
pub fn online_temperature_softmax(a: &[f32], temperature: f32) -> Result<Vec<f32>> {
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(anyhow!("Invalid temperature: {temperature}"));
    }
    // 第一趟遍历：使用 fold 计算最大值 m 和归一化项 d
    let (m_final, d_final) = a.iter().fold(
        (f32::NEG_INFINITY, 0.0), // 初始状态 (m, d)
        |(m_prev, d_prev), &x_j| {
            // 应用 temperature 缩放
            let x_j_scaled = x_j / temperature;

            // 更新当前最大值
            let m_j = m_prev.max(x_j_scaled);

            // 计算指数项（避免重复计算）
            let exp_prev = (m_prev - m_j).exp();
            let exp_curr = (x_j_scaled - m_j).exp();

            // 更新归一化项
            let d_j = d_prev * exp_prev + exp_curr;

            (m_j, d_j)
        },
    );
    // 第二趟遍历：使用 map 计算最终概率
    Ok(a.iter()
        .map(|&x_i| {
            let x_i_scaled = x_i / temperature;
            (x_i_scaled - m_final).exp() / d_final
        })
        .collect())
}

pub fn data_softmax<T>(data: &[(T, f32)], temperature: f32) -> Result<Vec<(T, f32)>>
where
    T: Clone, 
{
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(anyhow!("Invalid temperature: {temperature}"));
    }

    let (max_val, denom) = data.iter().fold(
        (f32::NEG_INFINITY, 0.0),
        |(max_prev, denom_prev), (_, logit)| {
            let scaled = *logit / temperature;
            let max_curr = max_prev.max(scaled);
            let exp_prev = (max_prev - max_curr).exp();
            let exp_curr = (scaled - max_curr).exp();
            (max_curr, denom_prev * exp_prev + exp_curr)
        },
    );

    Ok(data.iter()
        .map(|(custom, logit)| {
            let scaled = *logit / temperature;
            let prob = (scaled - max_val).exp() / denom;
            (custom.clone(), prob)
        })
        .collect())
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn test_basic_functionality() {
        let logits = [1.0, 2.0, 3.0];
        let probs = online_temperature_softmax(&logits, 1.0).unwrap();

        // 手动计算期望值
        let max = 3.0;
        let exp_sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
        let expected: Vec<_> = logits.iter().map(|x| (x - max).exp() / exp_sum).collect();

        assert_abs_diff_eq!(probs.as_slice(), expected.as_slice(), epsilon = 1e-6);
    }
    #[test]
    fn test_temperature_effects() {
        let logits = [1.0, 2.0, 3.0];

        // 高温测试（概率分布更均匀）
        let probs_high = online_temperature_softmax(&logits, 2.0).unwrap();
        assert_abs_diff_eq!(probs_high[0], 0.1863237232, epsilon = 1e-4);
        assert_abs_diff_eq!(probs_high[1], 0.307_196, epsilon = 1e-4);
        assert_abs_diff_eq!(probs_high[2], 0.506_480, epsilon = 1e-4);

        // 低温测试（概率分布更尖锐）
        let probs_low = online_temperature_softmax(&logits, 0.5).unwrap();
        assert_abs_diff_eq!(probs_low[0], 0.015_876, epsilon = 1e-4);
        assert_abs_diff_eq!(probs_low[1], 0.117_310, epsilon = 1e-4);
        assert_abs_diff_eq!(probs_low[2], 0.866_814, epsilon = 1e-4);
    }
    #[test]
    fn test_extreme_temperature() {
        let logits = [1.0, 2.0, 3.0];

        // 趋近零温（接近 one-hot）
        let probs_zero = online_temperature_softmax(&logits, 1e-6).unwrap();
        assert_abs_diff_eq!(probs_zero[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(probs_zero[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(probs_zero[2], 1.0, epsilon = 1e-6);

        // 高温极限（均匀分布）
        let probs_inf = online_temperature_softmax(&logits, 1e6).unwrap();
        let uniform = 1.0 / logits.len() as f32;
        for prob in probs_inf {
            assert_abs_diff_eq!(prob, uniform, epsilon = 1e-6);
        }
    }
    #[test]
    fn test_numerical_stability() {
        // 大数值测试
        let large_logits = [1000.0, 1001.0, 1002.0];
        let probs = online_temperature_softmax(&large_logits, 1.0).unwrap();
        let sum: f32 = probs.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

        // 包含负无穷
        let logits_with_inf = [1.0, f32::NEG_INFINITY, 3.0];
        let probs_inf = online_temperature_softmax(&logits_with_inf, 1.0).unwrap();
        assert_abs_diff_eq!(probs_inf[0], 0.119_203, epsilon = 1e-4);
        assert_abs_diff_eq!(probs_inf[1], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(probs_inf[2], 0.880_797, epsilon = 1e-4);
    }
    #[test]
    #[should_panic(expected = "Invalid temperature")]
    fn test_invalid_temperature() {
        online_temperature_softmax(&[1.0, 2.0, 3.0], 0.0).unwrap();
    }
    
    #[derive(Debug,Copy, Clone)]
    struct CustomData {
        pub data: u32,
    }
    #[test]
    fn test_data_softmax() {
        let data = vec![(CustomData{data: 1},0.5), (CustomData{data: 2},0.6), (CustomData{data: 3},0.2)];
        let normalized = data_softmax(&data,1.0).unwrap();
        assert_eq!(normalized[0].0.data, 1);
        assert_eq!(normalized[1].0.data, 2);
        assert_eq!(normalized[2].0.data, 3);
        assert_abs_diff_eq!(normalized[0].1, 0.351371, epsilon=1e-5);
        assert_abs_diff_eq!(normalized[1].1, 0.388326, epsilon=1e-5);
        assert_abs_diff_eq!(normalized[2].1, 0.260303, epsilon=1e-5);
        
    }
    
}