use jieba_rs::Jieba;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashSet;

pub const MASK_WORD: &str = " [masked] ";

pub struct MaskResult {
    pub masked_text: String,
    pub masked_count: usize,
    pub total_count: usize,
    pub missing_degree: f32,
}

/// 对文本执行遮罩：基于 missing_degree (0.0~1.0) 缺失度遮罩对应比例的词
pub fn mask_text(text: &str, missing_degree: f32, jieba: &Jieba) -> MaskResult {
    let words: Vec<&str> = jieba.cut(text, true);
    let total = words.len();
    if total == 0 || missing_degree <= 0.0 {
        return MaskResult {
            masked_text: text.to_string(),
            masked_count: 0,
            total_count: total,
            missing_degree,
        };
    }
    let n = ((missing_degree * total as f32).round() as usize).min(total);
    if n == 0 {
        return MaskResult {
            masked_text: text.to_string(),
            masked_count: 0,
            total_count: total,
            missing_degree,
        };
    }
    let seed = text_seed(text, missing_degree);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..total).collect();
    indices.shuffle(&mut rng);
    let masked_set: HashSet<usize> = indices.iter().take(n).copied().collect();
    let masked: Vec<String> = words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            if masked_set.contains(&i) {
                MASK_WORD.to_string()
            } else {
                w.to_string()
            }
        })
        .collect();
    let result_text = masked.concat();
    // 确保结果中包含 MASK_WORD（当 n > 0 时）
    MaskResult {
        masked_text: result_text,
        masked_count: n,
        total_count: total,
        missing_degree,
    }
}

/// 确定性种子：结合文本内容和缺失度产生稳定输出
fn text_seed(text: &str, degree: f32) -> u64 {
    let hash = text
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
    hash ^ (degree.to_bits() as u64).wrapping_mul(114514)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_zero_degree() {
        let jieba = Jieba::new();
        let text = "我在北京吃烤鸭";
        let result = mask_text(text, 0.0, &jieba);
        assert_eq!(result.masked_text, text);
        assert_eq!(result.masked_count, 0);
    }

    #[test]
    fn test_mask_full_degree() {
        let jieba = Jieba::new();
        let text = "我在北京吃烤鸭";
        let result = mask_text(text, 1.0, &jieba);
        assert_eq!(result.masked_count, result.total_count);
        assert!(result.masked_text.contains(MASK_WORD.trim()));
    }

    #[test]
    fn test_deterministic_masking() {
        let jieba = Jieba::new();
        let text = "今天天气很好适合出去散步";
        let r1 = mask_text(text, 0.3, &jieba);
        let r2 = mask_text(text, 0.3, &jieba);
        assert_eq!(r1.masked_text, r2.masked_text);
    }

    #[test]
    fn test_empty_text() {
        let jieba = Jieba::new();
        let result = mask_text("", 0.5, &jieba);
        assert_eq!(result.masked_text, "");
        assert_eq!(result.masked_count, 0);
    }
}
