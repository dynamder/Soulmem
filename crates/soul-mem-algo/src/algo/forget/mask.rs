use jieba_rs::Jieba;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashSet;

/// 用于替换被遮罩词元的占位符
pub const MASK_WORD: &str = " [masked] ";

/// 遮罩操作的结果
pub struct MaskResult {
    /// 遮罩后的文本
    pub masked_text: String,
    /// 被遮罩的词数
    pub masked_count: usize,
    /// 总词数
    pub total_count: usize,
    /// 缺失度（输入参数）
    pub missing_degree: f32,
}

/// 对文本执行遮罩：基于缺失度 `missing_degree`（0.0~1.0）遮罩对应比例的词。
///
/// 遮罩是确定性的：同一文本 + 同一缺失度 → 同一遮罩结果。
/// 使用 jieba 分词后，随机选取 `n` 个词替换为 `[masked]`。
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

    // 确定性随机：文本 hash + 缺失度作为种子
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

    MaskResult {
        masked_text: masked.concat(),
        masked_count: n,
        total_count: total,
        missing_degree,
    }
}

/// 结合文本内容和缺失度生成确定性种子
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
