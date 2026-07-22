#![allow(dead_code)]

use std::sync::LazyLock;

use nucleo::{
    pattern::{Atom, AtomKind, CaseMatching, Normalization},
    Config,
};
use parking_lot::Mutex;

pub static FUZZY_MATCHER: LazyLock<Mutex<nucleo::Matcher>> =
    LazyLock::new(|| Mutex::new(nucleo::Matcher::default()));

pub fn fuzzy_match<T: AsRef<str>>(
    pattern: Atom,
    items: impl IntoIterator<Item = T>,
    path_match: bool,
) -> Vec<(T, u16)> {
    let mut matcher = FUZZY_MATCHER.lock();
    matcher.config = Config::DEFAULT;
    if path_match {
        matcher.config.set_match_paths();
    }
    pattern.match_list(items, &mut matcher)
}

pub struct FuzzyPatternBuilder {
    case: CaseMatching,
    normalize: Normalization,
    kind: AtomKind,
    escape_whitespace: bool,
}

impl FuzzyPatternBuilder {
    pub fn case(mut self, case: CaseMatching) -> Self {
        self.case = case;
        self
    }
    pub fn normalize(mut self, normalize: Normalization) -> Self {
        self.normalize = normalize;
        self
    }
    pub fn kind(mut self, kind: AtomKind) -> Self {
        self.kind = kind;
        self
    }
    pub fn escape_whitespace(mut self, escape_whitespace: bool) -> Self {
        self.escape_whitespace = escape_whitespace;
        self
    }
    pub fn build(self, pattern: &str) -> Atom {
        Atom::new(
            pattern,
            self.case,
            self.normalize,
            self.kind,
            self.escape_whitespace,
        )
    }
}
impl Default for FuzzyPatternBuilder {
    fn default() -> Self {
        Self {
            case: CaseMatching::Smart,
            normalize: Normalization::Smart,
            kind: AtomKind::Fuzzy,
            escape_whitespace: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_match_exact() {
        let pattern = FuzzyPatternBuilder::default().build("test");
        let items = vec!["test", "other", "testing"];
        let results = fuzzy_match(pattern, items, false);
        assert!(!results.is_empty(), "should have matches");
        // exact match should score highest
        let first = &results[0];
        assert_eq!(first.0, "test");
    }

    #[test]
    fn test_fuzzy_match_partial() {
        let pattern = FuzzyPatternBuilder::default().build("tst");
        let items = vec!["test", "testing", "other"];
        let results = fuzzy_match(pattern, items, false);
        assert!(!results.is_empty(), "partial should also match");
    }

    #[test]
    fn test_fuzzy_match_no_match() {
        let pattern = FuzzyPatternBuilder::default().build("xyz123_never");
        let items = vec!["test", "other"];
        let results = fuzzy_match(pattern, items, false);
        assert!(results.is_empty(), "no match for unrelated pattern");
    }

    #[test]
    fn test_pattern_builder_default() {
        let b = FuzzyPatternBuilder::default();
        let atom = b.build("query");
        let items = vec!["query", "other"];
        let results = fuzzy_match(atom, items, false);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "query");
    }
}
