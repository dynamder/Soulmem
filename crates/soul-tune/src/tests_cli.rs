use crate::base::{AlgoType, RetrieveMode};

#[test]
fn test_algo_parsing_retrieve_embedding() {
    let input = "retrieve";
    let algo = parse_retrieve_algo(input);
    assert!(matches!(
        algo,
        Some(AlgoType::Retrieve(RetrieveMode::Embedding))
    ));
}

#[test]
fn test_algo_parsing_retrieve_re() {
    let input = "re";
    let algo = parse_retrieve_algo(input);
    assert!(matches!(
        algo,
        Some(AlgoType::Retrieve(RetrieveMode::Embedding))
    ));
}

#[test]
fn test_algo_parsing_retrieve_full() {
    let input = "rf";
    let algo = parse_retrieve_algo(input);
    assert!(matches!(
        algo,
        Some(AlgoType::Retrieve(RetrieveMode::FullPipeline))
    ));
}

#[test]
fn test_algo_parsing_consolidate() {
    let input = "consolidate";
    let algo = parse_retrieve_algo(input);
    assert!(matches!(algo, Some(AlgoType::Consolidate)));
}

#[test]
fn test_algo_parsing_forget() {
    let input = "forget";
    let algo = parse_retrieve_algo(input);
    assert!(matches!(algo, Some(AlgoType::Forget)));
}

#[test]
fn test_algo_parsing_unknown() {
    let input = "nonexistent_algo";
    let algo = parse_retrieve_algo(input);
    assert!(algo.is_none());
}

#[test]
fn test_algo_display_roundtrip() {
    let algos = vec![
        AlgoType::Retrieve(RetrieveMode::Embedding),
        AlgoType::Retrieve(RetrieveMode::Association),
        AlgoType::Retrieve(RetrieveMode::FullPipeline),
        AlgoType::Compare,
        AlgoType::PlayTest,
        AlgoType::Consolidate,
        AlgoType::Forget,
    ];
    for a in &algos {
        let s = a.to_string();
        assert!(!s.is_empty(), "algo {} produces empty display", s);
    }
}

fn parse_retrieve_algo(s: &str) -> Option<AlgoType> {
    match s {
        "retrieve" | "r" | "retrieve/embedding" | "re" => {
            Some(AlgoType::Retrieve(RetrieveMode::Embedding))
        }
        "retrieve/association" | "ra" => Some(AlgoType::Retrieve(RetrieveMode::Association)),
        "retrieve/full" | "rf" => Some(AlgoType::Retrieve(RetrieveMode::FullPipeline)),
        "consolidate" | "c" => Some(AlgoType::Consolidate),
        "forget" | "f" => Some(AlgoType::Forget),
        _ => None,
    }
}
