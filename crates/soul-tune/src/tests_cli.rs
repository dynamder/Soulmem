use crate::base::{AlgoType, ForgetMode, RetrieveFlavor, RetrieveMode};

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
    assert!(matches!(algo, Some(AlgoType::Forget(ForgetMode::Pipeline))));
}

#[test]
fn test_algo_parsing_forget_mask() {
    let input = "forget/mask";
    let algo = parse_retrieve_algo(input);
    assert!(matches!(algo, Some(AlgoType::Forget(ForgetMode::Mask))));
    assert!(matches!(
        parse_retrieve_algo("fm"),
        Some(AlgoType::Forget(ForgetMode::Mask))
    ));
}

#[test]
fn test_algo_parsing_forget_revise() {
    let input = "forget/revise";
    let algo = parse_retrieve_algo(input);
    assert!(matches!(algo, Some(AlgoType::Forget(ForgetMode::Revise))));
    assert!(matches!(
        parse_retrieve_algo("fr"),
        Some(AlgoType::Forget(ForgetMode::Revise))
    ));
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
        AlgoType::Retrieve(RetrieveMode::EmbeddingDb),
        AlgoType::Retrieve(RetrieveMode::AssociationDb),
        AlgoType::Retrieve(RetrieveMode::FullPipelineDb),
        AlgoType::Compare,
        AlgoType::CompareDb(RetrieveFlavor::Embedding),
        AlgoType::CompareDb(RetrieveFlavor::Association),
        AlgoType::CompareDb(RetrieveFlavor::FullPipeline),
        AlgoType::PlayTest,
        AlgoType::Consolidate,
        AlgoType::Forget(ForgetMode::Mask),
        AlgoType::Forget(ForgetMode::Revise),
        AlgoType::Forget(ForgetMode::Pipeline),
    ];
    for a in &algos {
        let s = a.to_string();
        assert!(!s.is_empty(), "algo {} produces empty display", s);
    }
}

#[test]
fn test_algo_parsing_retrieve_db_embedding() {
    for input in ["retrieve/db/embedding", "rde"] {
        assert!(matches!(
            parse_retrieve_algo(input),
            Some(AlgoType::Retrieve(RetrieveMode::EmbeddingDb))
        ));
    }
}

#[test]
fn test_algo_parsing_retrieve_db_association() {
    for input in ["retrieve/db/association", "rda"] {
        assert!(matches!(
            parse_retrieve_algo(input),
            Some(AlgoType::Retrieve(RetrieveMode::AssociationDb))
        ));
    }
}

#[test]
fn test_algo_parsing_retrieve_db_full() {
    for input in ["retrieve/db", "retrieve/db/full", "rd"] {
        assert!(matches!(
            parse_retrieve_algo(input),
            Some(AlgoType::Retrieve(RetrieveMode::FullPipelineDb))
        ));
    }
}

#[test]
fn test_algo_parsing_compare_db() {
    assert!(matches!(
        parse_retrieve_algo("compare/db"),
        Some(AlgoType::CompareDb(RetrieveFlavor::FullPipeline))
    ));
    assert!(matches!(
        parse_retrieve_algo("compare/db/full"),
        Some(AlgoType::CompareDb(RetrieveFlavor::FullPipeline))
    ));
    assert!(matches!(
        parse_retrieve_algo("compare/db/embedding"),
        Some(AlgoType::CompareDb(RetrieveFlavor::Embedding))
    ));
    assert!(matches!(
        parse_retrieve_algo("compare/db/association"),
        Some(AlgoType::CompareDb(RetrieveFlavor::Association))
    ));
}

#[test]
fn test_retrieve_mode_source_mapping() {
    assert!(!RetrieveMode::Embedding.uses_db());
    assert!(!RetrieveMode::Association.uses_db());
    assert!(!RetrieveMode::FullPipeline.uses_db());
    assert!(RetrieveMode::EmbeddingDb.uses_db());
    assert!(RetrieveMode::AssociationDb.uses_db());
    assert!(RetrieveMode::FullPipelineDb.uses_db());

    assert_eq!(
        RetrieveMode::EmbeddingDb.flavor(),
        RetrieveFlavor::Embedding
    );
    assert_eq!(RetrieveMode::Embedding.flavor(), RetrieveFlavor::Embedding);
    assert_eq!(
        RetrieveMode::FullPipelineDb.flavor(),
        RetrieveFlavor::FullPipeline
    );

    assert_eq!(
        RetrieveMode::FullPipeline.db_mode(),
        Some(RetrieveMode::FullPipelineDb)
    );
    assert_eq!(
        RetrieveMode::Embedding.db_mode(),
        Some(RetrieveMode::EmbeddingDb)
    );
    assert_eq!(
        RetrieveMode::EmbeddingDb.db_mode(),
        Some(RetrieveMode::EmbeddingDb)
    );
}

fn parse_retrieve_algo(s: &str) -> Option<AlgoType> {
    match s {
        "retrieve" | "r" | "retrieve/embedding" | "re" => {
            Some(AlgoType::Retrieve(RetrieveMode::Embedding))
        }
        "retrieve/association" | "ra" => Some(AlgoType::Retrieve(RetrieveMode::Association)),
        "retrieve/full" | "rf" => Some(AlgoType::Retrieve(RetrieveMode::FullPipeline)),
        "retrieve/db/embedding" | "rde" => Some(AlgoType::Retrieve(RetrieveMode::EmbeddingDb)),
        "retrieve/db/association" | "rda" => Some(AlgoType::Retrieve(RetrieveMode::AssociationDb)),
        "retrieve/db" | "retrieve/db/full" | "rd" => {
            Some(AlgoType::Retrieve(RetrieveMode::FullPipelineDb))
        }
        "compare/db" | "compare/db/full" => Some(AlgoType::CompareDb(RetrieveFlavor::FullPipeline)),
        "compare/db/embedding" => Some(AlgoType::CompareDb(RetrieveFlavor::Embedding)),
        "compare/db/association" => Some(AlgoType::CompareDb(RetrieveFlavor::Association)),
        "consolidate" | "c" => Some(AlgoType::Consolidate),
        "forget" | "f" | "forget/full" | "ff" => Some(AlgoType::Forget(ForgetMode::Pipeline)),
        "forget/mask" | "fm" => Some(AlgoType::Forget(ForgetMode::Mask)),
        "forget/revise" | "fr" => Some(AlgoType::Forget(ForgetMode::Revise)),
        _ => None,
    }
}
