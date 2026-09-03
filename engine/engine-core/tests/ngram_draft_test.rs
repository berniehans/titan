//! Context N-Gram Draft Proposer Tests (Phase 12, Sub-change 12.2).
//!
//! Validates:
//! 1. `NgramDraftProposer::propose` extracts candidate tokens on recurring phrases.
//! 2. Priority for longer matching n-grams over shorter ones.
//! 3. Empty return on non-matching or insufficient history.

use engine_core::ngram_draft::NgramDraftProposer;

#[test]
fn test_ngram_proposer_exact_phrase_match() {
    let proposer = NgramDraftProposer::new(3, 4, 2);

    // Sequence where "10, 20, 30" was previously followed by "40, 50, 60"
    let history = vec![1, 2, 10, 20, 30, 40, 50, 60, 99, 100, 10, 20, 30];

    let draft = proposer.propose(&history);
    assert_eq!(draft, vec![40, 50, 60]);
}

#[test]
fn test_ngram_proposer_recency_preference() {
    let proposer = NgramDraftProposer::new(2, 3, 2);

    // "10, 20" occurred twice:
    // First time followed by 30, 40
    // Second time followed by 80, 90
    let history = vec![1, 10, 20, 30, 40, 2, 10, 20, 80, 90, 3, 10, 20];

    let draft = proposer.propose(&history);
    // Most recent match is 80, 90
    assert_eq!(draft, vec![80, 90]);
}

#[test]
fn test_ngram_proposer_no_match() {
    let proposer = NgramDraftProposer::new(3, 4, 2);
    let history = vec![1, 2, 3, 4, 5, 6, 7];

    let draft = proposer.propose(&history);
    assert!(
        draft.is_empty(),
        "No recurring suffix should propose nothing"
    );
}

#[test]
fn test_ngram_proposer_short_history() {
    let proposer = NgramDraftProposer::new(3, 4, 2);
    let history = vec![1];

    let draft = proposer.propose(&history);
    assert!(
        draft.is_empty(),
        "Single token history should propose nothing"
    );
}
