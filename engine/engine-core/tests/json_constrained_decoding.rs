use engine_core::grammar::{GrammarParser, JsonObjectGrammar};

#[test]
fn test_json_object_grammar_state_transitions() {
    let mut grammar = JsonObjectGrammar::new();
    assert!(!grammar.is_accepted());

    // Token 1: "{"
    grammar.advance(1, "{\n").unwrap();
    assert!(!grammar.is_accepted());

    // Token 2: "  \"name\": \"Hermes\","
    grammar.advance(2, "  \"name\": \"Hermes\",\n").unwrap();
    assert!(!grammar.is_accepted());

    // Token 3: "  \"tool_call\": {\"fn\": \"web_search\"}\n"
    grammar.advance(3, "  \"tool_call\": {\"fn\": \"web_search\"}\n").unwrap();
    assert!(!grammar.is_accepted());

    // Token 4: "}" -> Closes root JSON object
    grammar.advance(4, "}").unwrap();
    assert!(grammar.is_accepted());

    // Token 5: Any attempt to append syntax after closing fails
    assert!(grammar.advance(5, "extra text").is_err());
}