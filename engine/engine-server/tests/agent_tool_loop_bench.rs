use engine_core::grammar::JsonGrammar;
use engine_kvcache::RadixTree;
use std::time::Instant;

#[test]
fn test_agent_tool_loop_prefix_cache_and_json_validity() {
    let mut prefix_tree = RadixTree::new(16);

    // Turn 1: System Prompt (128 tokens) + Tool Definitions (128 tokens) = 256 tokens (16 blocks)
    let system_and_tools: Vec<u32> = (1..=256).collect();
    let system_blocks: Vec<u32> = (1..=16).collect();
    prefix_tree.insert(&system_and_tools, &system_blocks, true);

    // Simulate 5-turn multi-turn agent conversation
    let mut total_prefilled_tokens = 0;
    let mut total_bypassed_tokens = 0;

    for turn in 1..=5 {
        let mut turn_prompt = system_and_tools.clone();
        // Append unique user turn message
        let turn_user_tokens: Vec<u32> = (1000 * turn..1000 * turn + 32).collect();
        turn_prompt.extend_from_slice(&turn_user_tokens);

        let t_match_start = Instant::now();
        let match_res = prefix_tree.match_prefix(&turn_prompt);
        let match_dur = t_match_start.elapsed();

        assert_eq!(match_res.matched_tokens, 256, "System + Tools prefix should match 100%");
        assert_eq!(match_res.matched_blocks.len(), 16);
        assert!(match_dur.as_micros() < 500, "Radix LCP match should take < 0.5 ms (was {} us)", match_dur.as_micros());

        total_bypassed_tokens += match_res.matched_tokens;
        total_prefilled_tokens += turn_user_tokens.len();

        // Model generates a structured JSON tool invocation
        let mut grammar = JsonGrammar::new();
        let tool_response_tokens = [
            "{\n",
            "  \"tool\": \"fetch_weather\",\n",
            "  \"arguments\": {\"location\": \"San Francisco\"}\n",
            "}"
        ];

        for tok_str in &tool_response_tokens {
            assert!(grammar.is_token_valid(tok_str), "Valid JSON tool call token: {}", tok_str);
            grammar.advance(tok_str);
        }

        assert!(grammar.is_complete(), "Tool call must be fully accepted as valid JSON object");
    }

    println!("\n=== AGENT MULTI-TURN PREFIX CACHE SUMMARY ===");
    println!("Total Prefix Tokens Bypassed: {} tokens", total_bypassed_tokens);
    println!("Total New Tokens Prefilled:    {} tokens", total_prefilled_tokens);
    println!("Cache Reuse Rate:              {:.1}%", (total_bypassed_tokens as f64 / (total_bypassed_tokens + total_prefilled_tokens) as f64) * 100.0);
    println!("==============================================\n");
}
