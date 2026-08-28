use engine_kvcache::RadixTree;

#[test]
fn test_radix_tree_basic_match_and_insert() {
    let mut tree = RadixTree::new(16);

    // Initial query on empty tree
    let res = tree.match_prefix(&[1, 2, 3, 4]);
    assert_eq!(res.matched_tokens, 0);
    assert!(res.matched_blocks.is_empty());

    // Insert System Prompt tokens (e.g. 32 tokens -> 2 blocks)
    let system_tokens: Vec<u32> = (1..=32).collect();
    let system_blocks = vec![101, 102];
    tree.insert(&system_tokens, &system_blocks, true);

    // Query with exact system prompt
    let res = tree.match_prefix(&system_tokens);
    assert_eq!(res.matched_tokens, 32);
    assert_eq!(res.matched_blocks, vec![101, 102]);

    // Query with system prompt + user question (40 tokens)
    let mut user_req = system_tokens.clone();
    user_req.extend_from_slice(&[33, 34, 35, 36, 37, 38, 39, 40]);
    let res = tree.match_prefix(&user_req);
    assert_eq!(res.matched_tokens, 32);
    assert_eq!(res.matched_blocks, vec![101, 102]);

    // Insert the combined sequence
    let full_blocks = vec![101, 102, 103];
    tree.insert(&user_req, &full_blocks, false);

    // Query again with full sequence
    let res = tree.match_prefix(&user_req);
    assert_eq!(res.matched_tokens, 40);
    assert_eq!(res.matched_blocks, vec![101, 102, 103]);
}

#[test]
fn test_radix_tree_branching_and_split() {
    let mut tree = RadixTree::new(16);

    let prefix_common = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; // 1 block
    let mut branch_a = prefix_common.clone();
    branch_a.extend_from_slice(&[17, 18, 19, 20]); // Branch A suffix

    let mut branch_b = prefix_common.clone();
    branch_b.extend_from_slice(&[100, 101, 102, 103]); // Branch B suffix

    tree.insert(&branch_a, &[1, 2], false);
    tree.insert(&branch_b, &[1, 3], false);

    // Query Branch A
    let res_a = tree.match_prefix(&branch_a);
    assert_eq!(res_a.matched_tokens, 20);
    assert_eq!(res_a.matched_blocks, vec![1, 2]);

    // Query Branch B
    let res_b = tree.match_prefix(&branch_b);
    assert_eq!(res_b.matched_tokens, 20);
    assert_eq!(res_b.matched_blocks, vec![1, 3]);

    // Query prefix only
    let res_p = tree.match_prefix(&prefix_common);
    assert_eq!(res_p.matched_tokens, 16);
    assert_eq!(res_p.matched_blocks, vec![1]);
}

#[test]
fn test_radix_tree_lru_eviction_with_pinned_protection() {
    let mut tree = RadixTree::new(16);

    // Insert pinned system prompt
    let sys_tokens = vec![1, 2, 3, 4];
    tree.insert(&sys_tokens, &[999], true);

    // Insert unpinned session A
    let session_a = vec![1, 2, 3, 4, 10, 20];
    tree.insert(&session_a, &[999, 100], false);

    // Insert unpinned session B
    let session_b = vec![1, 2, 3, 4, 30, 40];
    tree.insert(&session_b, &[999, 200], false);

    // Evict 1 block
    let freed = tree.evict_lru(1);
    assert!(!freed.is_empty());
    // Pinned block 999 should NEVER be in freed blocks
    assert!(!freed.contains(&999));

    // System prompt prefix should still be matched intact
    let res = tree.match_prefix(&sys_tokens);
    assert_eq!(res.matched_tokens, 4);
    assert_eq!(res.matched_blocks, vec![999]);
}
