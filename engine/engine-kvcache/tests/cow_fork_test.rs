use engine_kvcache::CowBlockTable;

#[test]
fn test_cow_block_table_fork_and_mutate() {
    let mut parent = CowBlockTable::new();
    parent.append_block(10);
    parent.append_block(20);
    assert_eq!(parent.len(), 2);
    assert_eq!(parent.physical_block_ids(), vec![10, 20]);
    assert!(!parent.is_shared(0));
    assert!(!parent.is_shared(1));

    // Fork child branch (O(1) cloning)
    let mut child = parent.fork();
    assert_eq!(child.len(), 2);
    assert_eq!(child.physical_block_ids(), vec![10, 20]);

    // Both tables should report sharing for blocks 0 and 1
    assert!(parent.is_shared(0));
    assert!(parent.is_shared(1));
    assert!(child.is_shared(0));
    assert!(child.is_shared(1));

    // Tail needs CoW before write
    assert!(parent.tail_needs_cow());
    assert!(child.tail_needs_cow());

    // Perform Copy-on-Write on child tail (block 1 -> block 99)
    child.perform_tail_cow(99);

    assert_eq!(child.physical_block_ids(), vec![10, 99]);
    assert_eq!(parent.physical_block_ids(), vec![10, 20]);

    // Block 0 is still shared (prefix), but Block 1 is now independent
    assert!(parent.is_shared(0));
    assert!(!parent.is_shared(1));
    assert!(child.is_shared(0));
    assert!(!child.is_shared(1));

    // Append a brand new block to child
    child.append_block(100);
    assert_eq!(child.physical_block_ids(), vec![10, 99, 100]);
    assert_eq!(parent.physical_block_ids(), vec![10, 20]);
}
