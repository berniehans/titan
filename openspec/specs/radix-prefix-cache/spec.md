# radix-prefix-cache Delta Specification

## Purpose

Provides automatic prefix caching (APC) and zero-copy sequence branching using a token-level Radix Tree and Copy-on-Write block allocation in `engine-kvcache`.

## Requirements

### Requirement: Automatic Prefix Matching & Prefill Bypassing
The KV-cache subsystem SHALL maintain a Radix Tree of token sequences and their associated physical KV-cache blocks (`PhysicalBlockId`). Upon receiving a prompt, the engine SHALL perform a Longest Common Prefix (LCP) search and reuse existing physical blocks for the matched prefix.

#### Scenario: Prompt with identical system instructions
- **WHEN** an incoming request shares a $K$-token prefix with an existing cached sequence in the Radix Tree
- **THEN** the engine skips prefill computation for the first $K$ tokens, directly reuses the matched physical blocks, and sets the starting sequence position to $K$.

### Requirement: Pinned Protection for System Prompts & Tool Schemas
The prefix cache SHALL support pinning nodes (`is_pinned = true`) to prevent eviction of frequently accessed system prompts, system instructions, and tool declarations during memory reclamation.

#### Scenario: Memory pressure during high-throughput multi-agent execution
- **WHEN** the physical block pool exceeds the eviction watermark
- **THEN** the LRU eviction policy reclaims unpinned branch nodes and leaves pinned system nodes intact.

### Requirement: Zero-Copy Sequence Branching (Copy-on-Write)
The sequence table SHALL allow instant sequence cloning via atomic reference counting (`Arc<SharedBlock>`). Physical block duplication SHALL occur only when a branch appends tokens to a shared block (*Copy-on-Write*).

#### Scenario: Reasoning branch fork (Tree-of-Thoughts / Subagents)
- **WHEN** a sequence forks into multiple child exploration branches
- **THEN** all child branches initially share the same physical KV blocks, and a physical copy is performed only when a child writes new tokens to the shared tail block.
