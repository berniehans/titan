/// Continuous-batching scheduler over the paged KV pool.
///
/// The scheduler owns the [`PagedKvCache`] and multiplexes many
/// [`GenerationSession`]s over it. On every [`BatchScheduler::advance`] each
/// still-active session steps forward exactly one token together; sessions that
/// have exhausted their budget are retired and their KV blocks are returned to
/// the pool, so one finished session never stalls the rest (no head-of-line
/// blocking).
use crate::session::GenerationSession;
use engine_kvcache::{KvCacheError, PagedKvCache, PagedKvCacheConfig};

/// Tracks a set of concurrent generation sessions over one KV pool.
pub struct BatchScheduler {
    cache: PagedKvCache,
    active: Vec<GenerationSession>,
}

impl BatchScheduler {
    /// Creates a scheduler over a fresh blocked KV pool.
    pub fn new(cfg: PagedKvCacheConfig) -> Result<Self, KvCacheError> {
        Ok(Self {
            cache: PagedKvCache::new(cfg)?,
            active: Vec::with_capacity(8),
        })
    }

    /// Starts a new generation session and returns its index.
    ///
    /// `vocab` bounds the stub token ids produced; `prompt_token` is the first
    /// token appended to the session's KV sequence; `max_tokens` is the
    /// completion budget.
    pub fn add(
        &mut self,
        vocab: u32,
        prompt_token: u32,
        max_tokens: u32,
    ) -> Result<usize, KvCacheError> {
        let seq = self.cache.new_sequence();
        let session =
            GenerationSession::begin(&mut self.cache, seq, vocab, prompt_token, max_tokens)?;
        self.active.push(session);
        Ok(self.active.len() - 1)
    }

    /// Advances every active session by exactly one token.
    ///
    /// Returns the tokens produced by each stepped session this round.
    /// Sessions that hit their budget are removed (and their KV blocks freed),
    /// so a finished session does not block the others.
    pub fn advance(&mut self) -> Vec<u32> {
        let mut produced = Vec::<u32>::new();
        let mut next_round = Vec::with_capacity(self.active.len());
        for mut session in self.active.drain(..) {
            if session.is_finished() {
                // Already exhausted: retire without stepping.
                self.cache.free_sequence(session.seq);
                continue;
            }
            if let Some(token) = session.step(&mut self.cache) {
                produced.push(token);
            }
            if session.is_finished() {
                self.cache.free_sequence(session.seq);
            } else {
                next_round.push(session);
            }
        }
        self.active = next_round;
        produced
    }

    /// Cancels an active session immediately (client drop) and frees its KV
    /// blocks back to the pool.
    pub fn cancel(&mut self, index: usize) {
        if index >= self.active.len() {
            return;
        }
        let session = self.active.remove(index);
        self.cache.free_sequence(session.seq);
    }

    /// Number of sessions currently being advanced.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Whether `index` currently identifies an active session.
    pub fn is_active(&self, index: usize) -> bool {
        index < self.active.len()
    }

    /// Reference to the active session at `index`, if any.
    pub fn session(&self, index: usize) -> Option<&GenerationSession> {
        self.active.get(index)
    }

    /// Physical KV blocks currently held by active sessions.
    pub fn blocks_used(&self) -> usize {
        self.cache.blocks_used()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler() -> BatchScheduler {
        BatchScheduler::new(PagedKvCacheConfig {
            n_blocks: 16,
            block_tokens: 128,
            heads: 1,
            head_dim: 1,
        })
        .expect("scheduler")
    }

    #[test]
    fn all_sessions_advance_and_finished_exit_without_blocking() {
        let mut s = scheduler();
        let _a = s.add(1000, 3, 2).expect("session a"); // finishes after 2 steps
        let _b = s.add(1000, 7, 5).expect("session b");
        assert_eq!(s.active_count(), 2);

        // Record how many sessions produced tokens each round.
        let mut rounds = Vec::new();
        while s.active_count() > 0 {
            rounds.push(s.advance().len());
        }

        // Round 1: both step. Round 2: both step (A finishes after this).
        // Rounds 3-5: only B. No head-of-line blocking.
        assert_eq!(rounds, vec![2, 2, 1, 1, 1]);
        assert_eq!(s.active_count(), 0);

        // All retired sessions returned their KV blocks to the pool.
        assert_eq!(s.blocks_used(), 0, "retired sessions must free KV blocks");
    }

    #[test]
    fn cancel_frees_blocks_and_returns_to_baseline() {
        let mut s = scheduler();
        let idx = s.add(1000, 11, 64).expect("session");
        let baseline = s.blocks_used();
        assert!(baseline > 0, "an active session must hold KV blocks");

        // Simulate a client dropping the SSE stream mid-generation.
        s.cancel(idx);
        assert_eq!(s.active_count(), 0);
        assert_eq!(
            s.blocks_used(),
            0,
            "cancelling a session must free its KV blocks"
        );
    }
}
