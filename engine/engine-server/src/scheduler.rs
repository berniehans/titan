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

/// A client generation request queued in the continuous batching scheduler.
#[derive(Debug)]
pub struct GenerationJob {
    pub id: String,
    pub prompt: String,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stream_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    pub response_tx: Option<tokio::sync::oneshot::Sender<String>>,
}

/// State of an active continuous batch generation slot.
#[derive(Debug)]
pub struct ActiveBatchSlot {
    pub slot_id: usize,
    pub job_id: String,
    pub prompt_len: usize,
    pub current_token: u32,
    pub tokens_generated: usize,
    pub max_tokens: usize,
    pub context_tokens: Vec<u32>,
    pub generated_text: String,
    pub stream_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    pub response_tx: Option<tokio::sync::oneshot::Sender<String>>,
    pub is_finished: bool,
}

/// Dynamic multi-slot continuous batching manager.
pub struct ContinuousBatchManager {
    pub max_slots: usize,
    pub active_slots: Vec<ActiveBatchSlot>,
    pub pending_queue: std::collections::VecDeque<GenerationJob>,
    pub completed_count: usize,
}

impl ContinuousBatchManager {
    /// Creates a new continuous batching manager with the given concurrency limit (e.g. 4 or 8 slots).
    pub fn new(max_slots: usize) -> Self {
        Self {
            max_slots: max_slots.max(1),
            active_slots: Vec::with_capacity(max_slots),
            pending_queue: std::collections::VecDeque::new(),
            completed_count: 0,
        }
    }

    /// Enqueues a new generation job.
    pub fn submit(&mut self, job: GenerationJob) {
        self.pending_queue.push_back(job);
    }

    /// Number of available slots.
    pub fn free_slots(&self) -> usize {
        self.max_slots.saturating_sub(self.active_slots.len())
    }

    /// Pulls pending jobs into free active slots.
    /// Returns the newly admitted jobs to run prefill on.
    pub fn schedule_admissions(&mut self) -> Vec<usize> {
        let mut admitted_indices = Vec::new();
        while self.active_slots.len() < self.max_slots {
            if let Some(job) = self.pending_queue.pop_front() {
                let slot_id = self.find_free_slot_id();
                let initial_token = job.prompt_tokens.last().copied().unwrap_or(0);
                let prompt_len = job.prompt_tokens.len();
                let slot = ActiveBatchSlot {
                    slot_id,
                    job_id: job.id,
                    prompt_len,
                    current_token: initial_token,
                    tokens_generated: 0,
                    max_tokens: job.max_tokens,
                    context_tokens: job.prompt_tokens,
                    generated_text: String::new(),
                    stream_tx: job.stream_tx,
                    response_tx: job.response_tx,
                    is_finished: false,
                };
                self.active_slots.push(slot);
                admitted_indices.push(self.active_slots.len() - 1);
            } else {
                break;
            }
        }
        admitted_indices
    }

    /// Finds a free slot ID in 0..max_slots.
    fn find_free_slot_id(&self) -> usize {
        let used: std::collections::HashSet<usize> =
            self.active_slots.iter().map(|s| s.slot_id).collect();
        for id in 0..self.max_slots {
            if !used.contains(&id) {
                return id;
            }
        }
        self.active_slots.len()
    }

    /// Retires finished slots, notifying oneshot channels if present.
    pub fn retire_finished(&mut self) {
        let mut still_active = Vec::with_capacity(self.active_slots.len());
        for slot in self.active_slots.drain(..) {
            if slot.is_finished || slot.tokens_generated >= slot.max_tokens {
                if let Some(tx) = slot.response_tx {
                    let _ = tx.send(slot.generated_text);
                }
                self.completed_count += 1;
            } else {
                still_active.push(slot);
            }
        }
        self.active_slots = still_active;
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

    #[test]
    fn test_continuous_batch_manager_multi_slot_admissions_and_retirement() {
        let mut mgr = ContinuousBatchManager::new(2);
        assert_eq!(mgr.free_slots(), 2);

        mgr.submit(GenerationJob {
            id: "req-1".to_string(),
            prompt: "hello".to_string(),
            prompt_tokens: vec![1, 2, 3],
            max_tokens: 3,
            temperature: 0.0,
            stream_tx: None,
            response_tx: None,
        });
        mgr.submit(GenerationJob {
            id: "req-2".to_string(),
            prompt: "world".to_string(),
            prompt_tokens: vec![4, 5],
            max_tokens: 2,
            temperature: 0.0,
            stream_tx: None,
            response_tx: None,
        });
        mgr.submit(GenerationJob {
            id: "req-3".to_string(),
            prompt: "queue".to_string(),
            prompt_tokens: vec![6],
            max_tokens: 4,
            temperature: 0.0,
            stream_tx: None,
            response_tx: None,
        });

        let admitted = mgr.schedule_admissions();
        assert_eq!(admitted.len(), 2);
        assert_eq!(mgr.active_slots.len(), 2);
        assert_eq!(mgr.free_slots(), 0);
        assert_eq!(mgr.pending_queue.len(), 1);

        // Advance slot 0 and slot 1
        mgr.active_slots[0].tokens_generated = 3;
        mgr.active_slots[1].tokens_generated = 1;

        mgr.retire_finished();
        assert_eq!(mgr.active_slots.len(), 1);
        assert_eq!(mgr.completed_count, 1);
        assert_eq!(mgr.free_slots(), 1);

        // Admit pending job req-3
        let admitted2 = mgr.schedule_admissions();
        assert_eq!(admitted2.len(), 1);
        assert_eq!(mgr.active_slots.len(), 2);
        assert_eq!(mgr.pending_queue.len(), 0);
    }
}
