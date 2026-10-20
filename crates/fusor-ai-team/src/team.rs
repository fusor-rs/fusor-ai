use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Team {
    threads: usize,
    barrier: Barrier,
}

/// One thread's view of a team run.
pub struct Ctx<'team> {
    pub index: usize,
    pub count: usize,
    barrier: &'team Barrier,
}

impl Team {
    /// The team on Rayon's global pool (all of its threads), or the calling thread alone
    /// without the `threads` feature.
    pub fn global() -> Self {
        #[cfg(feature = "threads")]
        let threads = rayon::current_num_threads();
        #[cfg(not(feature = "threads"))]
        let threads = 1;
        Self {
            threads,
            barrier: Barrier::new(threads),
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs `task` on every team thread and returns when all have finished.
    pub fn run(&self, task: impl Fn(&Ctx) + Sync) {
        #[cfg(feature = "threads")]
        if self.threads > 1 {
            rayon::broadcast(|broadcast| {
                debug_assert_eq!(
                    broadcast.num_threads(),
                    self.threads,
                    "the global pool was resized"
                );
                task(&self.ctx(broadcast.index()));
            });
            return;
        }
        task(&self.ctx(0));
    }

    fn ctx(&self, index: usize) -> Ctx<'_> {
        Ctx {
            index,
            count: self.threads,
            barrier: &self.barrier,
        }
    }
}

impl Ctx<'_> {
    pub fn barrier(&self) {
        if self.count > 1 {
            self.barrier.wait();
        }
    }

    /// This thread's share of `0..total`, in whole units of `align` (`total` a multiple of
    /// `align`). The shares of all threads cover every unit exactly once.
    pub fn split(&self, total: usize, align: usize) -> Range<usize> {
        let units = total / align;
        let start = units * self.index / self.count;
        let end = units * (self.index + 1) / self.count;
        start * align..end * align
    }

    pub fn is_first(&self) -> bool {
        self.index == 0
    }
}

/// Spins before yielding: phases are microseconds apart, and a parked thread costs more to wake.
const SPINS_BEFORE_YIELD: u32 = 4096;

struct Barrier {
    threads: usize,
    arrived: AtomicUsize,
    generation: AtomicUsize,
}

impl Barrier {
    fn new(threads: usize) -> Self {
        Self {
            threads,
            arrived: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
        }
    }

    fn wait(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.threads {
            // Reset before releasing: no waiter can reach the next barrier before it sees the
            // new generation.
            self.arrived.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::Release);
            return;
        }
        let mut spins = 0;
        while self.generation.load(Ordering::Acquire) == generation {
            if spins < SPINS_BEFORE_YIELD {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Barrier, Ctx};

    #[test]
    fn split_covers_every_unit_once() {
        let barrier = Barrier::new(3);
        let mut covered = vec![0; 14 * 64];
        for index in 0..3 {
            let ctx = Ctx {
                index,
                count: 3,
                barrier: &barrier,
            };
            for column in ctx.split(14 * 64, 64) {
                covered[column] += 1;
            }
        }
        assert!(covered.iter().all(|&times| times == 1));
    }
}
