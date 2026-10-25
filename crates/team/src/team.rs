use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Team {
    threads: usize,
    barrier: Barrier,
}

/// One thread's view of a team run.
pub struct Ctx<'team> {
    pub(crate) index: usize,
    count: usize,
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
            barrier: Barrier::default(),
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs `task` on every team thread and returns when all have finished.
    pub fn run(&self, task: impl Fn(&Ctx) + Sync) {
        self.run_on(self.threads(), task);
    }

    /// Runs `task` on `threads` threads of the pool (the calling thread alone when `threads` is 1
    /// or without the `threads` feature), the pool's other threads receiving an empty job, and
    /// returns when all have finished.
    ///
    /// # Panics
    ///
    /// If `threads` is not between 1 and the team's threads.
    pub fn run_on(&self, threads: usize, task: impl Fn(&Ctx) + Sync) {
        assert!(
            (1..=self.threads()).contains(&threads),
            "a run takes 1 to {} threads, not {threads}",
            self.threads()
        );
        #[cfg(feature = "threads")]
        if threads > 1 {
            rayon::broadcast(|broadcast| {
                debug_assert_eq!(
                    broadcast.num_threads(),
                    self.threads(),
                    "the global pool was resized"
                );
                if broadcast.index() >= threads {
                    return;
                }
                task(&self.ctx(broadcast.index(), threads));
            });
            return;
        }
        task(&self.ctx(0, 1));
    }

    fn ctx(&self, index: usize, count: usize) -> Ctx<'_> {
        Ctx {
            index,
            count,
            barrier: &self.barrier,
        }
    }
}

impl Ctx<'_> {
    pub fn barrier(&self) {
        if self.count > 1 {
            self.barrier.wait(self.count);
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

#[derive(Default)]
struct Barrier {
    arrived: AtomicUsize,
    generation: AtomicUsize,
}

impl Barrier {
    fn wait(&self, threads: usize) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == threads {
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
    use std::sync::Mutex;

    use super::Team;

    #[test]
    fn split_covers_every_unit_once() {
        let team = Team::global();
        for count in [team.threads(), team.threads().div_ceil(2)] {
            let covered = Mutex::new(vec![0; 37 * 64]);
            team.run_on(count, |ctx| {
                let share = ctx.split(37 * 64, 64);
                let mut covered = covered.lock().unwrap();
                for column in share {
                    covered[column] += 1;
                }
            });
            assert!(
                covered
                    .into_inner()
                    .unwrap()
                    .iter()
                    .all(|&times| times == 1)
            );
        }
    }
}
