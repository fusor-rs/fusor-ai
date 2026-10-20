#![expect(
    unsafe_code,
    reason = "interior mutability shared across team threads under the barrier discipline"
)]

use std::cell::UnsafeCell;
use std::ops::Range;

use crate::Ctx;

/// An f32 buffer the team threads share under the barrier discipline (see the crate docs).
/// Every element is its own `UnsafeCell`, so no access ever borrows the whole buffer mutably.
pub struct SharedBuf {
    cells: Box<[UnsafeCell<f32>]>,
}

// SAFETY: concurrent access follows the crate's discipline (disjoint writers, no reader of an
// element while it is written between two barriers); the barrier orders the phases.
unsafe impl Sync for SharedBuf {}

impl SharedBuf {
    pub fn new(len: usize) -> Self {
        Self {
            cells: (0..len).map(|_| UnsafeCell::new(0.0)).collect(),
        }
    }

    /// All values. Only in phases where no thread writes this buffer.
    pub fn read(&self) -> &[f32] {
        // SAFETY: UnsafeCell<f32> has the layout of f32; by the discipline no element is
        // written while this phase reads it.
        unsafe { std::slice::from_raw_parts(self.cells.as_ptr().cast(), self.cells.len()) }
    }

    /// Reads one value. Only for elements no other thread writes in this phase.
    pub fn get(&self, index: usize) -> f32 {
        // SAFETY: by the discipline no thread writes this element in this phase.
        unsafe { *self.cells[index].get() }
    }

    /// Writes one value. Only for elements no other thread reads or writes in this phase.
    pub fn set(&self, index: usize, value: f32) {
        // SAFETY: by the discipline this thread is the element's only accessor in this phase.
        unsafe { *self.cells[index].get() = value }
    }

    /// A range this thread owns in this phase (disjoint from every other thread's access).
    #[expect(
        clippy::mut_from_ref,
        reason = "disjoint ranges per thread, see the crate docs"
    )]
    pub fn part(&self, range: Range<usize>) -> &mut [f32] {
        let cells = &self.cells[range];
        // SAFETY: UnsafeCell<f32> has the layout of f32 and grants interior mutability; by the
        // discipline the range is this thread's alone in this phase.
        unsafe { std::slice::from_raw_parts_mut(UnsafeCell::raw_get(cells.as_ptr()), cells.len()) }
    }

    /// Exclusive access outside team runs.
    pub fn get_mut(&mut self) -> &mut [f32] {
        let len = self.cells.len();
        // SAFETY: `&mut self` excludes every other access; the layout is that of f32.
        unsafe { std::slice::from_raw_parts_mut(UnsafeCell::raw_get(self.cells.as_ptr()), len) }
    }
}

/// One value per team thread (scratch buffers, private copies of a small state).
pub struct PerThread<T> {
    cells: Vec<UnsafeCell<T>>,
}

// SAFETY: thread `i` of a team run only touches cell `i` (`get` takes its `Ctx`), and team runs
// do not overlap.
unsafe impl<T: Send> Sync for PerThread<T> {}

impl<T> PerThread<T> {
    pub fn new(threads: usize, make: impl Fn() -> T) -> Self {
        Self {
            cells: (0..threads).map(|_| UnsafeCell::new(make())).collect(),
        }
    }

    #[expect(clippy::mut_from_ref, reason = "one cell per team thread index")]
    pub fn get(&self, ctx: &Ctx) -> &mut T {
        // SAFETY: each team thread has a distinct `ctx.index` during a run.
        unsafe { &mut *self.cells[ctx.index].get() }
    }
}

#[cfg(all(test, feature = "threads"))]
mod tests {
    use super::SharedBuf;
    use crate::Team;

    #[test]
    fn barrier_orders_phases() {
        let team = Team::global();
        let total = team.threads() * 16;
        let values = SharedBuf::new(total);
        let sums = SharedBuf::new(team.threads());
        team.run(|ctx| {
            for index in ctx.split(total, 16) {
                values.set(index, 1.0);
            }
            ctx.barrier();
            sums.set(ctx.index, values.read().iter().sum());
        });
        let expected = 16.0 * team.threads() as f32;
        assert!(sums.read().iter().all(|&sum| sum == expected));
    }
}
