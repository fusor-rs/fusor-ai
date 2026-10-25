#![expect(
    unsafe_code,
    reason = "interior mutability shared across team threads under the barrier discipline"
)]

use std::cell::UnsafeCell;
use std::ops::Range;

use crate::Ctx;

/// A buffer of plain values (f32 unless said otherwise) the team threads share under the
/// barrier discipline (see the crate docs). Every element is its own `UnsafeCell`, so no access
/// ever borrows the whole buffer mutably.
pub struct SharedBuf<T = f32> {
    cells: Box<[UnsafeCell<T>]>,
}

// SAFETY: concurrent access follows the crate's discipline (disjoint writers, no reader of an
// element while it is written between two barriers); the barrier orders the phases.
unsafe impl<T: Send> Sync for SharedBuf<T> {}

impl<T: Copy + Default> SharedBuf<T> {
    pub fn new(len: usize) -> Self {
        Self {
            cells: (0..len).map(|_| UnsafeCell::new(T::default())).collect(),
        }
    }

    /// All values. Only in phases where no thread writes this buffer.
    pub fn read(&self) -> &[T] {
        // SAFETY: UnsafeCell<T> has the layout of T; by the discipline no element is written
        // while this phase reads it.
        unsafe { std::slice::from_raw_parts(self.cells.as_ptr().cast(), self.cells.len()) }
    }

    /// Reads one value. Only for elements no other thread writes in this phase.
    pub fn get(&self, index: usize) -> T {
        // SAFETY: by the discipline no thread writes this element in this phase.
        unsafe { *self.cells[index].get() }
    }

    /// Writes one value. Only for elements no other thread reads or writes in this phase.
    pub fn set(&self, index: usize, value: T) {
        // SAFETY: by the discipline this thread is the element's only accessor in this phase.
        unsafe { *self.cells[index].get() = value }
    }

    /// A range this thread owns in this phase (disjoint from every other thread's access).
    #[expect(
        clippy::mut_from_ref,
        reason = "disjoint ranges per thread, see the crate docs"
    )]
    pub fn part(&self, range: Range<usize>) -> &mut [T] {
        let cells = &self.cells[range];
        // SAFETY: UnsafeCell<T> has the layout of T and grants interior mutability; by the
        // discipline the range is this thread's alone in this phase.
        unsafe { std::slice::from_raw_parts_mut(UnsafeCell::raw_get(cells.as_ptr()), cells.len()) }
    }

    /// Exclusive access outside team runs.
    pub fn get_mut(&mut self) -> &mut [T] {
        let len = self.cells.len();
        // SAFETY: `&mut self` excludes every other access; the layout is that of T.
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

    /// Thread `index`'s value, outside team runs.
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        self.cells[index].get_mut()
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
        for count in [team.threads(), team.threads().div_ceil(2)] {
            let values: SharedBuf = SharedBuf::new(total);
            let sums: SharedBuf = SharedBuf::new(count);
            team.run_on(count, |ctx| {
                for index in ctx.split(total, 16) {
                    values.set(index, 1.0);
                }
                ctx.barrier();
                sums.set(ctx.index, values.read().iter().sum());
            });
            let expected = total as f32;
            assert!(sums.read().iter().all(|&sum| sum == expected));
        }
    }
}
