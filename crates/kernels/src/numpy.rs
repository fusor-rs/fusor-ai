//! NumPy's reductions as the reference computes them, so that float32 results agree bit for
//! bit: sums pairwise (`pairwise_sum` in `loops_utils.h`: eight accumulators over blocks of up
//! to 128 values, halves above), means as that sum over the count in the same precision, and
//! medians as the middle value or the mean of the two middle ones.

use std::cmp::Ordering;
use std::ops::{Add, Div};

/// Values under this are summed one after another.
const UNROLL: usize = 8;
/// Values up to this are summed by `UNROLL` accumulators; more are split in halves.
const BLOCK: usize = 128;

pub trait Float: Copy + Add<Output = Self> + Div<Output = Self> {
    const ZERO: Self;

    fn of_count(count: usize) -> Self;

    fn total_order(&self, other: &Self) -> Ordering;
}

impl Float for f32 {
    const ZERO: Self = 0.0;

    fn of_count(count: usize) -> Self {
        count as f32
    }

    fn total_order(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl Float for f64 {
    const ZERO: Self = 0.0;

    fn of_count(count: usize) -> Self {
        count as f64
    }

    fn total_order(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

/// `np.sum`.
pub fn sum<T: Float>(values: &[T]) -> T {
    let count = values.len();
    if count < UNROLL {
        return values.iter().fold(T::ZERO, |sum, &value| sum + value);
    }
    if count > BLOCK {
        let half = count / 2;
        let half = half - half % UNROLL;
        return sum(&values[..half]) + sum(&values[half..]);
    }
    let (blocks, rest) = values.as_chunks::<UNROLL>();
    let mut lanes = blocks[0];
    for block in &blocks[1..] {
        for (lane, value) in lanes.iter_mut().zip(block) {
            *lane = *lane + *value;
        }
    }
    let [l0, l1, l2, l3, l4, l5, l6, l7] = lanes;
    let total = ((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7));
    rest.iter().fold(total, |sum, &value| sum + value)
}

/// `np.mean` of a non-empty array.
pub fn mean<T: Float>(values: &[T]) -> T {
    sum(values) / T::of_count(values.len())
}

/// `np.median` of a non-empty array.
pub fn median<T: Float>(values: &[T]) -> T {
    let mut sorted = values.to_vec();
    sorted.sort_by(T::total_order);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        return sorted[middle];
    }
    mean(&sorted[middle - 1..=middle])
}

/// `np.argmax`: the index of the first of the largest values.
pub fn argmax<T: PartialOrd>(values: &[T]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}
