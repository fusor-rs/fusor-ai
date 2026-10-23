//! Reading order: the order network's line features, the order decoded from its pairwise
//! logits (a ranking, then insertion moves), layout blocks clustered from its same-block logits
//! (average linkage), and XY-cut, which orders lines inside a block and every line of a page the
//! network does not read.

use kernels::numpy::{argmax, median, sum};

use crate::crops::LineBox;
use crate::layout::{CLASSES, is_grid};
use crate::order::FEATURES;
use crate::text::{display_width, is_digit};

/// Sweeps of insertion moves at most, and the gain a move must bring.
const SWEEPS: usize = 4;
const MIN_GAIN: f32 = 1e-6;
const SENTENCE_END: &str = ".!?:;。！？：；．";
const ITEM_MARKS: &str = "•-–▪◦*·→";
/// The smallest width and height a box counts with.
const MIN_SIDE: f32 = 1e-3;
const ROW_SLACK: f32 = 1e-3;

/// The order network's features of each line: its box relative to the page, its height
/// relative to the page's median, its aspect, the page's median height, and features of its
/// text.
pub fn features(
    boxes: &[LineBox],
    (page_width, page_height): (usize, usize),
    texts: &[String],
) -> Vec<f32> {
    let (width, height) = (page_width as f32, page_height as f32);
    let widths: Vec<f32> = boxes
        .iter()
        .map(|line| (line[2] - line[0]).max(MIN_SIDE))
        .collect();
    let heights: Vec<f32> = boxes
        .iter()
        .map(|line| (line[3] - line[1]).max(MIN_SIDE))
        .collect();
    let median_height = median(&heights);
    let page_scale = (f64::from(median_height) / page_height as f64 * 20.0) as f32;
    let mut out = Vec::with_capacity(boxes.len() * FEATURES);
    for (index, line) in boxes.iter().enumerate() {
        let [x0, y0, x1, y1] = *line;
        out.extend([
            x0 / width,
            y0 / height,
            x1 / width,
            y1 / height,
            (x0 + x1) / (2.0 * width),
            (y0 + y1) / (2.0 * height),
            widths[index] / width,
            (heights[index] / median_height).ln(),
            (widths[index] / heights[index]).ln() / 4.0,
            page_scale,
        ]);
        out.extend(text_features(texts[index].trim()));
    }
    out
}

/// A line's length (wide characters twice, log-scaled), share of digits, whether it starts
/// like a list item, ends with a hyphen, ends a sentence, starts upper case.
fn text_features(text: &str) -> [f32; 6] {
    let (Some(first), Some(last)) = (text.chars().next(), text.chars().last()) else {
        return [0.0; 6];
    };
    let count = text.chars().count();
    let digits = text
        .chars()
        .filter(|&character| is_digit(character))
        .count();
    let head: String = text.chars().take(3).collect();
    let numbered = head.trim_end_matches(['.', ')']);
    let item = ITEM_MARKS.contains(first)
        || (!numbered.is_empty() && numbered.chars().all(is_digit) && count > 3);
    [
        ((display_width(text) as f64).ln_1p() / 5.0) as f32,
        (digits as f64 / count as f64) as f32,
        f32::from(u8::from(item)),
        f32::from(u8::from(text.ends_with('-'))),
        f32::from(u8::from(SENTENCE_END.contains(last))),
        f32::from(u8::from(first.is_uppercase())),
    ]
}

/// `log(eˣ + eʸ)` as NumPy's `logaddexp` computes it in f32.
fn log_add_exp(x: f32, y: f32) -> f32 {
    if x == y {
        return x + std::f32::consts::LN_2;
    }
    let difference = x - y;
    if difference > 0.0 {
        x + (-difference).exp().ln_1p()
    } else {
        y + difference.exp().ln_1p()
    }
}

/// The order of `count` lines from pairwise logits (`logits[i][j] > 0`: `i` before `j`): the
/// ranking by row and column sums of `log P(i before j)`, then insertion moves that each take a
/// line out and put it back where the summed log-probability gains most.
pub fn decode_order(logits: &[f32], count: usize) -> Vec<usize> {
    if count <= 1 {
        return (0..count).collect();
    }
    let probabilities: Vec<f32> = logits
        .iter()
        .map(|&logit| -log_add_exp(0.0, -logit))
        .collect();
    let at = |row: usize, column: usize| probabilities[row * count + column];
    let mut columns = vec![0.0f32; count];
    for row in probabilities.chunks_exact(count) {
        columns
            .iter_mut()
            .zip(row)
            .for_each(|(sum, value)| *sum += value);
    }
    let ranks: Vec<f32> = probabilities
        .chunks_exact(count)
        .zip(&columns)
        .map(|(row, column)| -sum(row) + column)
        .collect();
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|&left, &right| ranks[left].total_cmp(&ranks[right]));
    for _ in 0..SWEEPS {
        let mut improved = false;
        for line in order.clone() {
            let rest: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&other| other != line)
                .collect();
            let gains = insertion_gains(&rest, |other| at(other, line), |other| at(line, other));
            let best = argmax(&gains);
            let current = order
                .iter()
                .position(|&other| other == line)
                .expect("the line is in the order");
            if gains[best] > gains[current] + MIN_GAIN {
                order = rest;
                order.insert(best, line);
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    order
}

/// For each slot of `rest`, the summed log-probability of a line placed there: of following
/// each line before the slot (`follows`), and of preceding each line after it (`precedes`).
fn insertion_gains(
    rest: &[usize],
    follows: impl Fn(usize) -> f32,
    precedes: impl Fn(usize) -> f32,
) -> Vec<f32> {
    let mut leading = vec![0.0f32; rest.len() + 1];
    for (slot, &other) in rest.iter().enumerate() {
        leading[slot + 1] = leading[slot] + follows(other);
    }
    let mut trailing = vec![0.0f32; rest.len() + 1];
    for (slot, &other) in rest.iter().enumerate().rev() {
        trailing[slot] = trailing[slot + 1] + precedes(other);
    }
    leading
        .iter()
        .zip(&trailing)
        .map(|(leading, trailing)| leading + trailing)
        .collect()
}

/// Groups of lines by average linkage of pairwise "same group" logits (`count × count`):
/// the two groups with the highest mean logit between them merge while that mean is positive.
/// Each group's lines ascending, groups in order of their surviving index.
pub fn cluster(logits: &[f32], count: usize) -> Vec<Vec<usize>> {
    let mut linkage = Linkage::new(logits, count);
    for _ in 1..count {
        let first = argmax(&linkage.best_value);
        if linkage.best_value[first] <= 0.0 {
            break;
        }
        let partner = linkage.best[first];
        linkage.merge(first.min(partner), first.max(partner));
    }
    linkage.groups()
}

/// Average linkage's state: the summed logits between groups and their means (`-∞` on the
/// diagonal and for merged groups), each group's size and lines, and each row's best partner.
struct Linkage {
    count: usize,
    sums: Vec<f64>,
    means: Vec<f64>,
    size: Vec<f64>,
    alive: Vec<bool>,
    members: Vec<Vec<usize>>,
    best: Vec<usize>,
    best_value: Vec<f64>,
}

impl Linkage {
    fn new(logits: &[f32], count: usize) -> Self {
        let mut sums: Vec<f64> = logits.iter().map(|&logit| f64::from(logit)).collect();
        let mut means = sums.clone();
        for line in 0..count {
            sums[line * count + line] = 0.0;
            means[line * count + line] = f64::NEG_INFINITY;
        }
        let mut linkage = Self {
            count,
            sums,
            means,
            size: vec![1.0; count],
            alive: vec![true; count],
            members: (0..count).map(|line| vec![line]).collect(),
            best: vec![0; count],
            best_value: vec![0.0; count],
        };
        for row in 0..count {
            (linkage.best[row], linkage.best_value[row]) = linkage.best_of(row);
        }
        linkage
    }

    fn best_of(&self, row: usize) -> (usize, f64) {
        let values = &self.means[row * self.count..(row + 1) * self.count];
        let best = argmax(values);
        (best, values[best])
    }

    /// Merges group `gone` into `kept`, updating only the rows and columns that change.
    fn merge(&mut self, kept: usize, gone: usize) {
        let count = self.count;
        for column in 0..count {
            self.sums[kept * count + column] += self.sums[gone * count + column];
        }
        for row in 0..count {
            self.sums[row * count + kept] += self.sums[row * count + gone];
        }
        self.sums[kept * count + kept] = 0.0;
        self.size[kept] += self.size[gone];
        self.alive[gone] = false;
        let moved = std::mem::take(&mut self.members[gone]);
        self.members[kept].extend(moved);
        let row: Vec<f64> = (0..count)
            .map(|other| match self.alive[other] && other != kept {
                true => self.sums[kept * count + other] / (self.size[kept] * self.size[other]),
                false => f64::NEG_INFINITY,
            })
            .collect();
        for (other, &mean) in row.iter().enumerate() {
            self.means[kept * count + other] = mean;
            self.means[other * count + kept] = mean;
            self.means[gone * count + other] = f64::NEG_INFINITY;
            self.means[other * count + gone] = f64::NEG_INFINITY;
        }
        self.best_value[gone] = f64::NEG_INFINITY;
        (self.best[kept], self.best_value[kept]) = self.best_of(kept);
        for other in (0..count).filter(|&other| self.alive[other] && other != kept) {
            if self.best[other] == kept || self.best[other] == gone {
                (self.best[other], self.best_value[other]) = self.best_of(other);
            }
            if row[other] > self.best_value[other] {
                (self.best[other], self.best_value[other]) = (kept, row[other]);
            }
        }
    }

    /// The surviving groups, each's lines ascending, in order of their index.
    fn groups(self) -> Vec<Vec<usize>> {
        let alive = self.alive;
        self.members
            .into_iter()
            .zip(alive)
            .filter_map(|(mut group, alive)| {
                group.sort_unstable();
                alive.then_some(group)
            })
            .collect()
    }
}

/// The reading order with the layout heads: blocks from the same-block logits, the lines inside
/// a block by XY-cut (row by row in a grid block), the blocks by the mean order logit between
/// them. `classes`: each line's layout-class logits.
pub fn blocks_order(order: &[f32], same: &[f32], boxes: &[LineBox], classes: &[f32]) -> Vec<usize> {
    let count = boxes.len();
    if count <= 1 {
        return (0..count).collect();
    }
    let blocks = cluster(same, count);
    let inner: Vec<Vec<usize>> = blocks
        .iter()
        .map(|block| {
            let block_boxes: Vec<LineBox> = block.iter().map(|&line| boxes[line]).collect();
            let local = if is_grid(block_class(classes, block)) {
                top_down(&block_boxes)
            } else {
                xy_cut(&block_boxes)
            };
            local.into_iter().map(|index| block[index]).collect()
        })
        .collect();
    let means = block_means(order, count, &blocks);
    let blocks_count = blocks.len();
    decode_order(&means, blocks_count)
        .into_iter()
        .flat_map(|block| inner[block].iter().copied())
        .collect()
}

/// A block's class: the largest of its lines' summed class logits.
pub fn block_class(classes: &[f32], block: &[usize]) -> usize {
    let mut sums = [0.0f32; CLASSES];
    for &line in block {
        for (sum, logit) in sums
            .iter_mut()
            .zip(&classes[line * CLASSES..(line + 1) * CLASSES])
        {
            *sum += logit;
        }
    }
    argmax(&sums)
}

/// The mean order logit between blocks `p < q` (`M[p][q]`, `M = G · logits · Gᵀ` with `G`
/// each block's membership over its size), antisymmetric, in f32.
fn block_means(order: &[f32], count: usize, blocks: &[Vec<usize>]) -> Vec<f32> {
    let total = blocks.len();
    let mut means = vec![0.0f32; total * total];
    for first in 0..total {
        let share = 1.0 / blocks[first].len() as f64;
        for second in first + 1..total {
            let other_share = 1.0 / blocks[second].len() as f64;
            let mean: f64 = blocks[second]
                .iter()
                .map(|&column| {
                    let column_sum: f64 = blocks[first]
                        .iter()
                        .map(|&row| share * f64::from(order[row * count + column]))
                        .sum();
                    column_sum * other_share
                })
                .sum();
            means[first * total + second] = mean as f32;
            means[second * total + first] = (-mean) as f32;
        }
    }
    means
}

/// Rows top to bottom (centres within half the median line height share a row), each left to
/// right.
pub fn top_down(boxes: &[LineBox]) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let heights: Vec<f32> = boxes.iter().map(|line| line[3] - line[1]).collect();
    let row_height = (median(&heights) * 0.5).max(ROW_SLACK);
    let row = |line: &LineBox| (((line[1] + line[3]) / 2.0) / row_height).round_ties_even() as i64;
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&left, &right| {
        let key = |index: usize| (row(&boxes[index]), boxes[index][0]);
        let (left, right) = (key(left), key(right));
        left.0.cmp(&right.0).then(left.1.total_cmp(&right.1))
    });
    order
}

/// Recursive XY-cut: split at the widest whitespace gap on either axis, the upper or left part
/// first; without a gap, top-down order.
pub fn xy_cut(boxes: &[LineBox]) -> Vec<usize> {
    let all: Vec<usize> = (0..boxes.len()).collect();
    cut_recursively(boxes, &all)
}

fn cut_recursively(boxes: &[LineBox], ids: &[usize]) -> Vec<usize> {
    if ids.len() <= 1 {
        return ids.to_vec();
    }
    let part: Vec<LineBox> = ids.iter().map(|&id| boxes[id]).collect();
    let rows = widest_gap(&part, Axis::Rows);
    let columns = widest_gap(&part, Axis::Columns);
    let cut = match (rows, columns) {
        (None, None) => {
            return top_down(&part)
                .into_iter()
                .map(|index| ids[index])
                .collect();
        }
        (Some(rows), Some(columns)) => {
            if rows.gap >= columns.gap {
                rows
            } else {
                columns
            }
        }
        (Some(cut), None) | (None, Some(cut)) => cut,
    };
    let (first, second) = cut.order.split_at(cut.at);
    let first: Vec<usize> = first.iter().map(|&index| ids[index]).collect();
    let second: Vec<usize> = second.iter().map(|&index| ids[index]).collect();
    let mut out = cut_recursively(boxes, &first);
    out.extend(cut_recursively(boxes, &second));
    out
}

/// Blocks in reading order by recursive XY-cut at whitespace gaps of at least the median line
/// height, for pages the order network does not know (vertical pages, read turned); the lines
/// inside a block by `xy_cut`.
pub fn xy_cut_blocks(boxes: &[LineBox]) -> Vec<Vec<usize>> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let heights: Vec<f32> = boxes.iter().map(|line| line[3] - line[1]).collect();
    let min_gap = median(&heights);
    let all: Vec<usize> = (0..boxes.len()).collect();
    let mut blocks = Vec::new();
    cut_blocks(boxes, &all, min_gap, &mut blocks);
    blocks
}

fn cut_blocks(boxes: &[LineBox], ids: &[usize], min_gap: f32, blocks: &mut Vec<Vec<usize>>) {
    let part: Vec<LineBox> = ids.iter().map(|&id| boxes[id]).collect();
    if ids.len() > 1 {
        let rows = widest_gap(&part, Axis::Rows);
        let columns = widest_gap(&part, Axis::Columns);
        let gap = |cut: &Option<Cut>| cut.as_ref().map_or(0.0, |cut| cut.gap);
        let cut = if gap(&rows) >= gap(&columns) {
            rows
        } else {
            columns
        };
        if let Some(cut) = cut.filter(|cut| cut.gap >= min_gap) {
            let (first, second) = cut.order.split_at(cut.at);
            let first: Vec<usize> = first.iter().map(|&index| ids[index]).collect();
            let second: Vec<usize> = second.iter().map(|&index| ids[index]).collect();
            cut_blocks(boxes, &first, min_gap, blocks);
            cut_blocks(boxes, &second, min_gap, blocks);
            return;
        }
    }
    blocks.push(xy_cut(&part).into_iter().map(|index| ids[index]).collect());
}

#[derive(Clone, Copy)]
enum Axis {
    Rows,
    Columns,
}

/// A cut: the boxes sorted along the axis, split before position `at` at a gap of `gap`.
struct Cut {
    gap: f32,
    at: usize,
    order: Vec<usize>,
}

/// The widest gap between the boxes along `axis`, if any.
fn widest_gap(boxes: &[LineBox], axis: Axis) -> Option<Cut> {
    let (low, high) = match axis {
        Axis::Rows => (1, 3),
        Axis::Columns => (0, 2),
    };
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&left, &right| boxes[left][low].total_cmp(&boxes[right][low]));
    let mut end = boxes[order[0]][high];
    let (mut best_gap, mut best_at) = (0.0f32, None);
    for (at, &index) in order.iter().enumerate().skip(1) {
        let gap = boxes[index][low] - end;
        if gap > best_gap {
            (best_gap, best_at) = (gap, Some(at));
        }
        end = end.max(boxes[index][high]);
    }
    best_at.map(|at| Cut {
        gap: best_gap,
        at,
        order,
    })
}
