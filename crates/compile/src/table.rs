//! A compiler's rename table: each bundle tensor, in bundle order, as the row-concatenation of
//! checkpoint tensors (q, k and v fused), laid out from theirs.

use std::ops::Range;

use crate::{Checkpoint, Error, Profile, Tensors};

/// How a tensor's values are laid out in the bundle, from the checkpoint's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layout {
    AsIs,
    /// Zero rows (a vector: zero values) after the last, up to this many.
    PadRows(usize),
    /// Zero columns after each row's last, up to this many.
    PadColumns(usize),
    /// Rows and columns swapped.
    Transpose,
    /// Only these rows.
    Rows(Range<usize>),
}

/// A bundle tensor: the checkpoint tensors concatenated along rows, laid out as `layout`.
struct Entry {
    name: String,
    sources: Vec<String>,
    layout: Layout,
}

const PARTS: [&str; 2] = ["weight", "bias"];

/// `(runtime name, checkpoint sources, layout)`, in bundle order.
#[derive(Default)]
pub struct Table(Vec<Entry>);

impl Table {
    pub fn push(&mut self, name: String, sources: Vec<String>, layout: Layout) {
        self.0.push(Entry {
            name,
            sources,
            layout,
        });
    }

    pub fn one(&mut self, name: String, source: String) {
        self.push(name, vec![source], Layout::AsIs);
    }

    /// A layer's weight and bias, each the row-concatenation of those of `sources`.
    pub fn linear(&mut self, name: &str, sources: &[String], layout: &Layout) {
        for part in PARTS {
            let parts = sources.iter().map(|source| format!("{source}.{part}"));
            let part_layout = match (part, layout) {
                ("bias", Layout::PadColumns(_)) => Layout::AsIs,
                _ => layout.clone(),
            };
            self.push(format!("{name}.{part}"), parts.collect(), part_layout);
        }
    }

    /// Every entry, read from `checkpoint`, into `tensors`, in order.
    pub fn write<P: Profile>(
        self,
        checkpoint: &Checkpoint,
        tensors: &mut Tensors<P>,
    ) -> Result<(), Error> {
        for entry in self.0 {
            let (shape, values) = read(checkpoint, &entry)?;
            tensors.add(&entry.name, shape, &values)?;
        }
        Ok(())
    }
}

/// The entry's values and bundle shape: vectors stay vectors (a single source of one row, a
/// one-entry table, is one too), anything else is rows of its first dimension (a convolution's
/// kernel flattened), laid out as its layout says.
fn read(checkpoint: &Checkpoint, entry: &Entry) -> Result<(Vec<usize>, Vec<f32>), Error> {
    let mut values = Vec::new();
    let mut rows = 0;
    for source in &entry.sources {
        let shape = checkpoint.shape(source)?;
        rows += shape[0];
        values.extend(checkpoint.f32(source)?);
    }
    let first = checkpoint.shape(&entry.sources[0])?;
    let vector = first.len() == 1 || first[0] == 1 && entry.sources.len() == 1;
    let cols = values.len() / rows;
    match &entry.layout {
        Layout::AsIs if vector => Ok((vec![values.len()], values)),
        Layout::AsIs => Ok((vec![rows, cols], values)),
        Layout::PadRows(padded) => {
            values.resize(padded * cols, 0.0);
            let shape = if vector {
                vec![*padded]
            } else {
                vec![*padded, cols]
            };
            Ok((shape, values))
        }
        Layout::PadColumns(padded) => {
            let rows_of = values.chunks_exact(cols);
            let padded_rows = rows_of.flat_map(|row| {
                let zeros = std::iter::repeat_n(0.0, padded - cols);
                row.iter().copied().chain(zeros)
            });
            Ok((vec![rows, *padded], padded_rows.collect()))
        }
        Layout::Transpose => {
            let transposed = (0..cols).flat_map(|col| (0..rows).map(move |row| (row, col)));
            let values = transposed.map(|(row, col)| values[row * cols + col]);
            Ok((vec![cols, rows], values.collect()))
        }
        Layout::Rows(kept) => {
            let values = values[kept.start * cols..kept.end * cols].to_vec();
            Ok((vec![kept.len(), cols], values))
        }
    }
}
