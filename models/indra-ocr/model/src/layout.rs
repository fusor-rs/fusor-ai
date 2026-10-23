//! The page's layout blocks and Markdown from the ordered lines and the order network's layout
//! heads: a block is a run of the reading order whose neighbours' same-block logits are
//! positive, labeled by its lines' summed class logits, and written by its class: a section
//! header as `## `, a list one item per line, a table as HTML (rows by vertical centre,
//! columns anchored on the fullest rows), display math as `$$…$$`, page headers and footers
//! left out, anything else one paragraph.

use std::sync::LazyLock;

use kernels::numpy::{mean, median};
use regex::Regex;
use serde::Serialize;

use crate::crops::LineBox;
use crate::reading::block_class;
use crate::text::is_alpha;

/// The layout classes, in the order network's order.
pub const CLASSES: usize = 10;
const LABELS: [&str; CLASSES] = [
    "Text",
    "Section-Header",
    "Page-Header",
    "Page-Footer",
    "List-Group",
    "Table",
    "Caption",
    "Footnote",
    "Table-Of-Contents",
    "Form",
];
const SECTION: usize = 1;
const HEADER: usize = 2;
const FOOTER: usize = 3;
const LIST: usize = 4;
pub(crate) const TABLE: usize = 5;
/// Classes whose lines read row by row.
const GRID: [usize; 3] = [TABLE, 8, 9];
/// A list item's start, followed by its text or alone on its line.
static ITEM: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)")
        .expect("the list-item pattern compiles")
});
/// A wrapped cell's row: its gap to the row above under this share of the median row gap and
/// `WRAP_HEIGHT` line heights, with fewer lines than that row.
const WRAP_GAP: f64 = 0.5;
const WRAP_HEIGHT: f64 = 0.6;
/// A first row that is one line over this share of the table's width is its title.
const TITLE_SPAN: f32 = 0.6;

pub fn is_grid(class: usize) -> bool {
    GRID.contains(&class)
}

/// A layout block: its class, the union of its lines' boxes, and their text, a line each.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Block {
    pub label: &'static str,
    pub bbox: [f64; 4],
    pub text: String,
}

/// The lines of one page as the layout sees them, the classes per line `CLASSES` wide.
pub struct Lines<'page> {
    pub boxes: &'page [LineBox],
    pub texts: &'page [String],
    pub classes: &'page [f32],
    pub same: &'page [f32],
}

/// Runs of the reading order with a positive same-block logit between neighbours.
pub fn split_blocks(order: &[usize], same: &[f32], count: usize) -> Vec<Vec<usize>> {
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for &line in order {
        match blocks.last_mut() {
            Some(block) if same[block[block.len() - 1] * count + line] > 0.0 => block.push(line),
            _ => blocks.push(vec![line]),
        }
    }
    blocks
}

/// `round(value, digits)`, as Python rounds: the decimal nearest the exact binary value, ties
/// to even (as Rust's formatting rounds).
pub fn rounded(value: f64, digits: usize) -> f64 {
    format!("{value:.digits$}")
        .parse()
        .expect("a formatted float parses")
}

/// The union of the boxes of `lines`.
fn union(boxes: &[LineBox], lines: &[usize]) -> LineBox {
    let mut out = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for &line in lines {
        let [x0, y0, x1, y1] = boxes[line];
        out = [
            out[0].min(x0),
            out[1].min(y0),
            out[2].max(x1),
            out[3].max(y1),
        ];
    }
    out
}

/// The page's blocks in reading order.
pub fn blocks(lines: &Lines, order: &[usize]) -> Vec<Block> {
    split_blocks(order, lines.same, lines.boxes.len())
        .into_iter()
        .map(|block| Block {
            label: LABELS[block_class(lines.classes, &block)],
            bbox: union(lines.boxes, &block).map(|value| rounded(f64::from(value), 1)),
            text: block
                .iter()
                .map(|&line| lines.texts[line].as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        })
        .collect()
}

/// The page's Markdown. `cells(line)`: the line's table cells, a box and text each.
pub fn markdown(
    lines: &Lines,
    order: &[usize],
    cells: impl Fn(usize) -> Vec<(LineBox, String)>,
) -> String {
    let parts = split_blocks(order, lines.same, lines.boxes.len())
        .into_iter()
        .filter_map(|block| block_markdown(lines, &block, &cells));
    parts
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// One block's Markdown; nothing for a page header or footer.
fn block_markdown(
    lines: &Lines,
    block: &[usize],
    cells: &impl Fn(usize) -> Vec<(LineBox, String)>,
) -> Option<String> {
    let class = block_class(lines.classes, block);
    let texts: Vec<&str> = block
        .iter()
        .map(|&line| lines.texts[line].as_str())
        .collect();
    if class == HEADER || class == FOOTER {
        return None;
    }
    if texts.iter().all(|text| is_formula(text.trim())) {
        let formulas = texts.iter().map(|text| {
            let text = text.trim();
            format!("$${}$$", &text[1..text.len() - 1])
        });
        return Some(formulas.collect::<Vec<_>>().join("\n"));
    }
    Some(match class {
        SECTION => format!("## {}", join_lines(&texts)),
        TABLE if block.len() > 1 => {
            let (boxes, texts): (Vec<LineBox>, Vec<String>) = block
                .iter()
                .flat_map(|&line| {
                    let split = cells(line);
                    if split.is_empty() {
                        vec![(lines.boxes[line], lines.texts[line].clone())]
                    } else {
                        split
                    }
                })
                .unzip();
            table_html(&boxes, &texts)
        }
        LIST => list(&texts),
        _ => join_lines(&texts),
    })
}

/// A line that is one formula: `$…$` without another `$`.
fn is_formula(text: &str) -> bool {
    text.len() > 2
        && text.starts_with('$')
        && text.ends_with('$')
        && !text[1..text.len() - 1].contains('$')
}

/// One list item per line, an item starting at a bullet or number.
fn list(texts: &[&str]) -> String {
    let mut items: Vec<Vec<&str>> = Vec::new();
    for &text in texts {
        match items.last_mut() {
            Some(item) if !ITEM.is_match(text) => item.push(text),
            _ => items.push(vec![text]),
        }
    }
    items
        .iter()
        .map(|item| join_lines(item))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Han, kana, CJK punctuation and fullwidth forms, written without spaces.
fn is_cjk(character: char) -> bool {
    matches!(u32::from(character), 0x3000..=0x30FF | 0x3400..=0x9FFF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF | 0x20000..=0x2FA1F)
}

/// The lines of one paragraph joined: a word split by a hyphen at a line end joined, no space
/// between two CJK characters.
pub fn join_lines(texts: &[&str]) -> String {
    let mut out = String::new();
    for text in texts
        .iter()
        .map(|text| text.trim())
        .filter(|text| !text.is_empty())
    {
        let mut tail = out.chars().rev();
        let (last, before) = (tail.next(), tail.next());
        let first = text.chars().next().expect("the line is not empty");
        let hyphenated = last == Some('-')
            && before.is_some_and(is_alpha)
            && is_alpha(first)
            && first.is_lowercase();
        if hyphenated {
            out.pop();
        } else if !out.is_empty() && !(last.is_some_and(is_cjk) && is_cjk(first)) {
            out.push(' ');
        }
        out.push_str(text);
    }
    out
}

/// An HTML table from cell lines, one text line per cell. A first row that is one line across
/// most of the table is its title, written before it; the first remaining row is the header.
fn table_html(boxes: &[LineBox], texts: &[String]) -> String {
    let rows = rows(boxes);
    let left = boxes
        .iter()
        .map(|line| line[0])
        .fold(f32::INFINITY, f32::min);
    let right = boxes
        .iter()
        .map(|line| line[2])
        .fold(f32::NEG_INFINITY, f32::max);
    let widest = rows.iter().map(Vec::len).max().expect("a table has rows");
    let first = rows[0][0];
    let title = rows.len() > 2
        && rows[0].len() == 1
        && widest > 1
        && boxes[first][2] - boxes[first][0] > TITLE_SPAN * (right - left);
    if title {
        let kept: Vec<usize> = (0..texts.len()).filter(|&line| line != first).collect();
        let kept_boxes: Vec<LineBox> = kept.iter().map(|&line| boxes[line]).collect();
        let kept_texts: Vec<String> = kept.iter().map(|&line| texts[line].clone()).collect();
        return format!(
            "{}\n\n{}",
            texts[first],
            table_html(&kept_boxes, &kept_texts)
        );
    }
    let full: Vec<&Vec<usize>> = rows.iter().filter(|row| row.len() == widest).collect();
    let bands: Vec<[f32; 2]> = (0..widest)
        .map(|column| {
            let side = |side: usize| {
                let values: Vec<f32> = full.iter().map(|row| boxes[row[column]][side]).collect();
                median(&values)
            };
            [side(0), side(2)]
        })
        .collect();
    let mut edges = vec![f32::NEG_INFINITY];
    edges.extend(bands.windows(2).map(|pair| (pair[0][1] + pair[1][0]) / 2.0));
    edges.push(f32::INFINITY);
    let rows = merge_wrapped(rows, boxes);
    let mut out = vec!["<table>".to_string()];
    for (index, row) in rows.iter().enumerate() {
        let tag = if index == 0 && rows.len() > 1 {
            "th"
        } else {
            "td"
        };
        let columns = Columns {
            bands: &bands,
            edges: &edges,
        };
        out.push(table_row(row, (boxes, texts), columns, tag));
    }
    out.push("</table>".into());
    out.join("\n")
}

/// A table's column bands (each the median left and right of the fullest rows' cells) and the
/// edges between them (the midpoints, unbounded at the ends).
#[derive(Clone, Copy)]
struct Columns<'table> {
    bands: &'table [[f32; 2]],
    edges: &'table [f32],
}

impl Columns<'_> {
    /// The bands a cell spans: every band it covers over half of, else the band it starts in.
    fn span(&self, [x0, _, x1, _]: LineBox) -> (usize, usize) {
        let last_band = self.bands.len() - 1;
        let band_of = |x: f32| {
            self.edges
                .partition_point(|&edge| edge <= x)
                .saturating_sub(1)
        };
        let (first, last) = (band_of(x0), band_of(x1).min(last_band));
        let covered: Vec<usize> = (first..=last)
            .filter(|&band| {
                let [left, right] = self.bands[band];
                x1.min(right) - x0.max(left) > 0.5 * (right - left)
            })
            .collect();
        match covered.first() {
            Some(&start) => (start, covered.len()),
            None => (first.min(last_band), 1),
        }
    }
}

/// One table row: each cell in the band it starts in, spanning the bands it covers; lines in
/// one band joined.
fn table_row(
    row: &[usize],
    (boxes, texts): (&[LineBox], &[String]),
    columns: Columns,
    tag: &str,
) -> String {
    let mut lines = row.to_vec();
    lines.sort_by(|&left, &right| {
        let (left, right) = (boxes[left], boxes[right]);
        left[1]
            .total_cmp(&right[1])
            .then(left[0].total_cmp(&right[0]))
    });
    let mut cells: Vec<Option<(usize, String)>> = vec![None; columns.bands.len()];
    for line in lines {
        let (start, span) = columns.span(boxes[line]);
        cells[start] = Some(match cells[start].take() {
            None => (span, texts[line].clone()),
            Some((width, text)) => (width.max(span), format!("{text} {}", texts[line])),
        });
    }
    let mut out = String::from("<tr>");
    let mut column = 0;
    while column < cells.len() {
        match &cells[column] {
            Some((span, text)) => {
                let colspan = if *span > 1 {
                    format!(" colspan=\"{span}\"")
                } else {
                    String::new()
                };
                out.push_str(&format!("<{tag}{colspan}>{}</{tag}>", escape(text)));
                column += span;
            }
            None => {
                out.push_str(&format!("<{tag}></{tag}>"));
                column += 1;
            }
        }
    }
    out.push_str("</tr>");
    out
}

/// Cell lines grouped into rows by vertical centre (within half the median line height of the
/// row's mean centre), top to bottom, each left to right.
fn rows(boxes: &[LineBox]) -> Vec<Vec<usize>> {
    let centers: Vec<f32> = boxes.iter().map(|line| (line[1] + line[3]) / 2.0).collect();
    let heights: Vec<f32> = boxes.iter().map(|line| line[3] - line[1]).collect();
    let half = 0.5 * f64::from(median(&heights));
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&left, &right| centers[left].total_cmp(&centers[right]));
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for line in order {
        let joins = rows.last().is_some_and(|row| {
            let row_centers: Vec<f32> = row.iter().map(|&member| centers[member]).collect();
            f64::from((centers[line] - mean(&row_centers)).abs()) < half
        });
        match rows.last_mut() {
            Some(row) if joins => row.push(line),
            _ => rows.push(vec![line]),
        }
    }
    for row in &mut rows {
        row.sort_by(|&left, &right| boxes[left][0].total_cmp(&boxes[right][0]));
    }
    rows
}

/// The line rows of wrapped cells joined into their table row: a row whose gap to the row
/// above is under half the median row gap and `WRAP_HEIGHT` line heights, with fewer lines.
fn merge_wrapped(rows: Vec<Vec<usize>>, boxes: &[LineBox]) -> Vec<Vec<usize>> {
    if rows.len() < 3 {
        return rows;
    }
    let heights: Vec<f32> = boxes.iter().map(|line| line[3] - line[1]).collect();
    let height = f64::from(median(&heights));
    let gaps: Vec<f64> = rows
        .windows(2)
        .map(|pair| {
            let top = pair[1]
                .iter()
                .map(|&line| boxes[line][1])
                .fold(f32::INFINITY, f32::min);
            let bottom = pair[0]
                .iter()
                .map(|&line| boxes[line][3])
                .fold(f32::NEG_INFINITY, f32::max);
            f64::from(top - bottom)
        })
        .collect();
    let median_gap = median(&gaps);
    let mut out = vec![rows[0].clone()];
    for (gap, pair) in gaps.iter().zip(rows.windows(2)) {
        let wrapped = *gap < WRAP_GAP * median_gap
            && *gap < WRAP_HEIGHT * height
            && pair[1].len() < pair[0].len();
        match out.last_mut() {
            Some(last) if wrapped => last.extend(&pair[1]),
            _ => out.push(pair[1].clone()),
        }
    }
    out
}

/// `html.escape(text, quote=False)`.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
