//! The prompt, byte for byte as d1's reference renders it (`prompt.py`), and what each answer
//! slot reads (`question_slots`, `readout_ids`). A request's state follows its media's markup at
//! the head of the user turn, then `QUESTION:`; each question is its own suffix, closing the
//! user turn and opening the assistant's after an empty thinking block, its answer slots the
//! last token or the colons of named markers. Up to the checkpoint's set size of questions are
//! packed after one state (`pack`): the first as its lone prompt tokenizes, each further one
//! after `QUESTION:` again. A slot reads either the head's logits over its options' tokens (a
//! `Table`: every option's forms, the best one counting) or, for a read, the state's positions.
//!
//! The caller's text never holds a special token: a zero-width space after the first character
//! of each one breaks it (`plain`).

use bundle::python::{self, Layout};
use bundle::{Encoded, Tokenizer};
use serde_json::{Map, Value};

use crate::Error;
use crate::grid::Grid;
use crate::model::{Codes, Config, PromptVersion};
use crate::request::{Criterion, INSTANCES, Kind, Question, Written};

const START: &str = "<|startoftext|>";
const TURN: &str = "<|im_start|>";
const END: &str = "<|im_end|>";
/// What the template writes after the assistant header before a reply that does not think.
const LEAD: &str = "<think></think>";
/// What opens a question after a state.
const HEADER: &str = "QUESTION:\n";
const VISION_START: &str = "<|vision_start|>";
const VISION_END: &str = "<|vision_end|>";
pub const IMAGE: &str = "<|image|>";
pub const AUDIO: &str = "<|audio|>";
/// What breaks a special token in the caller's text.
const BREAK: char = '\u{200b}';
const YES: [&str; 3] = ["yes", "Yes", "YES"];
const NO: [&str; 3] = ["no", "No", "NO"];
const EDGES: [&str; 4] = ["top", "bottom", "left", "right"];
/// A box edge's values as codes, and as numbers in thousandths ("1000" takes two tokens).
const EDGE_CODES: Grid = Grid {
    low: 0.0,
    high: 1.0,
    size: 128,
};
const THOUSANDTHS: Grid = Grid {
    low: 0.0,
    high: 0.999,
    size: 1000,
};
const THOUSAND: u32 = 1000;
/// The most levels a grid read as a score's digits has; a larger one is read as a choice.
const DIGITS: usize = 10;
const LETTERS: usize = 26;

/// What an answer slot reads: the head's logits over a table's tokens, or the state's
/// positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readout {
    Code(usize),
    Pointer,
}

/// The options of a slot: every token once, and each option's forms as positions among them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub tokens: Vec<u32>,
    pub options: Vec<Vec<usize>>,
}

impl Table {
    fn new(groups: &[Vec<u32>]) -> Self {
        let mut tokens: Vec<u32> = Vec::new();
        let options = groups.iter().map(|group| {
            let forms =
                group.iter().map(
                    |&token| match tokens.iter().position(|&seen| seen == token) {
                        Some(at) => at,
                        None => {
                            tokens.push(token);
                            tokens.len() - 1
                        }
                    },
                );
            forms.collect()
        });
        let options = options.collect();
        Self { tokens, options }
    }
}

/// The tables of a request's slots: `YES_NO` and `EDGE` first, then each question's own.
pub struct Tables(pub Vec<Table>);
pub const YES_NO: usize = 0;
pub const EDGE: usize = 1;

/// A question's suffix, its slots' token offsets from its last token, and what each reads.
#[derive(Debug, Clone)]
pub struct Suffix {
    pub text: String,
    pub offsets: Vec<usize>,
    pub readouts: Vec<Readout>,
}

/// One state's questions packed after it: the ids, the state's rows, where each question's
/// rows end, and each question's slots' rows.
#[derive(Debug, Clone)]
pub struct Packed {
    pub ids: Vec<u32>,
    /// Each id's character span of the prompt the first question makes alone.
    pub spans: Vec<(usize, usize)>,
    pub state: usize,
    pub ends: Vec<usize>,
    pub slots: Vec<Vec<usize>>,
}

/// A grid read at one slot: its table, whether its options are the tokenizer's own numbers, and
/// the codes the prompt names them by.
struct GridSlot {
    table: Table,
    numbers: bool,
    codes: Vec<String>,
}

pub struct Prompter {
    tokenizer: Tokenizer,
    /// The special tokens, longest first, as characters.
    specials: Vec<Vec<char>>,
    codes: Codes,
    numbers: bool,
    version: PromptVersion,
    yes_no: Table,
    edge: GridSlot,
}

impl Prompter {
    /// # Errors
    /// When the tokenizer has no one-token yes and no, or no codes for a box's edges.
    pub fn new(tokenizer: Tokenizer, config: &Config) -> Result<Self, Error> {
        let mut specials: Vec<Vec<char>> = tokenizer
            .specials()
            .iter()
            .map(|special| special.chars().collect::<Vec<_>>())
            .filter(|special| special.len() > 1 && !special.contains(&BREAK))
            .collect();
        specials.sort_by_key(|special| std::cmp::Reverse(special.len()));
        let mut prompter = Self {
            tokenizer,
            specials,
            codes: config.d1_codes,
            numbers: config.d1_numbers,
            version: config.d1_prompt,
            yes_no: Table::new(&[]),
            edge: GridSlot {
                table: Table::new(&[]),
                numbers: false,
                codes: Vec::new(),
            },
        };
        let (yes, no) = (prompter.single(&YES)?, prompter.single(&NO)?);
        if yes.is_empty() || no.is_empty() {
            return Err(Error::Format(
                "the tokenizer has no one-token yes or no".into(),
            ));
        }
        prompter.yes_no = Table::new(&[yes, no]);
        prompter.edge = match prompter.numbers(THOUSANDTHS, THOUSAND)? {
            Some(table) => GridSlot {
                table,
                numbers: true,
                codes: Vec::new(),
            },
            None => prompter.coded(EDGE_CODES)?,
        };
        Ok(prompter)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The grid a box's edges are read on.
    pub fn edge_grid(&self) -> Grid {
        if self.edge.numbers {
            THOUSANDTHS
        } else {
            EDGE_CODES
        }
    }

    /// A request's tables before its questions add theirs.
    pub fn tables(&self) -> Tables {
        Tables(vec![self.yes_no.clone(), self.edge.table.clone()])
    }

    /// Everything before the first question: the media's markup at the head of the user turn,
    /// then the state and `QUESTION:`, the state as `state_block` writes it.
    pub fn prefix(&self, state: &Value, markup: &str) -> String {
        let body = match state {
            Value::Null => String::new(),
            state => format!("{}\n{HEADER}", self.plain(&state_block(state))),
        };
        format!("{START}{TURN}user\n{markup}{body}")
    }

    /// What opens a question packed after another: none when there is no state.
    pub fn header(state: &Value) -> &'static str {
        if state.is_null() { "" } else { HEADER }
    }

    /// `text` with every special token in it broken (`BREAK` after its first character), and
    /// each character boundary of the result mapped to `text`'s.
    pub fn plain_boundaries(&self, text: &str) -> (Vec<char>, Vec<usize>) {
        let mut chars: Vec<char> = text.chars().collect();
        let mut boundaries: Vec<usize> = (0..=chars.len()).collect();
        loop {
            let mut broken = false;
            let mut at = 0;
            while at < chars.len() {
                let found = self
                    .specials
                    .iter()
                    .find(|special| chars[at..].starts_with(special));
                let Some(special) = found else {
                    at += 1;
                    continue;
                };
                chars.insert(at + 1, BREAK);
                boundaries.insert(at + 1, boundaries[at + 1]);
                at += special.len() + 1;
                broken = true;
            }
            if !broken {
                return (chars, boundaries);
            }
        }
    }

    fn plain(&self, text: &str) -> String {
        self.plain_boundaries(text).0.into_iter().collect()
    }

    /// A question's suffix: its text up to its last answer slot, and its slots.
    ///
    /// # Errors
    /// When the tokenizer has no codes for its options.
    pub fn suffix(&self, question: &Question, tables: &mut Tables) -> Result<Suffix, Error> {
        let said = question.instructions.as_str();
        let asked = match &question.kind {
            Kind::Noul(criteria) => noul(said, criteria.as_ref())?,
            Kind::Choice {
                options,
                multiple: true,
            } => multiple(said, options),
            Kind::Choice { options, .. } => self.choice(said, options, tables)?,
            Kind::Score(levels) => {
                let legend = levels
                    .iter()
                    .enumerate()
                    .map(|(level, text)| format!("{level} {text}"));
                let legend = legend.collect::<Vec<_>>().join("\n");
                let last = levels.len() - 1;
                let body =
                    format!("{said}\n\n{legend}\n\nReply with a single digit 0-{last} only.");
                Asked::single(body, tables.add(self.digits(levels.len())?))
            }
            Kind::Read { value, unit } => read(said, *value, unit.as_deref()),
            Kind::Estimate { grid, unit } => self.estimate(said, *grid, unit.as_deref(), tables)?,
            Kind::Box => self.boxed(said, None),
            Kind::Boxes(blocks) => self.boxed(said, Some(*blocks)),
        };
        let after = matches!(question.kind, Kind::Boxes(blocks)
            if blocks < INSTANCES || self.version == PromptVersion::Second);
        let (text, offsets) = self.marked(&asked.body, &asked.markers, after)?;
        Ok(Suffix {
            text,
            offsets,
            readouts: asked.readouts,
        })
    }

    /// A choice: each option's code and text, its answer the code.
    fn choice(
        &self,
        said: &str,
        options: &[Criterion],
        tables: &mut Tables,
    ) -> Result<Asked, Error> {
        let names: Vec<&str> = options.iter().map(|option| option.name.as_str()).collect();
        let (codes, table) = self.aliased(&names)?;
        let lines = options.iter().zip(&codes).map(|(option, code)| {
            let text = match option.description.as_deref() {
                Some(description) if !description.is_empty() => description.to_owned(),
                _ => option.name.replace('_', " "),
            };
            format!("{code} {text}")
        });
        let lines = lines.collect::<Vec<_>>().join("\n");
        let body = format!("{said}\n\nOptions:\n{lines}\n\nReply with the option code only.");
        Ok(Asked::single(body, tables.add(table)))
    }

    /// An estimate: the tokenizer's numbers where the checkpoint reads them so, else codes.
    fn estimate(
        &self,
        said: &str,
        grid: Grid,
        unit: Option<&str>,
        tables: &mut Tables,
    ) -> Result<Asked, Error> {
        let slot = self.grid_slot(grid, 1)?;
        let unit = unit.map_or(String::new(), |unit| format!(" {unit}"));
        let (low, high, step) = (general(grid.low), general(grid.high), general(grid.step()));
        let body = if slot.numbers {
            format!(
                "{said}\n\nEstimate from {low} to {high}{unit}, in steps of {step}. Reply with the \
                 number only."
            )
        } else {
            let (first, last) = (&slot.codes[0], &slot.codes[slot.codes.len() - 1]);
            format!(
                "{said}\n\nEstimate from {low} to {high}{unit} on {} equally spaced values, step \
                 {step}: {first} is {low}, {last} is {high}. Reply with the code only.",
                grid.size
            )
        };
        Ok(Asked::single(body, tables.add(slot.table)))
    }

    /// A box, or with `blocks` every instance: a presence and four edges a block.
    fn boxed(&self, said: &str, blocks: Option<usize>) -> Asked {
        let legend = self.edge_legend();
        let body = match blocks {
            None => {
                let one = if self.edge.numbers {
                    "one number"
                } else {
                    "one code"
                };
                format!(
                    "{said}\n\nPresence is yes or no. {legend} Reply with yes or no at present, \
                     then {one} at top, bottom, left and right."
                )
            }
            Some(_) => {
                let most = match self.version {
                    PromptVersion::First => format!("{INSTANCES} at most, "),
                    PromptVersion::Second => String::new(),
                };
                format!(
                    "{said}\n\nEvery one of them, {most}in reading order (left edge, then top). \
                     Present k is yes when there are at least k. {legend} Reply with yes or no at \
                     each present, then its top, bottom, left and right."
                )
            }
        };
        let names = || ["present"].into_iter().chain(EDGES);
        let markers: Vec<String> = match blocks {
            None => names().map(str::to_owned).collect(),
            Some(blocks) => (1..=blocks)
                .flat_map(|block| names().map(move |name| format!("{name} {block}")))
                .collect(),
        };
        let block = [Readout::Code(YES_NO)]
            .into_iter()
            .chain([Readout::Code(EDGE); 4]);
        let readouts = block.cycle().take(markers.len()).collect();
        Asked {
            body,
            markers,
            readouts,
        }
    }

    /// The suffix of `body`, closing the user turn and opening the assistant's, then each
    /// marker on its own line ending with a colon, the colons' tokens its slots, else its last
    /// token alone; with `after`, a newline after the last colon where the tokenizer writes the
    /// two as one token. The offsets count back from the suffix's last token.
    fn marked(
        &self,
        body: &str,
        markers: &[String],
        after: bool,
    ) -> Result<(String, Vec<usize>), Error> {
        let mut text = format!("{}{END}\n{TURN}assistant\n{LEAD}", self.plain(body));
        if markers.is_empty() {
            return Ok((text, vec![0]));
        }
        let mut colons = Vec::with_capacity(markers.len());
        for (index, marker) in markers.iter().enumerate() {
            if index > 0 {
                text.push('\n');
            }
            text.push_str(&self.plain(marker));
            colons.push(text.chars().count());
            text.push(':');
        }
        if after && self.joins_newline(&text)? {
            text.push('\n');
        }
        let Encoded { ids, spans } = self.tokenizer.encode_spans(&text)?;
        let slots = colons.iter().map(|&colon| {
            let token = spans
                .iter()
                .position(|&(start, end)| start <= colon && colon < end);
            let token = token.ok_or_else(|| {
                Error::Format("a slot marker's colon has no token of its own".into())
            })?;
            Ok(ids.len() - 1 - token)
        });
        Ok((text, slots.collect::<Result<_, Error>>()?))
    }

    /// Whether `text` and a newline after it end on one token that ends with the colon.
    fn joins_newline(&self, text: &str) -> Result<bool, Error> {
        let ids = self.tokenizer.encode(&format!("{text}\n"))?;
        let last = self.tokenizer.decode(&ids[ids.len() - 1..])?;
        Ok(last.trim_end().ends_with(':'))
    }

    /// `prefix` and the questions `suffixes` packed after it: the first tokenized with the
    /// prefix, each further one after `header`.
    ///
    /// # Errors
    /// When the text doesn't tokenize.
    pub fn pack(&self, prefix: &str, header: &str, suffixes: &[&Suffix]) -> Result<Packed, Error> {
        let first = format!("{prefix}{}", suffixes[0].text);
        let Encoded { mut ids, spans } = self.tokenizer.encode_spans(&first)?;
        let state = ids.len() - self.tokenizer.encode(&suffixes[0].text)?.len();
        let mut ends = vec![ids.len()];
        for suffix in &suffixes[1..] {
            ids.extend(self.tokenizer.encode(&format!("{header}{}", suffix.text))?);
            ends.push(ids.len());
        }
        let slots = suffixes.iter().zip(&ends).map(|(suffix, &end)| {
            suffix
                .offsets
                .iter()
                .map(|offset| end - 1 - offset)
                .collect()
        });
        Ok(Packed {
            slots: slots.collect(),
            ids,
            spans,
            state,
            ends,
        })
    }

    /// A media item's markup: its tokens between the vision markers.
    pub fn markup(token: &str, count: usize) -> String {
        format!("{VISION_START}{}{VISION_END}", token.repeat(count))
    }

    /// The single-token ids of `texts`, each once.
    fn single(&self, texts: &[&str]) -> Result<Vec<u32>, Error> {
        let mut ids = Vec::new();
        for text in texts {
            if let [id] = self.tokenizer.encode(text)?[..]
                && !ids.contains(&id)
            {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    /// The digits `0` to `levels − 1`, a score's levels.
    fn digits(&self, levels: usize) -> Result<Table, Error> {
        let groups = (0..levels).map(|level| {
            let group = self.single(&[&level.to_string()])?;
            if group.is_empty() {
                return Err(Error::Format(format!(
                    "the tokenizer writes {level} in more than one token"
                )));
            }
            Ok(group)
        });
        Ok(Table::new(&groups.collect::<Result<Vec<_>, Error>>()?))
    }

    /// A grid's slot: the tokenizer's own numbers (each value times `scale`) where the
    /// checkpoint reads them so and each is one token, else the grid's codes.
    fn grid_slot(&self, grid: Grid, scale: u32) -> Result<GridSlot, Error> {
        match self.numbers(grid, scale)? {
            Some(table) => Ok(GridSlot {
                table,
                numbers: true,
                codes: Vec::new(),
            }),
            None => self.coded(grid),
        }
    }

    /// The table of a grid's values as numbers, or none when the checkpoint doesn't read them
    /// so, a value times `scale` is no integer or the tokenizer writes it in several tokens.
    fn numbers(&self, grid: Grid, scale: u32) -> Result<Option<Table>, Error> {
        if !self.numbers {
            return Ok(None);
        }
        let mut groups = Vec::with_capacity(grid.size);
        for value in grid.values() {
            let scaled = value * f64::from(scale);
            let number = scaled.round_ties_even();
            if (number - scaled).abs() > 1e-6 {
                return Ok(None);
            }
            let written = (number as i64).to_string();
            if self.tokenizer.encode(&written)?.len() != 1 {
                return Ok(None);
            }
            groups.push(self.single(&[&written, &format!(" {written}")])?);
        }
        Ok(Some(Table::new(&groups)))
    }

    /// A grid's values as a score's digits (10 or fewer) or a choice's codes.
    fn coded(&self, grid: Grid) -> Result<GridSlot, Error> {
        if grid.size <= DIGITS {
            return Ok(GridSlot {
                table: self.digits(grid.size)?,
                numbers: false,
                codes: (0..grid.size).map(|level| level.to_string()).collect(),
            });
        }
        let names: Vec<String> = (0..grid.size).map(|index| index.to_string()).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let (codes, table) = self.aliased(&names)?;
        Ok(GridSlot {
            table,
            numbers: false,
            codes,
        })
    }

    /// How the prompt says a box's edges are written.
    fn edge_legend(&self) -> String {
        if self.edge.numbers {
            return "Each edge is a number from 0 to 999, in thousandths of the picture: top and \
                    bottom from its top edge, left and right from its left edge."
                .into();
        }
        let codes = &self.edge.codes;
        format!(
            "Each edge is a code on a {}-value scale: {} is 0 (top or left), {} is 1 (bottom or \
             right). Values are equally spaced.",
            EDGE_CODES.size,
            codes[0],
            codes[codes.len() - 1]
        )
    }

    /// Each option's distinct one-token code (its native letter, A to Z, or the scheme's code;
    /// else the first free one of a fallback pool), and the table of the codes and their forms
    /// after a space.
    fn aliased(&self, names: &[&str]) -> Result<(Vec<String>, Table), Error> {
        let mut taken: Vec<(String, u32)> = Vec::with_capacity(names.len());
        for code in option_codes(names, self.codes) {
            let mut candidates = std::iter::once(code).chain(fallback_pool());
            let alias = loop {
                let Some(candidate) = candidates.next() else {
                    return Err(Error::Format(format!(
                        "no one-token code left for {} options",
                        names.len()
                    )));
                };
                if let [id] = self.tokenizer.encode(&candidate)?[..]
                    && taken.iter().all(|&(_, used)| used != id)
                {
                    break (candidate, id);
                }
            };
            taken.push(alias);
        }
        let groups = taken.iter().map(|(code, id)| {
            let spaced = self.single(&[&format!(" {code}")])?;
            Ok([*id]
                .into_iter()
                .chain(spaced.into_iter().filter(|spaced| spaced != id))
                .collect())
        });
        let table = Table::new(&groups.collect::<Result<Vec<Vec<u32>>, Error>>()?);
        Ok((taken.into_iter().map(|(code, _)| code).collect(), table))
    }
}

/// What a question's suffix holds before its markers are placed: its text, its markers and
/// what each slot reads.
struct Asked {
    body: String,
    markers: Vec<String>,
    readouts: Vec<Readout>,
}

impl Asked {
    /// A question answered at its suffix's last token.
    fn single(body: String, readout: Readout) -> Self {
        Self {
            body,
            markers: Vec::new(),
            readouts: vec![readout],
        }
    }
}

/// A yes or no, each perhaps described.
fn noul(said: &str, criteria: Option<&Map<String, Value>>) -> Result<Asked, Error> {
    let extra = match criteria.filter(|criteria| !criteria.is_empty()) {
        None => String::new(),
        Some(criteria) => {
            let text = |key| python::text(criteria.get(key).unwrap_or(&Value::Null));
            format!("\nYes: {}\nNo: {}", text("true")?, text("false")?)
        }
    };
    let body = format!("{said}{extra}\n\nReply with yes or no only.");
    Ok(Asked::single(body, Readout::Code(YES_NO)))
}

/// A multiple choice: each option a yes or no at its quoted name.
fn multiple(said: &str, options: &[Criterion]) -> Asked {
    let lines = options.iter().map(|option| {
        let described = option
            .description
            .as_deref()
            .filter(|text| !text.is_empty());
        let text = described.unwrap_or(&option.name);
        format!("{}: {text}", quoted(&option.name))
    });
    let lines = lines.collect::<Vec<_>>().join("\n");
    Asked {
        body: format!("{said}\n\nOptions:\n{lines}\n\nReply with yes or no for each option."),
        markers: options.iter().map(|option| quoted(&option.name)).collect(),
        readouts: vec![Readout::Code(YES_NO); options.len()],
    }
}

/// A read: its start and end in the state.
fn read(said: &str, value: Written, unit: Option<&str>) -> Asked {
    let unit = match (value, unit) {
        (Written::Number, Some(unit)) if !unit.is_empty() => format!(" (unit: {unit})"),
        _ => String::new(),
    };
    let name = value.name();
    Asked {
        body: format!(
            "{said}\n\nRead the {name}{unit} from the state, or nothing if it is absent."
        ),
        markers: vec!["start".into(), "end".into()],
        readouts: vec![Readout::Pointer; 2],
    }
}

impl Tables {
    fn add(&mut self, table: Table) -> Readout {
        self.0.push(table);
        Readout::Code(self.0.len() - 1)
    }
}

/// The state as the block before `QUESTION:`: a string as it is, any other value as indented
/// JSON, then a blank line.
pub fn state_block(state: &Value) -> String {
    match state {
        Value::String(text) => format!("{text}\n\n"),
        other => format!("{}\n\n", python::dumps(other, Layout::Indented(2))),
    }
}

/// A name as `json.dumps` quotes it.
fn quoted(name: &str) -> String {
    python::dumps(&Value::String(name.to_owned()), Layout::Inline)
}

fn letter(index: usize) -> char {
    char::from(b'A' + index as u8)
}

/// Native letters when every label is one, else A to Z, else the scheme's codes.
fn option_codes(labels: &[&str], codes: Codes) -> Vec<String> {
    let labels: Vec<&str> = labels.iter().map(|label| python::strip(label)).collect();
    let native = labels.iter().all(|label| {
        let mut chars = label.chars();
        chars.next().is_some_and(char::is_alphabetic) && chars.next().is_none()
    });
    if native && !labels.is_empty() {
        return labels.iter().map(|label| (*label).to_owned()).collect();
    }
    let count = labels.len();
    if count <= LETTERS || codes == Codes::Letters {
        let singles = (0..LETTERS).map(|index| letter(index).to_string());
        let pairs = (0..LETTERS * LETTERS)
            .map(|index| format!("{}{}", letter(index / LETTERS), letter(index % LETTERS)));
        return singles.chain(pairs).take(count).collect();
    }
    (0..count).map(|index| format!("{index:02}")).collect()
}

/// The codes tried when an option's own takes several tokens or is taken: capitals, two-digit
/// numbers, small letters, `#0` to `#199`, then two capitals.
fn fallback_pool() -> impl Iterator<Item = String> {
    let capitals = (0..LETTERS).map(|index| letter(index).to_string());
    let numbers = (0..100).map(|number| format!("{number:02}"));
    let small = (0..LETTERS).map(|index| letter(index).to_ascii_lowercase().to_string());
    let hashed = (0..200).map(|number| format!("#{number}"));
    let pairs = (0..LETTERS * LETTERS)
        .map(|index| format!("{}{}", letter(index / LETTERS), letter(index % LETTERS)));
    capitals
        .chain(numbers)
        .chain(small)
        .chain(hashed)
        .chain(pairs)
}

/// `value` as Python's `format(value, "g")` writes it: six significant digits, trailing zeros
/// dropped, in scientific notation below 1e-4 or from 1e6.
pub fn general(value: f64) -> String {
    const SIGNIFICANT: i32 = 6;
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("`{:e}` writes an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` writes an integer exponent");
    if (-4..SIGNIFICANT).contains(&exponent) {
        let decimals = (SIGNIFICANT - 1 - exponent).max(0) as usize;
        return trimmed(&format!("{value:.decimals$}")).to_owned();
    }
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{}e{sign}{:02}", trimmed(mantissa), exponent.abs())
}

/// A decimal without its trailing zeros, nor its point when nothing follows it.
fn trimmed(decimal: &str) -> &str {
    if decimal.contains('.') {
        decimal.trim_end_matches('0').trim_end_matches('.')
    } else {
        decimal
    }
}

#[cfg(test)]
mod tests {
    use super::general;

    #[test]
    fn numbers_are_written_as_pythons_general_format() {
        let cases = [
            (0.0, "0"),
            (1.0, "1"),
            (0.5, "0.5"),
            (0.001, "0.001"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (100_000.0, "100000"),
            (1_000_000.0, "1e+06"),
            (1_234_567.0, "1.23457e+06"),
            (-2.5, "-2.5"),
            (0.999, "0.999"),
            (1.0 / 127.0, "0.00787402"),
        ];
        for (value, written) in cases {
            assert_eq!(general(value), written, "{value}");
        }
    }
}
