//! Text as Python references write it into prompts: `json.dumps(value, ensure_ascii=False)`,
//! with or without `indent`, `str(value)` of a scalar, and `str.strip()`. Floats are written as
//! Python's `repr` writes them (`1e-05`, `1e+16`, `100.0`), integers as their digits.

use serde_json::{Number, Value};

use crate::Error;

/// Python's `float.repr` switches to scientific notation outside `1e-4 ≤ |x| < 1e16`.
const FIXED_EXPONENTS: std::ops::Range<i32> = -4..16;

/// How `json.dumps` lays out containers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// On one line, items after `, ` and values after `: ` (no `indent`).
    Inline,
    /// On one line, items after `,` and values after `:` (`separators=(",", ":")`).
    Compact,
    /// One item per line, indented by this many spaces per level (`indent=n`).
    Indented(usize),
}

impl Layout {
    fn key_separator(self) -> &'static str {
        match self {
            Layout::Compact => ":",
            Layout::Inline | Layout::Indented(_) => ": ",
        }
    }
}

/// `value` as `json.dumps(value, ensure_ascii=False)` writes it, laid out as `layout`.
pub fn dumps(value: &Value, layout: Layout) -> String {
    let mut out = String::new();
    write(&mut out, value, layout, 0);
    out
}

/// A description or an instruction: a string as it is, a scalar as Python's `str` writes it.
///
/// # Errors
/// When `value` is an array or an object.
pub fn text(value: &Value) -> Result<String, Error> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Number(number) => Ok(self::number(number)),
        Value::Bool(true) => Ok("True".into()),
        Value::Bool(false) => Ok("False".into()),
        Value::Null => Ok("None".into()),
        Value::Array(_) | Value::Object(_) => Err(Error::Request(
            "instructions and option descriptions must be text, numbers or booleans".into(),
        )),
    }
}

/// `text` without the leading and trailing characters `isspace` matches (`str.strip()`, Jinja's
/// `trim`).
pub fn strip(text: &str) -> &str {
    text.trim_matches(isspace)
}

/// Python's `str.isspace` of one character: Unicode's whitespace and the information separators
/// U+001C–U+001F.
pub fn isspace(character: char) -> bool {
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

fn write(out: &mut String, value: &Value, layout: Layout, depth: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&self::number(number)),
        Value::String(text) => out.push_str(&quoted(text)),
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Object(entries) if entries.is_empty() => out.push_str("{}"),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                open(out, index, layout, depth + 1);
                write(out, item, layout, depth + 1);
            }
            close(out, ']', layout, depth);
        }
        Value::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                open(out, index, layout, depth + 1);
                out.push_str(&quoted(key));
                out.push_str(layout.key_separator());
                write(out, item, layout, depth + 1);
            }
            close(out, '}', layout, depth);
        }
    }
}

/// The separator before item `index` of a container, then the item's indentation.
fn open(out: &mut String, index: usize, layout: Layout, depth: usize) {
    if index > 0 {
        out.push(',');
    }
    match layout {
        Layout::Inline if index > 0 => out.push(' '),
        Layout::Inline | Layout::Compact => {}
        Layout::Indented(indent) => line(out, indent * depth),
    }
}

fn close(out: &mut String, bracket: char, layout: Layout, depth: usize) {
    if let Layout::Indented(indent) = layout {
        line(out, indent * depth);
    }
    out.push(bracket);
}

fn line(out: &mut String, spaces: usize) {
    out.push('\n');
    out.push_str(&" ".repeat(spaces));
}

/// A JSON string: Python escapes `"`, `\` and the control characters as `serde_json` does.
fn quoted(text: &str) -> String {
    serde_json::to_string(text).expect("a string serializes")
}

fn number(number: &Number) -> String {
    match number.as_f64() {
        Some(float) if number.is_f64() => repr(float),
        _ => number.to_string(),
    }
}

/// Python's `repr` of a float: its shortest round-trip digits, in fixed notation (with at least
/// one decimal) within `FIXED_EXPONENTS`, else as `d.ddde±XX`.
fn repr(value: f64) -> String {
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("`{:e}` writes an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` writes an integer exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(magnitude) => ("-", magnitude),
        None => ("", mantissa),
    };
    let digits = mantissa.replace('.', "");
    if !FIXED_EXPONENTS.contains(&exponent) {
        let (first, rest) = digits.split_at(1);
        let point = if rest.is_empty() { "" } else { "." };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        return format!(
            "{sign}{first}{point}{rest}e{exponent_sign}{:02}",
            exponent.abs()
        );
    }
    let whole = exponent + 1;
    let fixed = match usize::try_from(whole) {
        Err(_) | Ok(0) => format!("0.{}{digits}", "0".repeat(whole.unsigned_abs() as usize)),
        Ok(whole) if whole >= digits.len() => {
            format!("{digits}{}.0", "0".repeat(whole - digits.len()))
        }
        Ok(whole) => format!("{}.{}", &digits[..whole], &digits[whole..]),
    };
    format!("{sign}{fixed}")
}
