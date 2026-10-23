//! JSON values as the reference writes them into the prompt: Python's `json.dumps(value,
//! ensure_ascii=False, indent=n)`, and `str(value)` of a scalar. Floats are written as Python's
//! `repr` writes them (`1e-05`, `1e+16`, `100.0`), integers as their digits.

use serde_json::{Number, Value};

use crate::Error;

/// Python's `float.repr` switches to scientific notation outside `1e-4 ≤ |x| < 1e16`.
const FIXED_EXPONENTS: std::ops::Range<i32> = -4..16;

/// `value` as `json.dumps(value, ensure_ascii=False, indent=indent)` writes it.
pub(crate) fn dumps(value: &Value, indent: usize) -> String {
    let mut out = String::new();
    write(&mut out, value, indent, 0);
    out
}

/// A description or an instruction: a string as it is, a scalar as Python's `str` writes it.
pub(crate) fn text(value: &Value) -> Result<String, Error> {
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

fn write(out: &mut String, value: &Value, indent: usize, depth: usize) {
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
                open(out, index, indent, depth + 1);
                write(out, item, indent, depth + 1);
            }
            close(out, ']', indent, depth);
        }
        Value::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                open(out, index, indent, depth + 1);
                out.push_str(&quoted(key));
                out.push_str(": ");
                write(out, item, indent, depth + 1);
            }
            close(out, '}', indent, depth);
        }
    }
}

/// The separator before item `index` of a container, then the item's indentation.
fn open(out: &mut String, index: usize, indent: usize, depth: usize) {
    if index > 0 {
        out.push(',');
    }
    out.push('\n');
    out.push_str(&" ".repeat(indent * depth));
}

fn close(out: &mut String, bracket: char, indent: usize, depth: usize) {
    out.push('\n');
    out.push_str(&" ".repeat(indent * depth));
    out.push(bracket);
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
