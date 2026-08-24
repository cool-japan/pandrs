//! Scalar operator semantics shared by every query evaluation path.
//!
//! Both the row-by-row interpreter ([`super::evaluator::Evaluator`]) and the
//! column-at-a-time engine ([`super::vectorized`]) apply exactly these rules, so
//! the two paths cannot drift apart. Every rule lives here once.
//!
//! # Missing values
//!
//! String columns are the only place where a missing value can appear in this
//! DataFrame representation (a `Series<T>` is dense). The spellings recognised
//! as "missing" are converted to `f64::NAN` rather than to `0`, so a missing
//! cell never satisfies a comparison and never contributes a fabricated value to
//! arithmetic.

use super::ast::{BinaryOp, LiteralValue, UnaryOp};
use crate::core::error::{Error, Result};

/// Textual spellings that denote a missing value in a string column.
///
/// Note that this only applies when a *numeric* interpretation of the cell is
/// required. String-to-string comparisons always compare the text verbatim.
pub(crate) fn is_missing_text(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("na")
        || trimmed.eq_ignore_ascii_case("n/a")
        || trimmed.eq_ignore_ascii_case("nan")
        || trimmed.eq_ignore_ascii_case("null")
        || trimmed.eq_ignore_ascii_case("none")
}

/// Interpret a text cell as a number.
///
/// Returns `Some(f64::NAN)` for missing markers (so comparisons yield `false`
/// and arithmetic propagates the missing value) and `None` when the text simply
/// is not numeric, which is a genuine type error.
pub(crate) fn text_to_number(value: &str) -> Option<f64> {
    if is_missing_text(value) {
        return Some(f64::NAN);
    }
    value.trim().parse::<f64>().ok()
}

/// Interpret a text cell as a boolean (`true`/`false`, any capitalisation).
pub(crate) fn text_to_bool(value: &str) -> Option<bool> {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("true") {
        Some(true)
    } else if trimmed.eq_ignore_ascii_case("false") {
        Some(false)
    } else {
        None
    }
}

/// Error raised when a text cell cannot be read as a number.
pub(crate) fn non_numeric_text_error(value: &str) -> Error {
    Error::InvalidValue(format!(
        "Cannot use non-numeric value '{}' in a numeric operation",
        value
    ))
}

/// Error raised when a text cell cannot be read as a boolean.
pub(crate) fn non_boolean_text_error(value: &str) -> Error {
    Error::InvalidValue(format!(
        "Cannot use value '{}' as a boolean (expected 'true' or 'false')",
        value
    ))
}

/// Apply a binary operator to two numbers.
pub(crate) fn number_binary(left: f64, op: &BinaryOp, right: f64) -> Result<LiteralValue> {
    match op {
        BinaryOp::Add => Ok(LiteralValue::Number(left + right)),
        BinaryOp::Subtract => Ok(LiteralValue::Number(left - right)),
        BinaryOp::Multiply => Ok(LiteralValue::Number(left * right)),
        BinaryOp::Divide => {
            if right == 0.0 {
                Err(Error::InvalidValue("Division by zero".to_string()))
            } else {
                Ok(LiteralValue::Number(left / right))
            }
        }
        BinaryOp::Modulo => Ok(LiteralValue::Number(left % right)),
        BinaryOp::Power => Ok(LiteralValue::Number(left.powf(right))),
        BinaryOp::Equal => Ok(LiteralValue::Boolean(numbers_equal(left, right))),
        BinaryOp::NotEqual => Ok(LiteralValue::Boolean(!numbers_equal(left, right))),
        BinaryOp::LessThan => Ok(LiteralValue::Boolean(left < right)),
        BinaryOp::LessThanOrEqual => Ok(LiteralValue::Boolean(left <= right)),
        BinaryOp::GreaterThan => Ok(LiteralValue::Boolean(left > right)),
        BinaryOp::GreaterThanOrEqual => Ok(LiteralValue::Boolean(left >= right)),
        BinaryOp::And | BinaryOp::Or => Err(Error::InvalidValue(
            "Logical operations require boolean operands".to_string(),
        )),
    }
}

/// Numeric equality with a tolerance, matching the historical query semantics.
///
/// `NaN` (a missing value) compares unequal to everything, including itself.
#[inline]
pub(crate) fn numbers_equal(left: f64, right: f64) -> bool {
    (left - right).abs() < f64::EPSILON
}

/// Apply a binary operator to two strings.
pub(crate) fn string_binary(left: &str, op: &BinaryOp, right: &str) -> Result<LiteralValue> {
    match op {
        BinaryOp::Equal => Ok(LiteralValue::Boolean(left == right)),
        BinaryOp::NotEqual => Ok(LiteralValue::Boolean(left != right)),
        BinaryOp::LessThan => Ok(LiteralValue::Boolean(left < right)),
        BinaryOp::LessThanOrEqual => Ok(LiteralValue::Boolean(left <= right)),
        BinaryOp::GreaterThan => Ok(LiteralValue::Boolean(left > right)),
        BinaryOp::GreaterThanOrEqual => Ok(LiteralValue::Boolean(left >= right)),
        BinaryOp::Add => Ok(LiteralValue::String(format!("{}{}", left, right))),
        _ => Err(Error::InvalidValue(
            "Unsupported operation for strings".to_string(),
        )),
    }
}

/// Apply a binary operator to two booleans.
pub(crate) fn bool_binary(left: bool, op: &BinaryOp, right: bool) -> Result<LiteralValue> {
    match op {
        BinaryOp::And => Ok(LiteralValue::Boolean(left && right)),
        BinaryOp::Or => Ok(LiteralValue::Boolean(left || right)),
        BinaryOp::Equal => Ok(LiteralValue::Boolean(left == right)),
        BinaryOp::NotEqual => Ok(LiteralValue::Boolean(left != right)),
        _ => Err(Error::InvalidValue(
            "Unsupported operation for booleans".to_string(),
        )),
    }
}

/// Apply a binary operation to two scalar values.
///
/// Mixed operands are coerced exactly once, here:
/// * number/string  - the string side is read as a number (missing markers
///   become `NaN`); non-numeric text is an error.
/// * boolean/string - the string side is read as a boolean.
/// * boolean/number - rejected; there is no meaningful implicit conversion.
pub(crate) fn apply_binary(
    left: &LiteralValue,
    op: &BinaryOp,
    right: &LiteralValue,
) -> Result<LiteralValue> {
    match (left, right) {
        (LiteralValue::Number(l), LiteralValue::Number(r)) => number_binary(*l, op, *r),
        (LiteralValue::String(l), LiteralValue::String(r)) => string_binary(l, op, r),
        (LiteralValue::Boolean(l), LiteralValue::Boolean(r)) => bool_binary(*l, op, *r),

        (LiteralValue::Number(l), LiteralValue::String(r)) => {
            let r = text_to_number(r).ok_or_else(|| non_numeric_text_error(r))?;
            number_binary(*l, op, r)
        }
        (LiteralValue::String(l), LiteralValue::Number(r)) => {
            let l = text_to_number(l).ok_or_else(|| non_numeric_text_error(l))?;
            number_binary(l, op, *r)
        }

        (LiteralValue::Boolean(l), LiteralValue::String(r)) => {
            let r = text_to_bool(r).ok_or_else(|| non_boolean_text_error(r))?;
            bool_binary(*l, op, r)
        }
        (LiteralValue::String(l), LiteralValue::Boolean(r)) => {
            let l = text_to_bool(l).ok_or_else(|| non_boolean_text_error(l))?;
            bool_binary(l, op, *r)
        }

        (LiteralValue::Boolean(_), LiteralValue::Number(_))
        | (LiteralValue::Number(_), LiteralValue::Boolean(_)) => Err(Error::InvalidValue(
            "Cannot combine a boolean and a number; compare against true/false instead".to_string(),
        )),
    }
}

/// Apply a unary operation to a scalar value.
pub(crate) fn apply_unary(op: &UnaryOp, operand: &LiteralValue) -> Result<LiteralValue> {
    match (op, operand) {
        (UnaryOp::Not, LiteralValue::Boolean(b)) => Ok(LiteralValue::Boolean(!b)),
        (UnaryOp::Not, LiteralValue::String(s)) => {
            let b = text_to_bool(s).ok_or_else(|| non_boolean_text_error(s))?;
            Ok(LiteralValue::Boolean(!b))
        }
        (UnaryOp::Negate, LiteralValue::Number(n)) => Ok(LiteralValue::Number(-n)),
        (UnaryOp::Negate, LiteralValue::String(s)) => {
            let n = text_to_number(s).ok_or_else(|| non_numeric_text_error(s))?;
            Ok(LiteralValue::Number(-n))
        }
        (UnaryOp::Not, LiteralValue::Number(_)) => Err(Error::InvalidValue(
            "Logical NOT requires a boolean operand".to_string(),
        )),
        (UnaryOp::Negate, LiteralValue::Boolean(_)) => Err(Error::InvalidValue(
            "Cannot negate a boolean value".to_string(),
        )),
    }
}

/// Interpret a scalar value as a mask entry.
pub(crate) fn value_to_bool(value: &LiteralValue) -> Result<bool> {
    match value {
        LiteralValue::Boolean(b) => Ok(*b),
        LiteralValue::String(s) => text_to_bool(s).ok_or_else(|| {
            Error::InvalidValue("Query expression must evaluate to boolean".to_string())
        }),
        LiteralValue::Number(_) => Err(Error::InvalidValue(
            "Query expression must evaluate to boolean".to_string(),
        )),
    }
}
