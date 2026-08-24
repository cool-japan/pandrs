//! Column-at-a-time evaluation of query expressions.
//!
//! The evaluator walks the expression tree once and produces a whole column of
//! results at each node, instead of walking the tree once per row. Column data
//! is fetched (and type-classified) a single time per query and shared between
//! nodes through an `Rc`, so a predicate such as `a > 1 && a < 10` reads the
//! column once.
//!
//! Scalar semantics (coercions, comparison rules, error messages) are *not*
//! duplicated here: every element-level decision is delegated to
//! [`super::ops`], which the row-by-row interpreter uses as well. That keeps the
//! two evaluation paths from drifting apart.
//!
//! No machine code is generated. This is a native, vectorized interpreter.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::ast::{BinaryOp, Expr, LiteralValue, UnaryOp};
use super::evaluator::QueryContext;
use super::ops;
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;

/// One evaluated value per row, or a single scalar broadcast over every row.
#[derive(Debug, Clone)]
pub(crate) enum ValueVec {
    /// Numeric values; `NaN` marks a missing value.
    Num(Vec<f64>),
    /// Boolean values.
    Bool(Vec<bool>),
    /// Text values.
    Str(Vec<String>),
    /// A single value that applies to every row.
    Scalar(LiteralValue),
}

impl ValueVec {
    /// Number of rows this value covers, or `None` for a broadcast scalar.
    fn len(&self) -> Option<usize> {
        match self {
            ValueVec::Num(v) => Some(v.len()),
            ValueVec::Bool(v) => Some(v.len()),
            ValueVec::Str(v) => Some(v.len()),
            ValueVec::Scalar(_) => None,
        }
    }

    /// The value at `idx` as a scalar literal.
    ///
    /// The row-by-row interpreter reads columns through this, so both tiers see
    /// exactly the same value for a given cell.
    pub(crate) fn cell_at(&self, idx: usize) -> Result<LiteralValue> {
        let out_of_bounds = |size: usize| Error::IndexOutOfBounds { index: idx, size };
        match self {
            ValueVec::Num(values) => values
                .get(idx)
                .map(|value| LiteralValue::Number(*value))
                .ok_or_else(|| out_of_bounds(values.len())),
            ValueVec::Bool(values) => values
                .get(idx)
                .map(|value| LiteralValue::Boolean(*value))
                .ok_or_else(|| out_of_bounds(values.len())),
            ValueVec::Str(values) => values
                .get(idx)
                .map(|value| LiteralValue::String(value.clone()))
                .ok_or_else(|| out_of_bounds(values.len())),
            ValueVec::Scalar(value) => Ok(value.clone()),
        }
    }
}

/// The element domain an operand belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Domain {
    Number,
    Boolean,
    Text,
}

fn domain_of(value: &ValueVec) -> Domain {
    match value {
        ValueVec::Num(_) | ValueVec::Scalar(LiteralValue::Number(_)) => Domain::Number,
        ValueVec::Bool(_) | ValueVec::Scalar(LiteralValue::Boolean(_)) => Domain::Boolean,
        ValueVec::Str(_) | ValueVec::Scalar(LiteralValue::String(_)) => Domain::Text,
    }
}

/// A numeric operand: either one value per row or a broadcast scalar.
enum NumOperand<'v> {
    Column(Cow<'v, [f64]>),
    Scalar(f64),
}

impl NumOperand<'_> {
    #[inline]
    fn at(&self, idx: usize) -> f64 {
        match self {
            NumOperand::Column(values) => values[idx],
            NumOperand::Scalar(value) => *value,
        }
    }

    fn scalar(&self) -> Option<f64> {
        match self {
            NumOperand::Scalar(value) => Some(*value),
            NumOperand::Column(_) => None,
        }
    }
}

/// A boolean operand: either one value per row or a broadcast scalar.
enum BoolOperand<'v> {
    Column(Cow<'v, [bool]>),
    Scalar(bool),
}

impl BoolOperand<'_> {
    #[inline]
    fn at(&self, idx: usize) -> bool {
        match self {
            BoolOperand::Column(values) => values[idx],
            BoolOperand::Scalar(value) => *value,
        }
    }

    fn scalar(&self) -> Option<bool> {
        match self {
            BoolOperand::Scalar(value) => Some(*value),
            BoolOperand::Column(_) => None,
        }
    }
}

/// A text operand: either one value per row or a broadcast scalar.
enum StrOperand<'v> {
    Column(&'v [String]),
    Scalar(&'v str),
}

impl StrOperand<'_> {
    #[inline]
    fn at(&self, idx: usize) -> &str {
        match self {
            StrOperand::Column(values) => values[idx].as_str(),
            StrOperand::Scalar(value) => value,
        }
    }

    fn scalar(&self) -> Option<&str> {
        match self {
            StrOperand::Scalar(value) => Some(value),
            StrOperand::Column(_) => None,
        }
    }
}

fn length_error(expected: usize, found: usize) -> Error {
    Error::InconsistentRowCount { expected, found }
}

/// Column-at-a-time expression evaluator.
pub(crate) struct VectorizedEvaluator<'a> {
    dataframe: &'a DataFrame,
    context: &'a QueryContext,
    row_count: usize,
    /// Column data, loaded and type-classified once per query.
    column_cache: RefCell<HashMap<String, Rc<ValueVec>>>,
}

impl<'a> VectorizedEvaluator<'a> {
    pub(crate) fn new(dataframe: &'a DataFrame, context: &'a QueryContext) -> Self {
        Self {
            dataframe,
            context,
            row_count: dataframe.row_count(),
            column_cache: RefCell::new(HashMap::new()),
        }
    }

    /// Evaluate `expr` and return one boolean per row.
    pub(crate) fn evaluate_mask(&self, expr: &Expr) -> Result<Vec<bool>> {
        let value = self.evaluate(expr)?;
        self.to_mask(&value)
    }

    /// Evaluate `expr` into a column of values.
    pub(crate) fn evaluate(&self, expr: &Expr) -> Result<Rc<ValueVec>> {
        match expr {
            Expr::Literal(value) => Ok(Rc::new(ValueVec::Scalar(value.clone()))),

            Expr::Column(name) => self.resolve_identifier(name),

            Expr::Variable(name) => match self.context.variables.get(name) {
                Some(value) => Ok(Rc::new(ValueVec::Scalar(value.clone()))),
                None => Err(Error::InvalidValue(format!(
                    "Undefined query variable '@{}'",
                    name
                ))),
            },

            Expr::Binary { left, op, right } => {
                // Short-circuit the logical operators on constant operands so a
                // `false && <expensive>` does not evaluate the right-hand side.
                if let Some(result) = self.short_circuit(left, op, right)? {
                    return Ok(result);
                }
                let left_value = self.evaluate(left)?;
                let right_value = self.evaluate(right)?;
                Ok(Rc::new(self.binary(&left_value, op, &right_value)?))
            }

            Expr::Unary { op, operand } => {
                let value = self.evaluate(operand)?;
                Ok(Rc::new(self.unary(op, &value)?))
            }

            Expr::Function { name, args } => Ok(Rc::new(self.function(name, args)?)),
        }
    }

    /// Resolve a bare identifier: a column first, then a context variable.
    fn resolve_identifier(&self, name: &str) -> Result<Rc<ValueVec>> {
        if let Some(cached) = self.column_cache.borrow().get(name) {
            return Ok(Rc::clone(cached));
        }

        if !self.dataframe.contains_column(name) {
            // Fall back to a context variable so `set_variable` works without
            // the `@` sigil, matching the historical documented behaviour.
            if let Some(value) = self.context.variables.get(name) {
                return Ok(Rc::new(ValueVec::Scalar(value.clone())));
            }
            return Err(Error::ColumnNotFound(name.to_string()));
        }

        let loaded = Rc::new(load_column_values(self.dataframe, name)?);
        self.column_cache
            .borrow_mut()
            .insert(name.to_string(), Rc::clone(&loaded));
        Ok(loaded)
    }

    /// Constant-operand short-circuiting for `&&` / `||`.
    fn short_circuit(
        &self,
        left: &Expr,
        op: &BinaryOp,
        right: &Expr,
    ) -> Result<Option<Rc<ValueVec>>> {
        let literal = match (op, left) {
            (BinaryOp::And, Expr::Literal(LiteralValue::Boolean(value)))
            | (BinaryOp::Or, Expr::Literal(LiteralValue::Boolean(value))) => *value,
            _ => return Ok(None),
        };

        match (op, literal) {
            // `false && x` is false without touching x; `true || x` is true.
            (BinaryOp::And, false) => Ok(Some(Rc::new(ValueVec::Scalar(LiteralValue::Boolean(
                false,
            ))))),
            (BinaryOp::Or, true) => {
                Ok(Some(Rc::new(ValueVec::Scalar(LiteralValue::Boolean(true)))))
            }
            // `true && x` / `false || x` reduce to x, but x must still be a
            // valid boolean expression, so evaluate and validate it.
            _ => {
                let value = self.evaluate(right)?;
                let mask = self.to_mask(&value)?;
                Ok(Some(Rc::new(ValueVec::Bool(mask))))
            }
        }
    }

    /// Apply a binary operator to two evaluated columns.
    fn binary(&self, left: &ValueVec, op: &BinaryOp, right: &ValueVec) -> Result<ValueVec> {
        self.check_len(left)?;
        self.check_len(right)?;

        match (domain_of(left), domain_of(right)) {
            (Domain::Number, Domain::Number)
            | (Domain::Number, Domain::Text)
            | (Domain::Text, Domain::Number) => {
                let left = self.to_num_operand(left)?;
                let right = self.to_num_operand(right)?;
                self.numeric_binary(&left, op, &right)
            }
            (Domain::Text, Domain::Text) => {
                let left = to_str_operand(left)?;
                let right = to_str_operand(right)?;
                self.text_binary(&left, op, &right)
            }
            (Domain::Boolean, Domain::Boolean)
            | (Domain::Boolean, Domain::Text)
            | (Domain::Text, Domain::Boolean) => {
                let left = self.to_bool_operand(left)?;
                let right = self.to_bool_operand(right)?;
                self.bool_binary(&left, op, &right)
            }
            (Domain::Boolean, Domain::Number) | (Domain::Number, Domain::Boolean) => {
                Err(Error::InvalidValue(
                    "Cannot combine a boolean and a number; compare against true/false instead"
                        .to_string(),
                ))
            }
        }
    }

    fn numeric_binary(
        &self,
        left: &NumOperand<'_>,
        op: &BinaryOp,
        right: &NumOperand<'_>,
    ) -> Result<ValueVec> {
        if let (Some(l), Some(r)) = (left.scalar(), right.scalar()) {
            return Ok(ValueVec::Scalar(ops::number_binary(l, op, r)?));
        }

        let n = self.row_count;
        match op {
            BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::LessThan
            | BinaryOp::LessThanOrEqual
            | BinaryOp::GreaterThan
            | BinaryOp::GreaterThanOrEqual => {
                let mut out = Vec::with_capacity(n);
                for idx in 0..n {
                    let (l, r) = (left.at(idx), right.at(idx));
                    out.push(match op {
                        BinaryOp::Equal => ops::numbers_equal(l, r),
                        BinaryOp::NotEqual => !ops::numbers_equal(l, r),
                        BinaryOp::LessThan => l < r,
                        BinaryOp::LessThanOrEqual => l <= r,
                        BinaryOp::GreaterThan => l > r,
                        _ => l >= r,
                    });
                }
                Ok(ValueVec::Bool(out))
            }
            BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Power => {
                let mut out = Vec::with_capacity(n);
                for idx in 0..n {
                    let (l, r) = (left.at(idx), right.at(idx));
                    out.push(match op {
                        BinaryOp::Add => l + r,
                        BinaryOp::Subtract => l - r,
                        BinaryOp::Multiply => l * r,
                        _ => l.powf(r),
                    });
                }
                Ok(ValueVec::Num(out))
            }
            BinaryOp::Divide | BinaryOp::Modulo => {
                let mut out = Vec::with_capacity(n);
                for idx in 0..n {
                    let (l, r) = (left.at(idx), right.at(idx));
                    // Division semantics live in one place; this also rejects a
                    // zero divisor identically to the interpreter.
                    match ops::number_binary(l, op, r)? {
                        LiteralValue::Number(value) => out.push(value),
                        other => {
                            return Err(Error::InvalidValue(format!(
                                "Arithmetic produced a non-numeric result: {:?}",
                                other
                            )))
                        }
                    }
                }
                Ok(ValueVec::Num(out))
            }
            BinaryOp::And | BinaryOp::Or => Err(Error::InvalidValue(
                "Logical operations require boolean operands".to_string(),
            )),
        }
    }

    fn text_binary(
        &self,
        left: &StrOperand<'_>,
        op: &BinaryOp,
        right: &StrOperand<'_>,
    ) -> Result<ValueVec> {
        if let (Some(l), Some(r)) = (left.scalar(), right.scalar()) {
            return Ok(ValueVec::Scalar(ops::string_binary(l, op, r)?));
        }

        let n = self.row_count;
        match op {
            BinaryOp::Add => {
                let mut out = Vec::with_capacity(n);
                for idx in 0..n {
                    out.push(format!("{}{}", left.at(idx), right.at(idx)));
                }
                Ok(ValueVec::Str(out))
            }
            BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::LessThan
            | BinaryOp::LessThanOrEqual
            | BinaryOp::GreaterThan
            | BinaryOp::GreaterThanOrEqual => {
                let mut out = Vec::with_capacity(n);
                for idx in 0..n {
                    let (l, r) = (left.at(idx), right.at(idx));
                    out.push(match op {
                        BinaryOp::Equal => l == r,
                        BinaryOp::NotEqual => l != r,
                        BinaryOp::LessThan => l < r,
                        BinaryOp::LessThanOrEqual => l <= r,
                        BinaryOp::GreaterThan => l > r,
                        _ => l >= r,
                    });
                }
                Ok(ValueVec::Bool(out))
            }
            _ => Err(Error::InvalidValue(
                "Unsupported operation for strings".to_string(),
            )),
        }
    }

    fn bool_binary(
        &self,
        left: &BoolOperand<'_>,
        op: &BinaryOp,
        right: &BoolOperand<'_>,
    ) -> Result<ValueVec> {
        if let (Some(l), Some(r)) = (left.scalar(), right.scalar()) {
            return Ok(ValueVec::Scalar(ops::bool_binary(l, op, r)?));
        }

        let n = self.row_count;
        match op {
            BinaryOp::And | BinaryOp::Or | BinaryOp::Equal | BinaryOp::NotEqual => {
                let mut out = Vec::with_capacity(n);
                for idx in 0..n {
                    let (l, r) = (left.at(idx), right.at(idx));
                    out.push(match op {
                        BinaryOp::And => l && r,
                        BinaryOp::Or => l || r,
                        BinaryOp::Equal => l == r,
                        _ => l != r,
                    });
                }
                Ok(ValueVec::Bool(out))
            }
            _ => Err(Error::InvalidValue(
                "Unsupported operation for booleans".to_string(),
            )),
        }
    }

    fn unary(&self, op: &UnaryOp, value: &ValueVec) -> Result<ValueVec> {
        self.check_len(value)?;

        match op {
            UnaryOp::Not => {
                if domain_of(value) == Domain::Number {
                    return Err(Error::InvalidValue(
                        "Logical NOT requires a boolean operand".to_string(),
                    ));
                }
                let operand = self.to_bool_operand(value)?;
                if let Some(scalar) = operand.scalar() {
                    return Ok(ValueVec::Scalar(LiteralValue::Boolean(!scalar)));
                }
                let mut out = Vec::with_capacity(self.row_count);
                for idx in 0..self.row_count {
                    out.push(!operand.at(idx));
                }
                Ok(ValueVec::Bool(out))
            }
            UnaryOp::Negate => {
                if domain_of(value) == Domain::Boolean {
                    return Err(Error::InvalidValue(
                        "Cannot negate a boolean value".to_string(),
                    ));
                }
                let operand = self.to_num_operand(value)?;
                if let Some(scalar) = operand.scalar() {
                    return Ok(ValueVec::Scalar(LiteralValue::Number(-scalar)));
                }
                let mut out = Vec::with_capacity(self.row_count);
                for idx in 0..self.row_count {
                    out.push(-operand.at(idx));
                }
                Ok(ValueVec::Num(out))
            }
        }
    }

    fn function(&self, name: &str, args: &[Expr]) -> Result<ValueVec> {
        let func = self
            .context
            .functions
            .get(name)
            .ok_or_else(|| Error::InvalidValue(format!("Unknown function: {}", name)))?;

        let evaluated: Vec<Rc<ValueVec>> = args
            .iter()
            .map(|arg| self.evaluate(arg))
            .collect::<Result<Vec<_>>>()?;

        let operands: Vec<NumOperand<'_>> = evaluated
            .iter()
            .map(|value| {
                if domain_of(value) == Domain::Boolean {
                    return Err(Error::InvalidValue(
                        "Function arguments must be numeric".to_string(),
                    ));
                }
                self.check_len(value)?;
                self.to_num_operand(value)
            })
            .collect::<Result<Vec<_>>>()?;

        // One reusable argument buffer instead of one allocation per row.
        let mut buffer = vec![0.0f64; operands.len()];

        if operands.iter().all(|operand| operand.scalar().is_some()) {
            for (slot, operand) in buffer.iter_mut().zip(operands.iter()) {
                *slot = operand.at(0);
            }
            return Ok(ValueVec::Scalar(LiteralValue::Number(func(&buffer))));
        }

        let mut out = Vec::with_capacity(self.row_count);
        for idx in 0..self.row_count {
            for (slot, operand) in buffer.iter_mut().zip(operands.iter()) {
                *slot = operand.at(idx);
            }
            out.push(func(&buffer));
        }
        Ok(ValueVec::Num(out))
    }

    /// Convert an evaluated value into a filtering mask.
    pub(crate) fn to_mask(&self, value: &ValueVec) -> Result<Vec<bool>> {
        self.check_len(value)?;
        match value {
            ValueVec::Bool(values) => Ok(values.clone()),
            ValueVec::Scalar(LiteralValue::Boolean(flag)) => Ok(vec![*flag; self.row_count]),
            ValueVec::Str(values) => values
                .iter()
                .map(|text| {
                    ops::text_to_bool(text).ok_or_else(|| {
                        Error::InvalidValue("Query expression must evaluate to boolean".to_string())
                    })
                })
                .collect(),
            ValueVec::Scalar(LiteralValue::String(text)) => match ops::text_to_bool(text) {
                Some(flag) => Ok(vec![flag; self.row_count]),
                None => Err(Error::InvalidValue(
                    "Query expression must evaluate to boolean".to_string(),
                )),
            },
            ValueVec::Num(_) | ValueVec::Scalar(LiteralValue::Number(_)) => Err(
                Error::InvalidValue("Query expression must evaluate to boolean".to_string()),
            ),
        }
    }

    fn check_len(&self, value: &ValueVec) -> Result<()> {
        match value.len() {
            Some(len) if len != self.row_count => Err(length_error(self.row_count, len)),
            _ => Ok(()),
        }
    }

    fn to_num_operand<'v>(&self, value: &'v ValueVec) -> Result<NumOperand<'v>> {
        match value {
            ValueVec::Num(values) => Ok(NumOperand::Column(Cow::Borrowed(values))),
            ValueVec::Scalar(LiteralValue::Number(number)) => Ok(NumOperand::Scalar(*number)),
            ValueVec::Str(values) => {
                let mut converted = Vec::with_capacity(values.len());
                for text in values {
                    converted.push(
                        ops::text_to_number(text)
                            .ok_or_else(|| ops::non_numeric_text_error(text))?,
                    );
                }
                Ok(NumOperand::Column(Cow::Owned(converted)))
            }
            ValueVec::Scalar(LiteralValue::String(text)) => Ok(NumOperand::Scalar(
                ops::text_to_number(text).ok_or_else(|| ops::non_numeric_text_error(text))?,
            )),
            ValueVec::Bool(_) | ValueVec::Scalar(LiteralValue::Boolean(_)) => {
                Err(Error::InvalidValue(
                    "Cannot use a boolean value in a numeric operation".to_string(),
                ))
            }
        }
    }

    fn to_bool_operand<'v>(&self, value: &'v ValueVec) -> Result<BoolOperand<'v>> {
        match value {
            ValueVec::Bool(values) => Ok(BoolOperand::Column(Cow::Borrowed(values))),
            ValueVec::Scalar(LiteralValue::Boolean(flag)) => Ok(BoolOperand::Scalar(*flag)),
            ValueVec::Str(values) => {
                let mut converted = Vec::with_capacity(values.len());
                for text in values {
                    converted.push(
                        ops::text_to_bool(text).ok_or_else(|| ops::non_boolean_text_error(text))?,
                    );
                }
                Ok(BoolOperand::Column(Cow::Owned(converted)))
            }
            ValueVec::Scalar(LiteralValue::String(text)) => Ok(BoolOperand::Scalar(
                ops::text_to_bool(text).ok_or_else(|| ops::non_boolean_text_error(text))?,
            )),
            ValueVec::Num(_) | ValueVec::Scalar(LiteralValue::Number(_)) => Err(
                Error::InvalidValue("Logical operations require boolean operands".to_string()),
            ),
        }
    }
}

/// Load a column, preserving its element domain.
///
/// Both evaluation tiers read columns through this function, so a given cell
/// means the same thing in each of them.
///
/// Columns physically stored as strings (everything produced by `read_csv`) are
/// classified once: a column whose cells are all numbers becomes numeric, one
/// whose cells are all `true`/`false` becomes boolean, and anything else stays
/// text.
pub(crate) fn load_column_values(dataframe: &DataFrame, name: &str) -> Result<ValueVec> {
    if let Ok(series) = dataframe.get_column::<f64>(name) {
        return Ok(ValueVec::Num(series.values().to_vec()));
    }
    if let Ok(series) = dataframe.get_column::<i64>(name) {
        return Ok(ValueVec::Num(
            series.values().iter().map(|v| *v as f64).collect(),
        ));
    }
    if let Ok(series) = dataframe.get_column::<bool>(name) {
        return Ok(ValueVec::Bool(series.values().to_vec()));
    }
    if let Ok(series) = dataframe.get_column::<String>(name) {
        return Ok(classify_text_column(series.values()));
    }

    macro_rules! try_numeric {
        ($($ty:ty),+ $(,)?) => {
            $(
                if let Ok(series) = dataframe.get_column::<$ty>(name) {
                    return Ok(ValueVec::Num(
                        series.values().iter().map(|v| *v as f64).collect(),
                    ));
                }
            )+
        };
    }
    try_numeric!(i32, f32, u64, u32, usize, i16, u16, i8, u8, isize, i128, u128);

    // Any other element type is reached through its string rendering, which
    // still supports text comparisons.
    Ok(classify_text_column(
        &dataframe.get_column_string_values(name)?,
    ))
}

fn to_str_operand(value: &ValueVec) -> Result<StrOperand<'_>> {
    match value {
        ValueVec::Str(values) => Ok(StrOperand::Column(values)),
        ValueVec::Scalar(LiteralValue::String(text)) => Ok(StrOperand::Scalar(text)),
        _ => Err(Error::InvalidValue(
            "Expected a text operand for a string operation".to_string(),
        )),
    }
}

/// Classify a physically-textual column into its logical domain.
///
/// A column is numeric when every cell is a number or a missing marker, boolean
/// when every cell is `true`/`false`, and textual otherwise. This mirrors the
/// per-cell parsing of the row-by-row interpreter for homogeneous columns, which
/// is what every real column is.
fn classify_text_column(values: &[String]) -> ValueVec {
    if values.is_empty() {
        return ValueVec::Str(Vec::new());
    }

    let mut numeric = Vec::with_capacity(values.len());
    let mut all_numeric = true;
    // A column made only of missing markers carries no type information, so it
    // stays textual rather than pretending to be numeric.
    let mut any_real_number = false;
    for text in values {
        match ops::text_to_number(text) {
            Some(number) => {
                if !ops::is_missing_text(text) {
                    any_real_number = true;
                }
                numeric.push(number);
            }
            None => {
                all_numeric = false;
                break;
            }
        }
    }
    if all_numeric && any_real_number {
        return ValueVec::Num(numeric);
    }

    let mut booleans = Vec::with_capacity(values.len());
    let mut all_boolean = true;
    for text in values {
        match ops::text_to_bool(text) {
            Some(flag) => booleans.push(flag),
            None => {
                all_boolean = false;
                break;
            }
        }
    }
    if all_boolean {
        return ValueVec::Bool(booleans);
    }

    ValueVec::Str(values.to_vec())
}
