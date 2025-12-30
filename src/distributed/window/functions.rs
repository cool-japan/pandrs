//! # Common Window Functions
//!
//! This module provides common window function factory methods for distributed processing.

use super::core::{WindowFrame, WindowFrameBoundary, WindowFunction};

/// Creates a ROW_NUMBER() window function
pub fn row_number(
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
) -> WindowFunction {
    WindowFunction::single_input("ROW_NUMBER", "", output, partition_by, order_by, None)
}

/// Creates a RANK() window function
pub fn rank(output: &str, partition_by: &[&str], order_by: &[(&str, bool)]) -> WindowFunction {
    WindowFunction::single_input("RANK", "", output, partition_by, order_by, None)
}

/// Creates a DENSE_RANK() window function
pub fn dense_rank(
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
) -> WindowFunction {
    WindowFunction::single_input("DENSE_RANK", "", output, partition_by, order_by, None)
}

/// Creates a PERCENT_RANK() window function
pub fn percent_rank(
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
) -> WindowFunction {
    WindowFunction::single_input("PERCENT_RANK", "", output, partition_by, order_by, None)
}

/// Creates a SUM() window function
pub fn sum(
    input: &str,
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
    frame: Option<WindowFrame>,
) -> WindowFunction {
    WindowFunction::single_input("SUM", input, output, partition_by, order_by, frame)
}

/// Creates a cumulative sum
pub fn cumulative_sum(
    input: &str,
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
) -> WindowFunction {
    WindowFunction::single_input(
        "SUM",
        input,
        output,
        partition_by,
        order_by,
        Some(WindowFrame::rows(
            WindowFrameBoundary::UnboundedPreceding,
            WindowFrameBoundary::CurrentRow,
        )),
    )
}

/// Creates an AVG() window function
pub fn avg(
    input: &str,
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
    frame: Option<WindowFrame>,
) -> WindowFunction {
    WindowFunction::single_input("AVG", input, output, partition_by, order_by, frame)
}

/// Creates a rolling average
pub fn rolling_avg(
    input: &str,
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
    window_size: usize,
) -> WindowFunction {
    WindowFunction::single_input(
        "AVG",
        input,
        output,
        partition_by,
        order_by,
        Some(WindowFrame::rows(
            WindowFrameBoundary::Preceding(window_size - 1),
            WindowFrameBoundary::CurrentRow,
        )),
    )
}

/// Creates a MAX() window function
pub fn max(
    input: &str,
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
    frame: Option<WindowFrame>,
) -> WindowFunction {
    WindowFunction::single_input("MAX", input, output, partition_by, order_by, frame)
}

/// Creates a MIN() window function
pub fn min(
    input: &str,
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
    frame: Option<WindowFrame>,
) -> WindowFunction {
    WindowFunction::single_input("MIN", input, output, partition_by, order_by, frame)
}

/// Creates a LAG() window function
pub fn lag(
    input: &str,
    output: &str,
    offset: usize,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
) -> WindowFunction {
    let function = format!("LAG({}, {})", input, offset);
    WindowFunction::new(&function, &[], output, partition_by, order_by, None)
}

/// Creates a LEAD() window function
pub fn lead(
    input: &str,
    output: &str,
    offset: usize,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
) -> WindowFunction {
    let function = format!("LEAD({}, {})", input, offset);
    WindowFunction::new(&function, &[], output, partition_by, order_by, None)
}

/// Creates a FIRST_VALUE() window function
pub fn first_value(
    input: &str,
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
) -> WindowFunction {
    WindowFunction::single_input("FIRST_VALUE", input, output, partition_by, order_by, None)
}

/// Creates a LAST_VALUE() window function
pub fn last_value(
    input: &str,
    output: &str,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
    frame: Option<WindowFrame>,
) -> WindowFunction {
    WindowFunction::single_input("LAST_VALUE", input, output, partition_by, order_by, frame)
}

/// Creates a NTH_VALUE() window function
pub fn nth_value(
    input: &str,
    output: &str,
    n: usize,
    partition_by: &[&str],
    order_by: &[(&str, bool)],
) -> WindowFunction {
    let function = format!("NTH_VALUE({}, {})", input, n);
    WindowFunction::new(&function, &[], output, partition_by, order_by, None)
}
