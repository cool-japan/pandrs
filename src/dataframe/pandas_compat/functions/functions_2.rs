//! Pandas compatibility extension trait implementation for DataFrame
//!
//! The implementation is split across helper modules for maintainability:
//! - functions_2_impl_part1: Core operations
//! - functions_2_impl_part2: Reshape and analysis operations
//! - functions_2_impl_part3: Element-wise, string, comparison, and utility operations

use super::super::helpers::{aggregations, comparison_ops, math_ops, string_ops, window_ops};
use super::super::merge;
use super::super::trait_def::PandasCompatExt;
use super::super::types::{Axis, CorrelationMatrix, DescribeStats, RankMethod, SeriesValue};
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use super::functions::select_rows_by_indices;
use super::functions_3::{covariance, pearson_correlation};

impl PandasCompatExt for DataFrame {
    fn assign<F, T>(&self, name: &str, func: F) -> Result<DataFrame>
    where
        F: FnOnce(&DataFrame) -> Vec<T>,
        T: Into<SeriesValue>,
    {
        super::functions_2_impl_part1::assign(self, name, func)
    }
    fn assign_many(&self, assignments: Vec<(&str, Vec<f64>)>) -> Result<DataFrame> {
        super::functions_2_impl_part1::assign_many(self, assignments)
    }
    fn pipe<F, R>(&self, func: F) -> R
    where
        F: FnOnce(&Self) -> R,
    {
        super::functions_2_impl_part1::pipe(self, func)
    }
    fn pipe_result<F>(&self, func: F) -> Result<DataFrame>
    where
        F: FnOnce(&Self) -> Result<DataFrame>,
    {
        super::functions_2_impl_part1::pipe_result(self, func)
    }
    fn isin(&self, column: &str, values: &[&str]) -> Result<Vec<bool>> {
        super::functions_2_impl_part1::isin(self, column, values)
    }
    fn isin_numeric(&self, column: &str, values: &[f64]) -> Result<Vec<bool>> {
        super::functions_2_impl_part1::isin_numeric(self, column, values)
    }
    fn nlargest(&self, n: usize, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part1::nlargest(self, n, column)
    }
    fn nsmallest(&self, n: usize, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part1::nsmallest(self, n, column)
    }
    fn idxmax(&self, column: &str) -> Result<Option<usize>> {
        super::functions_2_impl_part1::idxmax(self, column)
    }
    fn idxmin(&self, column: &str) -> Result<Option<usize>> {
        super::functions_2_impl_part1::idxmin(self, column)
    }
    fn rank(&self, column: &str, method: RankMethod) -> Result<Vec<f64>> {
        super::functions_2_impl_part1::rank(self, column, method)
    }
    fn clip(&self, column: &str, lower: Option<f64>, upper: Option<f64>) -> Result<DataFrame> {
        super::functions_2_impl_part1::clip(self, column, lower, upper)
    }
    fn between(&self, column: &str, lower: f64, upper: f64) -> Result<Vec<bool>> {
        super::functions_2_impl_part1::between(self, column, lower, upper)
    }
    fn transpose(&self) -> Result<DataFrame> {
        super::functions_2_impl_part1::transpose(self)
    }
    fn cumsum(&self, column: &str) -> Result<Vec<f64>> {
        super::functions_2_impl_part1::cumsum(self, column)
    }
    fn cumprod(&self, column: &str) -> Result<Vec<f64>> {
        super::functions_2_impl_part1::cumprod(self, column)
    }
    fn cummax(&self, column: &str) -> Result<Vec<f64>> {
        super::functions_2_impl_part1::cummax(self, column)
    }
    fn cummin(&self, column: &str) -> Result<Vec<f64>> {
        super::functions_2_impl_part1::cummin(self, column)
    }
    fn shift(&self, column: &str, periods: i32) -> Result<Vec<Option<f64>>> {
        super::functions_2_impl_part1::shift(self, column, periods)
    }
    fn nunique(&self) -> Result<Vec<(String, usize)>> {
        super::functions_2_impl_part1::nunique(self)
    }
    fn memory_usage(&self) -> usize {
        super::functions_2_impl_part1::memory_usage(self)
    }
    fn value_counts(&self, column: &str) -> Result<Vec<(String, usize)>> {
        super::functions_2_impl_part1::value_counts(self, column)
    }
    fn value_counts_numeric(&self, column: &str) -> Result<Vec<(f64, usize)>> {
        super::functions_2_impl_part1::value_counts_numeric(self, column)
    }
    fn describe(&self, column: &str) -> Result<DescribeStats> {
        super::functions_2_impl_part1::describe(self, column)
    }
    fn apply<F, T>(&self, func: F, axis: Axis) -> Result<Vec<T>>
    where
        F: Fn(&[f64]) -> T,
    {
        super::functions_2_impl_part1::apply(self, func, axis)
    }
    fn corr(&self) -> Result<CorrelationMatrix> {
        super::functions_2_impl_part1::corr(self)
    }
    fn cov(&self) -> Result<CorrelationMatrix> {
        super::functions_2_impl_part1::cov(self)
    }
    fn pct_change(&self, column: &str, periods: usize) -> Result<Vec<f64>> {
        super::functions_2_impl_part1::pct_change(self, column, periods)
    }
    fn diff(&self, column: &str, periods: usize) -> Result<Vec<f64>> {
        super::functions_2_impl_part1::diff(self, column, periods)
    }
    fn replace(&self, column: &str, to_replace: &[&str], values: &[&str]) -> Result<DataFrame> {
        super::functions_2_impl_part1::replace(self, column, to_replace, values)
    }
    fn replace_numeric(
        &self,
        column: &str,
        to_replace: &[f64],
        values: &[f64],
    ) -> Result<DataFrame> {
        super::functions_2_impl_part1::replace_numeric(self, column, to_replace, values)
    }
    fn sample(&self, n: usize, replace: bool) -> Result<DataFrame> {
        super::functions_2_impl_part1::sample(self, n, replace)
    }
    fn drop_columns(&self, labels: &[&str]) -> Result<DataFrame> {
        super::functions_2_impl_part1::drop_columns(self, labels)
    }
    fn rename_columns(&self, mapper: &HashMap<String, String>) -> Result<DataFrame> {
        super::functions_2_impl_part1::rename_columns(self, mapper)
    }
    fn abs(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part1::abs(self, column)
    }
    fn round(&self, column: &str, decimals: i32) -> Result<DataFrame> {
        super::functions_2_impl_part1::round(self, column, decimals)
    }
    fn quantile(&self, column: &str, q: f64) -> Result<f64> {
        super::functions_2_impl_part1::quantile(self, column, q)
    }
    fn head(&self, n: usize) -> Result<DataFrame> {
        super::functions_2_impl_part1::head(self, n)
    }
    fn tail(&self, n: usize) -> Result<DataFrame> {
        super::functions_2_impl_part1::tail(self, n)
    }
    fn unique(&self, column: &str) -> Result<Vec<String>> {
        super::functions_2_impl_part1::unique(self, column)
    }
    fn unique_numeric(&self, column: &str) -> Result<Vec<f64>> {
        super::functions_2_impl_part1::unique_numeric(self, column)
    }
    fn fillna(&self, column: &str, value: f64) -> Result<DataFrame> {
        super::functions_2_impl_part1::fillna(self, column, value)
    }
    fn fillna_method(&self, column: &str, method: &str) -> Result<DataFrame> {
        super::functions_2_impl_part1::fillna_method(self, column, method)
    }
    fn interpolate(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part1::interpolate(self, column)
    }
    fn dropna(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part1::dropna(self, column)
    }
    fn isna(&self, column: &str) -> Result<Vec<bool>> {
        super::functions_2_impl_part1::isna(self, column)
    }
    fn sum_all(&self) -> Result<Vec<(String, f64)>> {
        super::functions_2_impl_part1::sum_all(self)
    }
    fn mean_all(&self) -> Result<Vec<(String, f64)>> {
        super::functions_2_impl_part1::mean_all(self)
    }
    fn std_all(&self) -> Result<Vec<(String, f64)>> {
        super::functions_2_impl_part1::std_all(self)
    }
    fn var_all(&self) -> Result<Vec<(String, f64)>> {
        super::functions_2_impl_part1::var_all(self)
    }
    fn min_all(&self) -> Result<Vec<(String, f64)>> {
        super::functions_2_impl_part1::min_all(self)
    }
    fn max_all(&self) -> Result<Vec<(String, f64)>> {
        super::functions_2_impl_part1::max_all(self)
    }
    fn sort_values(&self, column: &str, ascending: bool) -> Result<DataFrame> {
        super::functions_2_impl_part1::sort_values(self, column, ascending)
    }
    fn sort_by_columns(&self, columns: &[&str], ascending: &[bool]) -> Result<DataFrame> {
        super::functions_2_impl_part1::sort_by_columns(self, columns, ascending)
    }
    fn merge(
        &self,
        other: &DataFrame,
        on: &str,
        how: super::super::merge::JoinType,
        suffixes: (&str, &str),
    ) -> Result<DataFrame> {
        super::functions_2_impl_part1::merge(self, other, on, how, suffixes)
    }
    fn where_cond(&self, column: &str, condition: &[bool], other: f64) -> Result<DataFrame> {
        super::functions_2_impl_part1::where_cond(self, column, condition, other)
    }
    fn mask(&self, column: &str, condition: &[bool], other: f64) -> Result<DataFrame> {
        super::functions_2_impl_part1::mask(self, column, condition, other)
    }
    fn drop_duplicates(&self, subset: Option<&[&str]>, keep: &str) -> Result<DataFrame> {
        super::functions_2_impl_part1::drop_duplicates(self, subset, keep)
    }
    fn select_dtypes(&self, include: &[&str]) -> Result<DataFrame> {
        super::functions_2_impl_part1::select_dtypes(self, include)
    }
    fn any_numeric(&self) -> Result<Vec<(String, bool)>> {
        super::functions_2_impl_part1::any_numeric(self)
    }
    fn all_numeric(&self) -> Result<Vec<(String, bool)>> {
        super::functions_2_impl_part1::all_numeric(self)
    }
    fn count_valid(&self) -> Result<Vec<(String, usize)>> {
        super::functions_2_impl_part1::count_valid(self)
    }
    fn reverse_columns(&self) -> Result<DataFrame> {
        super::functions_2_impl_part1::reverse_columns(self)
    }
    fn reverse_rows(&self) -> Result<DataFrame> {
        super::functions_2_impl_part1::reverse_rows(self)
    }
    fn notna(&self, column: &str) -> Result<Vec<bool>> {
        super::functions_2_impl_part1::notna(self, column)
    }
    fn melt(
        &self,
        id_vars: &[&str],
        value_vars: Option<&[&str]>,
        var_name: &str,
        value_name: &str,
    ) -> Result<DataFrame> {
        super::functions_2_impl_part1::melt(self, id_vars, value_vars, var_name, value_name)
    }
    fn explode(&self, column: &str, separator: &str) -> Result<DataFrame> {
        super::functions_2_impl_part2::explode(self, column, separator)
    }
    fn duplicated(&self, subset: Option<&[&str]>, keep: &str) -> Result<Vec<bool>> {
        super::functions_2_impl_part2::duplicated(self, subset, keep)
    }
    fn copy(&self) -> DataFrame {
        super::functions_2_impl_part2::copy(self)
    }
    fn to_dict(&self) -> Result<HashMap<String, Vec<String>>> {
        super::functions_2_impl_part2::to_dict(self)
    }
    fn first_valid_index(&self, column: &str) -> Result<Option<usize>> {
        super::functions_2_impl_part2::first_valid_index(self, column)
    }
    fn last_valid_index(&self, column: &str) -> Result<Option<usize>> {
        super::functions_2_impl_part2::last_valid_index(self, column)
    }
    fn product_all(&self) -> Result<Vec<(String, f64)>> {
        super::functions_2_impl_part2::product_all(self)
    }
    fn median_all(&self) -> Result<Vec<(String, f64)>> {
        super::functions_2_impl_part2::median_all(self)
    }
    fn skew(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part2::skew(self, column)
    }
    fn kurtosis(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part2::kurtosis(self, column)
    }
    fn add_prefix(&self, prefix: &str) -> Result<DataFrame> {
        super::functions_2_impl_part2::add_prefix(self, prefix)
    }
    fn add_suffix(&self, suffix: &str) -> Result<DataFrame> {
        super::functions_2_impl_part2::add_suffix(self, suffix)
    }
    fn filter_by_mask(&self, mask: &[bool]) -> Result<DataFrame> {
        super::functions_2_impl_part2::filter_by_mask(self, mask)
    }
    fn mode_numeric(&self, column: &str) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::mode_numeric(self, column)
    }
    fn mode_string(&self, column: &str) -> Result<Vec<String>> {
        super::functions_2_impl_part2::mode_string(self, column)
    }
    fn percentile(&self, column: &str, n: f64) -> Result<f64> {
        super::functions_2_impl_part2::percentile(self, column, n)
    }
    fn ewma(&self, column: &str, span: usize) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::ewma(self, column, span)
    }
    fn iloc(&self, index: usize) -> Result<HashMap<String, String>> {
        super::functions_2_impl_part2::iloc(self, index)
    }
    fn iloc_range(&self, start: usize, end: usize) -> Result<DataFrame> {
        super::functions_2_impl_part2::iloc_range(self, start, end)
    }
    fn info(&self) -> String {
        super::functions_2_impl_part2::info(self)
    }
    fn equals(&self, other: &DataFrame) -> bool {
        super::functions_2_impl_part2::equals(self, other)
    }
    fn compare(&self, other: &DataFrame) -> Result<DataFrame> {
        super::functions_2_impl_part2::compare(self, other)
    }
    fn keys(&self) -> Vec<String> {
        super::functions_2_impl_part2::keys(self)
    }
    fn pop_column(&self, column: &str) -> Result<(DataFrame, Vec<f64>)> {
        super::functions_2_impl_part2::pop_column(self, column)
    }
    fn insert_column(&self, loc: usize, name: &str, values: Vec<f64>) -> Result<DataFrame> {
        super::functions_2_impl_part2::insert_column(self, loc, name, values)
    }
    fn rolling_sum(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::rolling_sum(self, column, window, min_periods)
    }
    fn rolling_mean(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::rolling_mean(self, column, window, min_periods)
    }
    fn rolling_std(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::rolling_std(self, column, window, min_periods)
    }
    fn rolling_min(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::rolling_min(self, column, window, min_periods)
    }
    fn rolling_max(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::rolling_max(self, column, window, min_periods)
    }
    fn rolling_var(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::rolling_var(self, column, window, min_periods)
    }
    fn rolling_median(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::rolling_median(self, column, window, min_periods)
    }
    fn rolling_count(&self, column: &str, window: usize) -> Result<Vec<usize>> {
        super::functions_2_impl_part2::rolling_count(self, column, window)
    }
    fn rolling_apply<F>(
        &self,
        column: &str,
        window: usize,
        func: F,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        super::functions_2_impl_part2::rolling_apply(self, column, window, func, min_periods)
    }
    fn cumcount(&self, column: &str) -> Result<Vec<usize>> {
        super::functions_2_impl_part2::cumcount(self, column)
    }
    fn nth(&self, n: i32) -> Result<HashMap<String, String>> {
        super::functions_2_impl_part2::nth(self, n)
    }
    fn transform<F>(&self, column: &str, func: F) -> Result<DataFrame>
    where
        F: Fn(f64) -> f64,
    {
        super::functions_2_impl_part2::transform(self, column, func)
    }
    fn crosstab(&self, col1: &str, col2: &str) -> Result<DataFrame> {
        super::functions_2_impl_part2::crosstab(self, col1, col2)
    }
    fn expanding_sum(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::expanding_sum(self, column, min_periods)
    }
    fn expanding_mean(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::expanding_mean(self, column, min_periods)
    }
    fn expanding_std(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::expanding_std(self, column, min_periods)
    }
    fn expanding_min(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::expanding_min(self, column, min_periods)
    }
    fn expanding_max(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::expanding_max(self, column, min_periods)
    }
    fn expanding_var(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::expanding_var(self, column, min_periods)
    }
    fn expanding_apply<F>(&self, column: &str, func: F, min_periods: usize) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        super::functions_2_impl_part2::expanding_apply(self, column, func, min_periods)
    }
    fn align(&self, other: &DataFrame) -> Result<(DataFrame, DataFrame)> {
        super::functions_2_impl_part2::align(self, other)
    }
    fn reindex_columns(&self, columns: &[&str]) -> Result<DataFrame> {
        super::functions_2_impl_part2::reindex_columns(self, columns)
    }
    fn value_range(&self, column: &str) -> Result<(f64, f64)> {
        super::functions_2_impl_part2::value_range(self, column)
    }
    fn zscore(&self, column: &str) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::zscore(self, column)
    }
    fn normalize(&self, column: &str) -> Result<Vec<f64>> {
        super::functions_2_impl_part2::normalize(self, column)
    }
    fn cut(&self, column: &str, bins: usize) -> Result<Vec<String>> {
        super::functions_2_impl_part2::cut(self, column, bins)
    }
    fn qcut(&self, column: &str, q: usize) -> Result<Vec<String>> {
        super::functions_2_impl_part2::qcut(self, column, q)
    }
    fn stack(&self, columns: Option<&[&str]>) -> Result<DataFrame> {
        super::functions_2_impl_part2::stack(self, columns)
    }
    fn unstack(&self, index_col: &str, columns_col: &str, values_col: &str) -> Result<DataFrame> {
        super::functions_2_impl_part2::unstack(self, index_col, columns_col, values_col)
    }
    fn pivot(&self, index: &str, columns: &str, values: &str) -> Result<DataFrame> {
        super::functions_2_impl_part2::pivot(self, index, columns, values)
    }
    fn astype(&self, column: &str, dtype: &str) -> Result<DataFrame> {
        super::functions_2_impl_part2::astype(self, column, dtype)
    }
    fn applymap<F>(&self, func: F) -> Result<DataFrame>
    where
        F: Fn(f64) -> f64,
    {
        super::functions_2_impl_part2::applymap(self, func)
    }
    fn agg(&self, column: &str, funcs: &[&str]) -> Result<HashMap<String, f64>> {
        super::functions_2_impl_part2::agg(self, column, funcs)
    }
    fn dtypes(&self) -> Vec<(String, String)> {
        super::functions_2_impl_part2::dtypes(self)
    }
    fn set_values(&self, column: &str, indices: &[usize], values: &[f64]) -> Result<DataFrame> {
        super::functions_2_impl_part2::set_values(self, column, indices, values)
    }
    fn query_eq(&self, column: &str, value: f64) -> Result<DataFrame> {
        super::functions_2_impl_part2::query_eq(self, column, value)
    }
    fn query_gt(&self, column: &str, value: f64) -> Result<DataFrame> {
        super::functions_2_impl_part2::query_gt(self, column, value)
    }
    fn query_lt(&self, column: &str, value: f64) -> Result<DataFrame> {
        super::functions_2_impl_part2::query_lt(self, column, value)
    }
    fn query_contains(&self, column: &str, pattern: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::query_contains(self, column, pattern)
    }
    fn select_columns(&self, columns: &[&str]) -> Result<DataFrame> {
        super::functions_2_impl_part3::select_columns(self, columns)
    }
    fn add_scalar(&self, column: &str, value: f64) -> Result<DataFrame> {
        super::functions_2_impl_part3::add_scalar(self, column, value)
    }
    fn mul_scalar(&self, column: &str, value: f64) -> Result<DataFrame> {
        super::functions_2_impl_part3::mul_scalar(self, column, value)
    }
    fn sub_scalar(&self, column: &str, value: f64) -> Result<DataFrame> {
        super::functions_2_impl_part3::sub_scalar(self, column, value)
    }
    fn div_scalar(&self, column: &str, value: f64) -> Result<DataFrame> {
        super::functions_2_impl_part3::div_scalar(self, column, value)
    }
    fn pow(&self, column: &str, exponent: f64) -> Result<DataFrame> {
        super::functions_2_impl_part3::pow(self, column, exponent)
    }
    fn sqrt(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::sqrt(self, column)
    }
    fn log(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::log(self, column)
    }
    fn exp(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::exp(self, column)
    }
    fn col_add(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::col_add(self, col1, col2, result_name)
    }
    fn col_mul(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::col_mul(self, col1, col2, result_name)
    }
    fn col_sub(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::col_sub(self, col1, col2, result_name)
    }
    fn col_div(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::col_div(self, col1, col2, result_name)
    }
    fn iterrows(&self) -> Vec<(usize, HashMap<String, SeriesValue>)> {
        super::functions_2_impl_part3::iterrows(self)
    }
    fn at(&self, row: usize, column: &str) -> Result<SeriesValue> {
        super::functions_2_impl_part3::at(self, row, column)
    }
    fn iat(&self, row: usize, col_idx: usize) -> Result<SeriesValue> {
        super::functions_2_impl_part3::iat(self, row, col_idx)
    }
    fn drop_rows(&self, indices: &[usize]) -> Result<DataFrame> {
        super::functions_2_impl_part3::drop_rows(self, indices)
    }
    fn set_index(&self, column: &str, drop: bool) -> Result<(DataFrame, Vec<String>)> {
        super::functions_2_impl_part3::set_index(self, column, drop)
    }
    fn reset_index(&self, index_values: Option<&[String]>, name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::reset_index(self, index_values, name)
    }
    fn to_records(&self) -> Vec<HashMap<String, SeriesValue>> {
        super::functions_2_impl_part3::to_records(self)
    }
    fn items(&self) -> Vec<(String, Vec<SeriesValue>)> {
        super::functions_2_impl_part3::items(self)
    }
    fn update(&self, other: &DataFrame) -> Result<DataFrame> {
        super::functions_2_impl_part3::update(self, other)
    }
    fn combine<F>(&self, other: &DataFrame, func: F) -> Result<DataFrame>
    where
        F: Fn(Option<f64>, Option<f64>) -> f64,
    {
        super::functions_2_impl_part3::combine(self, other, func)
    }
    fn shape(&self) -> (usize, usize) {
        super::functions_2_impl_part3::shape(self)
    }
    fn size(&self) -> usize {
        super::functions_2_impl_part3::size(self)
    }
    fn empty(&self) -> bool {
        super::functions_2_impl_part3::empty(self)
    }
    fn first_row(&self) -> Result<HashMap<String, SeriesValue>> {
        super::functions_2_impl_part3::first_row(self)
    }
    fn last_row(&self) -> Result<HashMap<String, SeriesValue>> {
        super::functions_2_impl_part3::last_row(self)
    }
    fn get_value(&self, row: usize, column: &str, default: SeriesValue) -> SeriesValue {
        super::functions_2_impl_part3::get_value(self, row, column, default)
    }
    fn lookup(
        &self,
        lookup_col: &str,
        other: &DataFrame,
        other_col: &str,
        result_col: &str,
    ) -> Result<DataFrame> {
        super::functions_2_impl_part3::lookup(self, lookup_col, other, other_col, result_col)
    }
    fn get_column_by_index(&self, idx: usize) -> Result<(String, Vec<SeriesValue>)> {
        super::functions_2_impl_part3::get_column_by_index(self, idx)
    }
    fn swap_columns(&self, col1: &str, col2: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::swap_columns(self, col1, col2)
    }
    fn sort_columns(&self, ascending: bool) -> Result<DataFrame> {
        super::functions_2_impl_part3::sort_columns(self, ascending)
    }
    fn rename_column(&self, old_name: &str, new_name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::rename_column(self, old_name, new_name)
    }
    fn to_categorical(&self, column: &str) -> Result<(DataFrame, HashMap<String, i64>)> {
        super::functions_2_impl_part3::to_categorical(self, column)
    }
    fn row_hash(&self) -> Vec<u64> {
        super::functions_2_impl_part3::row_hash(self)
    }
    fn sample_frac(&self, frac: f64, replace: bool) -> Result<DataFrame> {
        super::functions_2_impl_part3::sample_frac(self, frac, replace)
    }
    fn take(&self, indices: &[usize]) -> Result<DataFrame> {
        super::functions_2_impl_part3::take(self, indices)
    }
    fn duplicated_rows(&self, subset: Option<&[&str]>, keep: &str) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::duplicated_rows(self, subset, keep)
    }
    fn get_column_as_f64(&self, column: &str) -> Result<Vec<f64>> {
        super::functions_2_impl_part3::get_column_as_f64(self, column)
    }
    fn get_column_as_string(&self, column: &str) -> Result<Vec<String>> {
        super::functions_2_impl_part3::get_column_as_string(self, column)
    }
    fn groupby_apply<F>(&self, by: &str, func: F) -> Result<DataFrame>
    where
        F: Fn(&DataFrame) -> Result<HashMap<String, f64>>,
    {
        super::functions_2_impl_part3::groupby_apply(self, by, func)
    }
    fn corr_columns(&self, col1: &str, col2: &str) -> Result<f64> {
        super::functions_2_impl_part3::corr_columns(self, col1, col2)
    }
    fn cov_columns(&self, col1: &str, col2: &str) -> Result<f64> {
        super::functions_2_impl_part3::cov_columns(self, col1, col2)
    }
    fn var_column(&self, column: &str, ddof: usize) -> Result<f64> {
        super::functions_2_impl_part3::var_column(self, column, ddof)
    }
    fn std_column(&self, column: &str, ddof: usize) -> Result<f64> {
        super::functions_2_impl_part3::std_column(self, column, ddof)
    }
    fn str_lower(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::str_lower(self, column)
    }
    fn str_upper(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::str_upper(self, column)
    }
    fn str_strip(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::str_strip(self, column)
    }
    fn str_contains(&self, column: &str, pattern: &str) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::str_contains(self, column, pattern)
    }
    fn str_replace(&self, column: &str, pattern: &str, replacement: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::str_replace(self, column, pattern, replacement)
    }
    fn str_split(&self, column: &str, delimiter: &str) -> Result<Vec<Vec<String>>> {
        super::functions_2_impl_part3::str_split(self, column, delimiter)
    }
    fn str_len(&self, column: &str) -> Result<Vec<usize>> {
        super::functions_2_impl_part3::str_len(self, column)
    }
    fn sem(&self, column: &str, ddof: usize) -> Result<f64> {
        super::functions_2_impl_part3::sem(self, column, ddof)
    }
    fn mad(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part3::mad(self, column)
    }
    fn ffill(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::ffill(self, column)
    }
    fn bfill(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::bfill(self, column)
    }
    fn pct_rank(&self, column: &str) -> Result<Vec<f64>> {
        super::functions_2_impl_part3::pct_rank(self, column)
    }
    fn abs_column(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::abs_column(self, column)
    }
    fn round_column(&self, column: &str, decimals: i32) -> Result<DataFrame> {
        super::functions_2_impl_part3::round_column(self, column, decimals)
    }
    fn argmax(&self, column: &str) -> Result<usize> {
        super::functions_2_impl_part3::argmax(self, column)
    }
    fn argmin(&self, column: &str) -> Result<usize> {
        super::functions_2_impl_part3::argmin(self, column)
    }
    fn gt(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::gt(self, column, value)
    }
    fn ge(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::ge(self, column, value)
    }
    fn lt(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::lt(self, column, value)
    }
    fn le(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::le(self, column, value)
    }
    fn eq_value(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::eq_value(self, column, value)
    }
    fn ne_value(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::ne_value(self, column, value)
    }
    fn clip_lower(&self, column: &str, min: f64) -> Result<DataFrame> {
        super::functions_2_impl_part3::clip_lower(self, column, min)
    }
    fn clip_upper(&self, column: &str, max: f64) -> Result<DataFrame> {
        super::functions_2_impl_part3::clip_upper(self, column, max)
    }
    fn any_column(&self, column: &str) -> Result<bool> {
        super::functions_2_impl_part3::any_column(self, column)
    }
    fn all_column(&self, column: &str) -> Result<bool> {
        super::functions_2_impl_part3::all_column(self, column)
    }
    fn count_na(&self, column: &str) -> Result<usize> {
        super::functions_2_impl_part3::count_na(self, column)
    }
    fn prod(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part3::prod(self, column)
    }
    fn coalesce(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::coalesce(self, col1, col2, result_name)
    }
    fn first_valid(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part3::first_valid(self, column)
    }
    fn last_valid(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part3::last_valid(self, column)
    }
    fn add_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::add_columns(self, col1, col2, result_name)
    }
    fn sub_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::sub_columns(self, col1, col2, result_name)
    }
    fn mul_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::mul_columns(self, col1, col2, result_name)
    }
    fn div_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::div_columns(self, col1, col2, result_name)
    }
    fn mod_column(&self, column: &str, divisor: f64) -> Result<DataFrame> {
        super::functions_2_impl_part3::mod_column(self, column, divisor)
    }
    fn floordiv(&self, column: &str, divisor: f64) -> Result<DataFrame> {
        super::functions_2_impl_part3::floordiv(self, column, divisor)
    }
    fn neg(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::neg(self, column)
    }
    fn sign(&self, column: &str) -> Result<Vec<i32>> {
        super::functions_2_impl_part3::sign(self, column)
    }
    fn is_finite(&self, column: &str) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::is_finite(self, column)
    }
    fn is_infinite(&self, column: &str) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::is_infinite(self, column)
    }
    fn replace_inf(&self, column: &str, replacement: f64) -> Result<DataFrame> {
        super::functions_2_impl_part3::replace_inf(self, column, replacement)
    }
    fn str_startswith(&self, column: &str, prefix: &str) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::str_startswith(self, column, prefix)
    }
    fn str_endswith(&self, column: &str, suffix: &str) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::str_endswith(self, column, suffix)
    }
    fn str_pad_left(&self, column: &str, width: usize, fillchar: char) -> Result<DataFrame> {
        super::functions_2_impl_part3::str_pad_left(self, column, width, fillchar)
    }
    fn str_pad_right(&self, column: &str, width: usize, fillchar: char) -> Result<DataFrame> {
        super::functions_2_impl_part3::str_pad_right(self, column, width, fillchar)
    }
    fn str_slice(&self, column: &str, start: usize, end: Option<usize>) -> Result<DataFrame> {
        super::functions_2_impl_part3::str_slice(self, column, start, end)
    }
    fn floor(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::floor(self, column)
    }
    fn ceil(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::ceil(self, column)
    }
    fn trunc(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::trunc(self, column)
    }
    fn fract(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::fract(self, column)
    }
    fn reciprocal(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::reciprocal(self, column)
    }
    fn count_value(&self, column: &str, value: f64) -> Result<usize> {
        super::functions_2_impl_part3::count_value(self, column, value)
    }
    fn fillna_zero(&self, column: &str) -> Result<DataFrame> {
        super::functions_2_impl_part3::fillna_zero(self, column)
    }
    fn nunique_all(&self) -> Result<HashMap<String, usize>> {
        super::functions_2_impl_part3::nunique_all(self)
    }
    fn is_between(
        &self,
        column: &str,
        lower: f64,
        upper: f64,
        inclusive: bool,
    ) -> Result<Vec<bool>> {
        super::functions_2_impl_part3::is_between(self, column, lower, upper, inclusive)
    }
    fn str_count(&self, column: &str, pattern: &str) -> Result<Vec<usize>> {
        super::functions_2_impl_part3::str_count(self, column, pattern)
    }
    fn str_repeat(&self, column: &str, n: usize) -> Result<DataFrame> {
        super::functions_2_impl_part3::str_repeat(self, column, n)
    }
    fn str_center(&self, column: &str, width: usize, fillchar: char) -> Result<DataFrame> {
        super::functions_2_impl_part3::str_center(self, column, width, fillchar)
    }
    fn str_zfill(&self, column: &str, width: usize) -> Result<DataFrame> {
        super::functions_2_impl_part3::str_zfill(self, column, width)
    }
    fn is_numeric_column(&self, column: &str) -> bool {
        super::functions_2_impl_part3::is_numeric_column(self, column)
    }
    fn is_string_column(&self, column: &str) -> bool {
        super::functions_2_impl_part3::is_string_column(self, column)
    }
    fn has_nulls(&self, column: &str) -> Result<bool> {
        super::functions_2_impl_part3::has_nulls(self, column)
    }
    fn describe_column(&self, column: &str) -> Result<HashMap<String, f64>> {
        super::functions_2_impl_part3::describe_column(self, column)
    }
    fn memory_usage_column(&self, column: &str) -> Result<usize> {
        super::functions_2_impl_part3::memory_usage_column(self, column)
    }
    fn range(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part3::range(self, column)
    }
    fn abs_sum(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part3::abs_sum(self, column)
    }
    fn is_unique(&self, column: &str) -> Result<bool> {
        super::functions_2_impl_part3::is_unique(self, column)
    }
    fn mode_with_count(&self, column: &str) -> Result<(f64, usize)> {
        super::functions_2_impl_part3::mode_with_count(self, column)
    }
    fn geometric_mean(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part3::geometric_mean(self, column)
    }
    fn harmonic_mean(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part3::harmonic_mean(self, column)
    }
    fn iqr(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part3::iqr(self, column)
    }
    fn cv(&self, column: &str) -> Result<f64> {
        super::functions_2_impl_part3::cv(self, column)
    }
    fn percentile_value(&self, column: &str, q: f64) -> Result<f64> {
        super::functions_2_impl_part3::percentile_value(self, column, q)
    }
    fn trimmed_mean(&self, column: &str, trim_fraction: f64) -> Result<f64> {
        super::functions_2_impl_part3::trimmed_mean(self, column, trim_fraction)
    }
}
