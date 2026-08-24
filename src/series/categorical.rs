use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::core::error::{Error, Result};
use crate::na::NA;
use crate::series::{NASeries, Series};

// Re-export from legacy module for backward compatibility
pub use crate::series::categorical::{
    Categorical as LegacyCategorical, CategoricalOrder as LegacyCategoricalOrder,
    StringCategorical as LegacyStringCategorical,
};

/// Enumeration for categorical order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CategoricalOrder {
    /// No specific order is defined
    Unordered,
    /// Categories have a specific order
    Ordered,
}

/// Categorical data type with memory-efficient storage
///
/// Stores categorical data using integer codes that map to category values,
/// providing significant memory savings for columns with repeated string values.
///
/// `codes` and `categories_list` (plus the `category_to_code` lookup built
/// from it) are the single source of truth for every row's logical value:
/// row `i`'s value is `codes[i] < 0 ? NA : categories_list[codes[i]]`. There
/// is deliberately no separate "original values" array kept alongside them
/// -- earlier versions of this type stored one, and every accessor that
/// read it instead of `codes`/`categories_list` went stale the moment a
/// category was added, removed, or reordered (or simply saw `len() == 0`
/// under [`Categorical::new_compact`], which never populated it). Routing
/// every accessor through `codes` + `categories_list` means there is only
/// one representation to keep consistent.
#[derive(Debug, Clone)]
pub struct Categorical<T>
where
    T: Debug + Clone + Eq + Hash,
{
    _phantom: std::marker::PhantomData<T>,
    /// Unique category values, indexed by code.
    categories_list: Vec<T>,
    /// Integer codes mapping each position to a category index (-1 for NA).
    codes: Vec<i32>,
    /// Whether categories have a meaningful order
    ordered_flag: bool,
    /// Category to code lookup for fast encoding
    category_to_code: HashMap<T, i32>,
}

impl<T> Categorical<T>
where
    T: Debug + Clone + Eq + Hash,
{
    /// Create a new Categorical with proper code mapping
    ///
    /// # Arguments
    /// * `values` - The input values to categorize
    /// * `categories` - Optional predefined categories. If None, categories are inferred.
    /// * `ordered` - Whether the categories have a meaningful order
    pub fn new(values: Vec<T>, categories: Option<Vec<T>>, ordered: bool) -> Result<Self> {
        // Build unique categories if not provided
        let mut categories_list = if let Some(cats) = categories {
            cats
        } else {
            // Extract unique categories from values, preserving order
            let mut unique = Vec::new();
            for v in &values {
                if !unique.contains(v) {
                    unique.push(v.clone());
                }
            }
            unique
        };

        // Build category to code mapping
        let mut category_to_code: HashMap<T, i32> = HashMap::new();
        for (i, cat) in categories_list.iter().enumerate() {
            category_to_code.insert(cat.clone(), i as i32);
        }

        // Compute codes for each value
        let mut codes = Vec::with_capacity(values.len());
        for v in &values {
            if let Some(&code) = category_to_code.get(v) {
                codes.push(code);
            } else {
                // Value not in categories - add it
                let new_code = categories_list.len() as i32;
                categories_list.push(v.clone());
                category_to_code.insert(v.clone(), new_code);
                codes.push(new_code);
            }
        }

        Ok(Self {
            _phantom: std::marker::PhantomData,
            categories_list,
            codes,
            ordered_flag: ordered,
            category_to_code,
        })
    }

    /// Create a categorical from values, without a redundant raw-value copy.
    ///
    /// All [`Categorical`] instances are code-compact now (there is no raw
    /// `values` array to omit -- see the struct-level docs), so this is a
    /// thin alias for [`Categorical::new`], kept for API compatibility with
    /// callers that asked for the memory-efficient constructor explicitly.
    pub fn new_compact(values: Vec<T>, categories: Option<Vec<T>>, ordered: bool) -> Result<Self> {
        Self::new(values, categories, ordered)
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_usage_bytes(&self) -> usize {
        let codes_size = self.codes.len() * std::mem::size_of::<i32>();
        let categories_overhead = self.categories_list.len() * std::mem::size_of::<T>();
        codes_size + categories_overhead
    }

    /// Decode codes back to values
    pub fn decode(&self) -> Vec<Option<T>> {
        self.codes
            .iter()
            .map(|&code| {
                if code < 0 {
                    None
                } else {
                    self.categories_list.get(code as usize).cloned()
                }
            })
            .collect()
    }

    /// The non-NA values in row order, materialized from `codes` +
    /// `categories_list` (i.e. `decode()` with the `NA` rows dropped).
    ///
    /// This is shorter than `len()` whenever any row is NA -- callers that
    /// need the NA rows represented (even as a gap) should use
    /// [`Categorical::decode`] or [`Categorical::to_na_vec`] instead.
    fn materialized_values(&self) -> Vec<T> {
        self.decode().into_iter().flatten().collect()
    }

    /// Encode new values using existing categories
    pub fn encode(&self, values: &[T]) -> Vec<i32> {
        values
            .iter()
            .map(|v| self.category_to_code.get(v).copied().unwrap_or(-1))
            .collect()
    }

    /// Get the number of unique categories
    pub fn num_categories(&self) -> usize {
        self.categories_list.len()
    }

    /// Check if a value exists in categories
    pub fn contains_category(&self, value: &T) -> bool {
        self.category_to_code.contains_key(value)
    }

    /// Get code for a specific value
    pub fn get_code(&self, value: &T) -> Option<i32> {
        self.category_to_code.get(value).copied()
    }

    /// Get category for a specific code
    pub fn get_category(&self, code: i32) -> Option<&T> {
        if code < 0 {
            None
        } else {
            self.categories_list.get(code as usize)
        }
    }

    /// Remove unused categories
    pub fn remove_unused_categories(&mut self) -> Result<()> {
        let mut used_codes: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for &code in &self.codes {
            if code >= 0 {
                used_codes.insert(code);
            }
        }

        let mut new_categories = Vec::new();
        let mut old_to_new: HashMap<i32, i32> = HashMap::new();

        for (old_code, cat) in self.categories_list.iter().enumerate() {
            if used_codes.contains(&(old_code as i32)) {
                let new_code = new_categories.len() as i32;
                old_to_new.insert(old_code as i32, new_code);
                new_categories.push(cat.clone());
            }
        }

        for code in &mut self.codes {
            if *code >= 0 {
                *code = old_to_new.get(code).copied().unwrap_or(-1);
            }
        }

        self.category_to_code.clear();
        for (i, cat) in new_categories.iter().enumerate() {
            self.category_to_code.insert(cat.clone(), i as i32);
        }

        self.categories_list = new_categories;
        Ok(())
    }

    /// Convert to a factorized representation (codes, uniques)
    pub fn factorize(&self) -> (Vec<i32>, Vec<T>) {
        (self.codes.clone(), self.categories_list.clone())
    }

    /// Create from a vector with NA values.
    ///
    /// Unlike an earlier implementation, this preserves both the length and
    /// the position of every entry: an `NA::NA` at input index `i` becomes
    /// code `-1` at row `i` (not a dropped row), so
    /// `from_na_vec(v, ..).len() == v.len()` and
    /// `from_na_vec(v, ..).to_na_vec()` round-trips `v` back out (see
    /// [`Categorical::to_na_vec`]). Categories are inferred from the
    /// non-NA values (in first-seen order) when `categories` is `None`.
    pub fn from_na_vec(
        values: Vec<NA<T>>,
        categories: Option<Vec<T>>,
        ordered: Option<CategoricalOrder>,
    ) -> Result<Self> {
        let mut categories_list = if let Some(cats) = categories {
            cats
        } else {
            let mut unique = Vec::new();
            for v in &values {
                if let NA::Value(val) = v {
                    if !unique.contains(val) {
                        unique.push(val.clone());
                    }
                }
            }
            unique
        };

        let mut category_to_code: HashMap<T, i32> = HashMap::new();
        for (i, cat) in categories_list.iter().enumerate() {
            category_to_code.insert(cat.clone(), i as i32);
        }

        let mut codes = Vec::with_capacity(values.len());
        for v in &values {
            match v {
                NA::NA => codes.push(-1),
                NA::Value(val) => {
                    if let Some(&code) = category_to_code.get(val) {
                        codes.push(code);
                    } else {
                        let new_code = categories_list.len() as i32;
                        categories_list.push(val.clone());
                        category_to_code.insert(val.clone(), new_code);
                        codes.push(new_code);
                    }
                }
            }
        }

        Ok(Self {
            _phantom: std::marker::PhantomData,
            categories_list,
            codes,
            ordered_flag: ordered.map_or(false, |o| matches!(o, CategoricalOrder::Ordered)),
            category_to_code,
        })
    }

    /// Get the categories
    pub fn categories(&self) -> &Vec<T> {
        &self.categories_list
    }

    /// Get the length of the categorical data (including NA rows).
    ///
    /// This is the number of codes (one per logical row), not the number of
    /// non-NA values -- it stays correct for compact categoricals and for
    /// categoricals containing NA alike. Note this can be *larger* than
    /// [`Categorical::to_series`]'s output length whenever any row is NA,
    /// since `Series<T>` has no NA representation to preserve that row with
    /// (see that method's docs).
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if the categorical data is empty
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Get the category codes
    pub fn codes(&self) -> &Vec<i32> {
        &self.codes
    }

    /// Get the order status
    pub fn ordered(&self) -> CategoricalOrder {
        if self.ordered_flag {
            CategoricalOrder::Ordered
        } else {
            CategoricalOrder::Unordered
        }
    }

    /// Set the order status
    pub fn set_ordered(&mut self, order: CategoricalOrder) {
        self.ordered_flag = matches!(order, CategoricalOrder::Ordered);
    }

    /// Get value at index.
    ///
    /// Returns `None` both when `index` is out of range and when row
    /// `index` is NA (code `-1`) -- resolved through `codes` +
    /// `categories_list` like every other accessor here, so this reflects
    /// the categorical's *current* state (e.g. after
    /// [`Categorical::remove_categories`] orphans a row's code) rather than
    /// a stale copy of the value first passed in.
    pub fn get(&self, index: usize) -> Option<&T> {
        let code = *self.codes.get(index)?;
        if code < 0 {
            None
        } else {
            self.categories_list.get(code as usize)
        }
    }

    /// Convert categorical to series.
    ///
    /// `Series<T>` has no NA representation, so any row whose code is `-1`
    /// (from [`Categorical::from_na_vec`], or from
    /// [`Categorical::remove_categories`] orphaning a row) is dropped
    /// rather than fabricating a placeholder `T` for it: the result's
    /// length is the *non-NA* count, which can be shorter than
    /// [`Categorical::len`]. Callers that need the NA rows preserved should
    /// use [`Categorical::to_na_series`] instead, which returns an
    /// NA-aware `NASeries<T>` of exactly `len()` rows.
    pub fn to_series(&self, name: Option<String>) -> Result<Series<T>>
    where
        T: 'static + Clone + Debug + Send + Sync,
    {
        Series::new(self.materialized_values(), name)
    }

    /// Reorder categories.
    ///
    /// `new_categories` must contain exactly the same categories as the
    /// current list (any order, no additions, removals, or duplicates) --
    /// matching pandas' `reorder_categories`, which raises under the same
    /// conditions. Every row's code is remapped so it keeps pointing at the
    /// same logical value after the reorder (the previous implementation
    /// swapped `categories_list` without touching `codes` at all, silently
    /// repointing every row at whatever category ended up at its old
    /// numeric code).
    pub fn reorder_categories(&mut self, new_categories: Vec<T>) -> Result<()> {
        if new_categories.len() != self.categories_list.len() {
            return Err(Error::InvalidValue(format!(
                "reorder_categories: expected {} categories, got {}",
                self.categories_list.len(),
                new_categories.len()
            )));
        }

        let mut new_category_to_code: HashMap<T, i32> = HashMap::new();
        for (i, cat) in new_categories.iter().enumerate() {
            if new_category_to_code.insert(cat.clone(), i as i32).is_some() {
                return Err(Error::InvalidValue(
                    "reorder_categories: new_categories contains a duplicate".to_string(),
                ));
            }
        }
        for old_cat in &self.categories_list {
            if !new_category_to_code.contains_key(old_cat) {
                return Err(Error::InvalidValue(format!(
                    "reorder_categories: category {:?} is missing from new_categories",
                    old_cat
                )));
            }
        }

        // Old code `i` (index into the old `categories_list`) maps to
        // whatever code that same category now has in `new_categories`.
        let old_to_new: Vec<i32> = self
            .categories_list
            .iter()
            .map(|old_cat| new_category_to_code.get(old_cat).copied().unwrap_or(-1))
            .collect();

        for code in &mut self.codes {
            if *code >= 0 {
                *code = old_to_new[*code as usize];
            }
        }

        self.category_to_code = new_category_to_code;
        self.categories_list = new_categories;
        Ok(())
    }

    /// Add new categories to the end of the category list.
    ///
    /// Categories already present are left untouched rather than erroring
    /// (a deliberate, documented divergence from pandas' `add_categories`,
    /// which raises `ValueError` for a category that already exists --
    /// silently ignoring the duplicate keeps this idempotent). Existing
    /// codes are unaffected since new categories are only ever appended.
    pub fn add_categories(&mut self, new_categories: Vec<T>) -> Result<()> {
        for cat in new_categories {
            if !self.category_to_code.contains_key(&cat) {
                let new_code = self.categories_list.len() as i32;
                self.category_to_code.insert(cat.clone(), new_code);
                self.categories_list.push(cat);
            }
        }
        Ok(())
    }

    /// Remove categories.
    ///
    /// Every row that belonged to a removed category becomes NA (code
    /// `-1`), matching pandas' `remove_categories`. Remaining codes are
    /// remapped so they still index correctly into the shrunk category
    /// list -- the same remapping [`Categorical::remove_unused_categories`]
    /// performs, just driven by an explicit removal list instead of
    /// "unused" detection. (Previously `categories_list` was filtered
    /// without touching `codes` at all, so surviving rows silently ended up
    /// pointing at the wrong category, or out of bounds, once the list
    /// shrank underneath their unchanged numeric codes.)
    pub fn remove_categories(&mut self, categories_to_remove: &[T]) -> Result<()> {
        let remove_set: std::collections::HashSet<&T> = categories_to_remove.iter().collect();

        let mut new_categories = Vec::with_capacity(self.categories_list.len());
        let mut old_to_new: Vec<i32> = Vec::with_capacity(self.categories_list.len());
        for cat in &self.categories_list {
            if remove_set.contains(cat) {
                old_to_new.push(-1);
            } else {
                old_to_new.push(new_categories.len() as i32);
                new_categories.push(cat.clone());
            }
        }

        for code in &mut self.codes {
            if *code >= 0 {
                *code = old_to_new[*code as usize];
            }
        }

        self.category_to_code.clear();
        for (i, cat) in new_categories.iter().enumerate() {
            self.category_to_code.insert(cat.clone(), i as i32);
        }
        self.categories_list = new_categories;
        Ok(())
    }

    /// Count value occurrences.
    ///
    /// The result has one entry per category, in the same order as
    /// [`Categorical::categories`] -- `value_counts()?.values()[i]` is the
    /// occurrence count of `categories()[i]`, including categories with a
    /// count of zero. This is deterministic (the previous implementation
    /// iterated a `HashMap`, so both the row order *and* the mapping from a
    /// count back to its category -- which was computed into a local
    /// `indices` vector and then never attached to the returned `Series`,
    /// since `Series<T>` carries no index -- varied from call to call and
    /// could not be recovered at all). NA rows are excluded, matching
    /// pandas' `value_counts(dropna=True)` default. Callers that want the
    /// labels alongside the counts can `cat.categories().iter().zip(...)`.
    pub fn value_counts(&self) -> Result<Series<usize>> {
        let mut counts = vec![0usize; self.categories_list.len()];
        for &code in &self.codes {
            if code >= 0 {
                counts[code as usize] += 1;
            }
        }
        Series::new(counts, Some("count".to_string()))
    }

    /// Convert categorical data to a vector of NA values.
    ///
    /// This is [`Categorical::decode`] with each row rewrapped as `NA<T>`
    /// (`NA::Value` for a resolved code, `NA::NA` for code `-1`), so the
    /// output always has exactly `len()` entries in row order -- it is the
    /// inverse of [`Categorical::from_na_vec`]:
    /// `Categorical::from_na_vec(v, cats, ord)?.to_na_vec() == v` (up to
    /// category inference when `cats` is `None`). Previously this ignored
    /// codes entirely and just wrapped the (possibly already NA-dropping)
    /// raw values in `NA::Value`, so no output ever contained `NA::NA`.
    pub fn to_na_vec(&self) -> Vec<NA<T>>
    where
        T: Clone,
    {
        self.decode()
            .into_iter()
            .map(|opt| opt.map_or(NA::NA, NA::Value))
            .collect()
    }

    /// Convert categorical data to an NASeries
    pub fn to_na_series(&self, name: Option<String>) -> Result<NASeries<T>>
    where
        T: 'static + Clone + Debug + Send + Sync,
    {
        // Create NASeries from values
        NASeries::new(self.to_na_vec(), name)
    }

    /// Union of two categoricals
    pub fn union(&self, other: &Self) -> Result<Self> {
        // Combine categories from both sets and make unique
        let mut all_categories = self.categories_list.clone();

        for cat in other.categories() {
            if !all_categories.contains(cat) {
                all_categories.push(cat.clone());
            }
        }

        // Create a new categorical with the combined categories
        // For simplicity, just use self's (non-NA) values
        Self::new(
            self.materialized_values(),
            Some(all_categories),
            self.ordered_flag,
        )
    }

    /// Intersection of two categoricals
    pub fn intersection(&self, other: &Self) -> Result<Self> {
        // Keep only categories that appear in both categoricals
        let mut common_categories = Vec::new();

        for cat in self.categories() {
            if other.categories().contains(cat) {
                common_categories.push(cat.clone());
            }
        }

        // Filter values to only include those in common categories
        let filtered_values: Vec<T> = self
            .materialized_values()
            .into_iter()
            .filter(|v| common_categories.contains(v))
            .collect();

        // Create a new categorical with the common categories
        Self::new(filtered_values, Some(common_categories), self.ordered_flag)
    }

    /// Difference of two categoricals (self - other)
    pub fn difference(&self, other: &Self) -> Result<Self> {
        // Keep only categories that appear in self but not in other
        let mut diff_categories = Vec::new();

        for cat in self.categories() {
            if !other.categories().contains(cat) {
                diff_categories.push(cat.clone());
            }
        }

        // Filter values to only include those in diff categories
        let filtered_values: Vec<T> = self
            .materialized_values()
            .into_iter()
            .filter(|v| diff_categories.contains(v))
            .collect();

        // Create a new categorical with the different categories
        Self::new(filtered_values, Some(diff_categories), self.ordered_flag)
    }
}

/// String categorical type - convenience alias
pub type StringCategorical = Categorical<String>;
