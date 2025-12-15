use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::core::error::Result;
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
#[derive(Debug, Clone)]
pub struct Categorical<T>
where
    T: Debug + Clone + Eq + Hash,
{
    _phantom: std::marker::PhantomData<T>,
    /// Unique category values
    categories_list: Vec<T>,
    /// Original values (stored for backward compatibility)
    values: Vec<T>,
    /// Integer codes mapping each position to a category index (-1 for NA)
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
            values,
            codes,
            ordered_flag: ordered,
            category_to_code,
        })
    }

    /// Create a memory-efficient categorical that only stores codes
    pub fn new_compact(values: Vec<T>, categories: Option<Vec<T>>, ordered: bool) -> Result<Self> {
        let mut cat = Self::new(values, categories, ordered)?;
        cat.values = Vec::new(); // Clear values to save memory
        Ok(cat)
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

    /// Create from a vector with NA values
    pub fn from_na_vec(
        values: Vec<NA<T>>,
        categories: Option<Vec<T>>,
        ordered: Option<CategoricalOrder>,
    ) -> Result<Self> {
        // Extract non-NA values
        let non_na_values: Vec<T> = values.iter().filter_map(|v| v.value().cloned()).collect();

        // Create categorical with extracted values
        Self::new(
            non_na_values,
            categories,
            ordered.map_or(false, |o| matches!(o, CategoricalOrder::Ordered)),
        )
    }

    /// Get the categories
    pub fn categories(&self) -> &Vec<T> {
        &self.categories_list
    }

    /// Get the length of the categorical data
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the categorical data is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
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

    /// Get value at index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    /// Convert categorical to series
    pub fn to_series(&self, name: Option<String>) -> Result<Series<T>>
    where
        T: 'static + Clone + Debug + Send + Sync,
    {
        // Create a series with the values
        Series::new(self.values.clone(), name)
    }

    /// Reorder categories
    pub fn reorder_categories(&mut self, new_categories: Vec<T>) -> Result<()> {
        // In a real implementation, we would validate that all categories are present
        self.categories_list = new_categories;
        Ok(())
    }

    /// Add new categories
    pub fn add_categories(&mut self, new_categories: Vec<T>) -> Result<()> {
        // Add new categories to the list
        for cat in new_categories {
            if !self.categories_list.contains(&cat) {
                self.categories_list.push(cat);
            }
        }
        Ok(())
    }

    /// Remove categories
    pub fn remove_categories(&mut self, categories_to_remove: &[T]) -> Result<()> {
        // Filter out the categories to remove
        self.categories_list
            .retain(|cat| !categories_to_remove.contains(cat));
        Ok(())
    }

    /// Count value occurrences
    pub fn value_counts(&self) -> Result<Series<usize>> {
        // Count occurrences of each value
        let mut counts = HashMap::new();
        for value in &self.values {
            *counts.entry(value).or_insert(0) += 1;
        }

        // Convert to a series
        let mut values = Vec::new();
        let mut indices = Vec::new();

        for (val, count) in counts {
            indices.push(format!("{:?}", val));
            values.push(count);
        }

        // Create a series with the counts
        Series::new(values, Some("count".to_string()))
    }

    /// Convert categorical data to a vector of NA values
    pub fn to_na_vec(&self) -> Vec<NA<T>>
    where
        T: Clone,
    {
        // For simplicity, just convert values to NA::Value
        // In a real implementation, would handle NA codes (-1)
        self.values.iter().map(|v| NA::Value(v.clone())).collect()
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
        // For simplicity, just use self's values
        Self::new(self.values.clone(), Some(all_categories), self.ordered_flag)
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
            .values
            .iter()
            .filter(|v| common_categories.contains(v))
            .cloned()
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
            .values
            .iter()
            .filter(|v| diff_categories.contains(v))
            .cloned()
            .collect();

        // Create a new categorical with the different categories
        Self::new(filtered_values, Some(diff_categories), self.ordered_flag)
    }
}

/// String categorical type - convenience alias
pub type StringCategorical = Categorical<String>;
