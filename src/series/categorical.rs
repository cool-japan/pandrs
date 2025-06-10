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

/// Categorical data type
#[derive(Debug, Clone)]
pub struct Categorical<T>
where
    T: Debug + Clone + Eq + Hash,
{
    // This is a stub implementation - we'll develop this further
    // Currently just forwarding to the legacy implementation
    _phantom: std::marker::PhantomData<T>,
    categories_list: Vec<T>,
    values: Vec<T>,
    codes: Vec<i32>, // Mapping from values to categories index
    ordered_flag: bool,
}

impl<T> Categorical<T>
where
    T: Debug + Clone + Eq + Hash,
{
    /// Create a new Categorical
    pub fn new(values: Vec<T>, categories: Option<Vec<T>>, ordered: bool) -> Result<Self> {
        // For our stub implementation, we'll just store the values and categories
        // In a real implementation, we would compute codes and validate data
        let categories_list = categories.unwrap_or_else(|| values.clone());

        // Create a simple code representation (just 0s for now)
        // In a real implementation, we'd map values to their category index
        let codes = vec![0; values.len()];

        Ok(Self {
            _phantom: std::marker::PhantomData,
            categories_list,
            values,
            codes,
            ordered_flag: ordered,
        })
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

        // Create a new categorical with the common categories
        Self::new(
            self.values.clone(),
            Some(common_categories),
            self.ordered_flag,
        )
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

        // Create a new categorical with the differnt categories
        Self::new(
            self.values.clone(),
            Some(diff_categories),
            self.ordered_flag,
        )
    }
}

/// String categorical type - convenience alias
pub type StringCategorical = Categorical<String>;
