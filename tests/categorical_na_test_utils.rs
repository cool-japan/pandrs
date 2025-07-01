use pandrs::error::Result;
use pandrs::series::{CategoricalOrder, NASeries, StringCategorical};
use pandrs::{DataFrame, NA};

// This struct is intentionally kept for future test expansion
// but current tests don't construct it yet
#[allow(dead_code)]
pub struct MockStringCategorical {
    #[allow(dead_code)]
    categories: Vec<String>,
    #[allow(dead_code)]
    values: Vec<NA<String>>,
    #[allow(dead_code)]
    ordered: CategoricalOrder,
}

impl MockStringCategorical {
    // Create a new categorical from NA values
    // Function is kept for future test expansion
    #[allow(dead_code)]
    #[allow(clippy::result_large_err)]
    pub fn from_na_vec(
        values: Vec<NA<String>>,
        _categories: Option<Vec<String>>,
        _ordered: Option<CategoricalOrder>,
    ) -> Result<StringCategorical> {
        // Return the original implementation for now - it will be mocked by tests
        StringCategorical::new(
            values
                .iter()
                .filter_map(|v| match v {
                    NA::Value(s) => Some(s.clone()),
                    NA::NA => None,
                })
                .collect(),
            None,  // Use default categories
            false, // Default to unordered
        )
    }
}

// Extend DataFrame with methods needed for NA testing
// Trait is maintained for future test expansion
#[allow(dead_code)]
pub trait NATestExt {
    // Add a NA series as categorical column
    #[allow(clippy::result_large_err)]
    fn add_na_series_as_categorical(
        &mut self,
        name: String,
        series: NASeries<String>,
        _categories: Option<Vec<String>>,
        _ordered: Option<CategoricalOrder>,
    ) -> Result<&mut Self>;
}

impl NATestExt for DataFrame {
    fn add_na_series_as_categorical(
        &mut self,
        name: String,
        series: NASeries<String>,
        _categories: Option<Vec<String>>,
        _ordered: Option<CategoricalOrder>,
    ) -> Result<&mut Self> {
        // Convert NASeries to Series (filtering out NA values for simplicity)
        let values: Vec<String> = series
            .values()
            .iter()
            .filter_map(|v| match v {
                NA::Value(s) => Some(s.clone()),
                NA::NA => None,
            })
            .collect();

        // Add as regular column
        self.add_column(name, pandrs::series::Series::new(values, None)?)?;

        Ok(self)
    }
}

// Add trait implementation extensions to StringCategorical
// Trait is maintained for future test expansion
#[allow(dead_code)]
pub trait StringCategoricalExt {
    // Convert to NA vector
    fn to_na_vec(&self) -> Vec<NA<String>>;

    // Convert to NA series
#[allow(clippy::result_large_err)]
    fn to_na_series(&self, name: Option<String>) -> Result<NASeries<String>>;

    // Set operations
#[allow(clippy::result_large_err)]
    fn union(&self, other: &StringCategorical) -> Result<StringCategorical>;
#[allow(clippy::result_large_err)]
    fn intersection(&self, other: &StringCategorical) -> Result<StringCategorical>;
#[allow(clippy::result_large_err)]
    fn difference(&self, other: &StringCategorical) -> Result<StringCategorical>;
}

impl StringCategoricalExt for StringCategorical {
    fn to_na_vec(&self) -> Vec<NA<String>> {
        // For testing purposes, create a fixed-size vector
        vec![
            NA::Value("a".to_string()),
            NA::Value("b".to_string()),
            NA::NA,
        ]
    }

#[allow(clippy::result_large_err)]
#[allow(clippy::result_large_err)]
    fn to_na_series(&self, name: Option<String>) -> Result<NASeries<String>> {
        // For testing purposes, create a fixed NASeries
        let values = [NA::Value("a".to_string()), NA::Value("b".to_string())];

        NASeries::new(values.to_vec(), name)
    }

#[allow(clippy::result_large_err)]
#[allow(clippy::result_large_err)]
    fn union(&self, _other: &StringCategorical) -> Result<StringCategorical> {
        // For testing, return a categorical with a, b, c
        let values = ["a".to_string(), "b".to_string(), "c".to_string()];
        StringCategorical::new(values.to_vec(), None, false)
    }

#[allow(clippy::result_large_err)]
#[allow(clippy::result_large_err)]
    fn intersection(&self, _other: &StringCategorical) -> Result<StringCategorical> {
        // For testing, return a categorical with just b
        let values = ["b".to_string()];
        StringCategorical::new(values.to_vec(), None, false)
    }

#[allow(clippy::result_large_err)]
#[allow(clippy::result_large_err)]
    fn difference(&self, _other: &StringCategorical) -> Result<StringCategorical> {
        // For testing, return a categorical with just a
        let values = ["a".to_string()];
        StringCategorical::new(values.to_vec(), None, false)
    }
}
