use std::path::Path;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;

/// Serialization functionality for DataFrames
pub trait SerializeExt {
    /// Save DataFrame to a CSV file
    fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<()>;

    /// Load DataFrame from a CSV file
    fn from_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<Self>
    where
        Self: Sized;

    /// Convert DataFrame to JSON string
    fn to_json(&self) -> Result<String>;

    /// Create DataFrame from JSON string
    fn from_json(json: &str) -> Result<Self>
    where
        Self: Sized;

    /// Save DataFrame to a Parquet file
    fn to_parquet<P: AsRef<Path>>(&self, path: P) -> Result<()>;

    /// Load DataFrame from a Parquet file
    fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self>
    where
        Self: Sized;
}

impl SerializeExt for DataFrame {
    fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        // Delegate to the real (inherent) CSV writer on `DataFrame`.
        DataFrame::to_csv(self, path)
    }

    fn from_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<Self> {
        // Delegate to the real CSV reader in `crate::io::csv`.
        crate::io::csv::read_csv(path, has_header)
    }

    fn to_json(&self) -> Result<String> {
        use serde_json::{Map, Value};

        let mut object = Map::new();
        for col_name in self.column_names() {
            // Emit native JSON value types based on the real column storage type;
            // fall back to strings for everything else.
            let array: Vec<Value> = if let Ok(series) = self.get_column::<i64>(col_name) {
                series.values().iter().map(|v| Value::from(*v)).collect()
            } else if let Ok(series) = self.get_column::<f64>(col_name) {
                series.values().iter().map(|v| Value::from(*v)).collect()
            } else if let Ok(series) = self.get_column::<bool>(col_name) {
                series.values().iter().map(|v| Value::from(*v)).collect()
            } else {
                self.get_column_string_values(col_name)?
                    .into_iter()
                    .map(Value::from)
                    .collect()
            };
            object.insert(col_name.clone(), Value::Array(array));
        }

        serde_json::to_string(&Value::Object(object))
            .map_err(|e| Error::InvalidInput(format!("JSON serialization failed: {}", e)))
    }

    fn from_json(json: &str) -> Result<Self> {
        // Delegate to the real (inherent) JSON parser on `DataFrame`.
        DataFrame::from_json(json)
    }

    fn to_parquet<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        #[cfg(feature = "parquet")]
        {
            write_parquet_file(self, path)
        }
        #[cfg(not(feature = "parquet"))]
        {
            let _ = path;
            Err(Error::NotImplemented(
                "Parquet support requires the 'parquet' feature".to_string(),
            ))
        }
    }

    fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        #[cfg(feature = "parquet")]
        {
            crate::io::parquet::read_parquet(path)
        }
        #[cfg(not(feature = "parquet"))]
        {
            let _ = path;
            Err(Error::NotImplemented(
                "Parquet support requires the 'parquet' feature".to_string(),
            ))
        }
    }
}

/// Write a DataFrame to a Parquet file by building Arrow arrays directly from
/// the real column data. Numeric and boolean columns keep their native Arrow
/// type; all other columns are written as UTF-8 (lossless string form).
#[cfg(feature = "parquet")]
fn write_parquet_file<P: AsRef<Path>>(df: &DataFrame, path: P) -> Result<()> {
    use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let mut fields: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for col_name in df.column_names() {
        let (field, array): (Field, ArrayRef) = if let Ok(series) = df.get_column::<i64>(col_name) {
            (
                Field::new(col_name, DataType::Int64, false),
                Arc::new(Int64Array::from(series.values().to_vec())),
            )
        } else if let Ok(series) = df.get_column::<f64>(col_name) {
            (
                Field::new(col_name, DataType::Float64, false),
                Arc::new(Float64Array::from(series.values().to_vec())),
            )
        } else if let Ok(series) = df.get_column::<bool>(col_name) {
            (
                Field::new(col_name, DataType::Boolean, false),
                Arc::new(BooleanArray::from(series.values().to_vec())),
            )
        } else {
            let values = df.get_column_string_values(col_name)?;
            (
                Field::new(col_name, DataType::Utf8, false),
                Arc::new(StringArray::from_iter_values(values)),
            )
        };
        fields.push(field);
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| Error::InvalidOperation(format!("Failed to build Arrow batch: {}", e)))?;

    let file = std::fs::File::create(path.as_ref())
        .map_err(|e| Error::IoError(format!("Failed to create Parquet file: {}", e)))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|e| Error::IoError(format!("Failed to create Parquet writer: {}", e)))?;
    writer
        .write(&batch)
        .map_err(|e| Error::IoError(format!("Failed to write Parquet batch: {}", e)))?;
    writer
        .close()
        .map_err(|e| Error::IoError(format!("Failed to finalize Parquet file: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    fn sample_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column("age".to_string(), Series::new(vec![1i64, 2], None).unwrap())
            .unwrap();
        df.add_column(
            "name".to_string(),
            Series::new(vec!["a".to_string(), "b".to_string()], None).unwrap(),
        )
        .unwrap();
        df
    }

    #[test]
    fn test_serialize_ext_csv_round_trip() {
        let df = sample_df();
        let mut path = std::env::temp_dir();
        path.push(format!("pandrs_serext_csv_{}.csv", std::process::id()));

        // Exercise the trait methods explicitly (not the inherent shadows).
        SerializeExt::to_csv(&df, &path).unwrap();
        let loaded = <DataFrame as SerializeExt>::from_csv(&path, true).unwrap();

        assert_eq!(loaded.row_count(), 2);
        assert_eq!(
            loaded.get_column_string_values("name").unwrap(),
            vec!["a", "b"]
        );
        assert_eq!(
            loaded.get_column_string_values("age").unwrap(),
            vec!["1", "2"]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_serialize_ext_json_round_trip() {
        let df = sample_df();
        let json = SerializeExt::to_json(&df).unwrap();
        // Numeric columns are emitted as JSON numbers.
        assert!(json.contains("\"age\""));
        assert!(json.contains('1'));

        let loaded = <DataFrame as SerializeExt>::from_json(&json).unwrap();
        assert_eq!(loaded.row_count(), 2);
        assert_eq!(
            loaded.get_column_string_values("name").unwrap(),
            vec!["a", "b"]
        );
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_serialize_ext_parquet_round_trip() {
        let mut df = sample_df();
        df.add_column(
            "score".to_string(),
            Series::new(vec![1.5f64, 2.5], None).unwrap(),
        )
        .unwrap();

        let mut path = std::env::temp_dir();
        path.push(format!(
            "pandrs_serext_parquet_{}.parquet",
            std::process::id()
        ));

        SerializeExt::to_parquet(&df, &path).unwrap();
        let loaded = <DataFrame as SerializeExt>::from_parquet(&path).unwrap();

        assert_eq!(loaded.row_count(), 2);
        assert_eq!(loaded.column_names().len(), 3);
        assert_eq!(
            loaded.get_column_string_values("name").unwrap(),
            vec!["a", "b"]
        );
        assert_eq!(
            loaded.get_column_string_values("age").unwrap(),
            vec!["1", "2"]
        );
        assert_eq!(
            loaded.get_column_string_values("score").unwrap(),
            vec!["1.5", "2.5"]
        );
        let _ = std::fs::remove_file(&path);
    }
}
