//! Regression tests for the DataFrame <-> OptimizedDataFrame bridge
//! (`src/optimized/convert.rs`) and `OptimizedDataFrame`'s CSV/Parquet I/O
//! (`src/optimized/split_dataframe/io.rs`).
//!
//! Covers:
//! 1. C1: typed `Series<i64>`/`<f64>`/`<bool>` columns used to be silently
//!    dropped by `optimize_dataframe` (only `Series<String>` survived, so a
//!    typed DataFrame converted to an OptimizedDataFrame with 0 columns).
//! 2. Empty-string type inference: a blank cell is NULL, never silently
//!    coerced into `0` / `0.0` / `false`, and an all-blank column is never
//!    misclassified as numeric.
//! 3. `MultiIndex` preserved through `optimize_dataframe` (it used to be
//!    silently dropped on the SplitDataFrame -> OptimizedDataFrame leg).
//! 4. `standard_dataframe`'s NULL policy: `Int64Column` NULLs upcast to
//!    `Series<f64>`/NaN, `BooleanColumn` NULLs promote to `Series<String>`;
//!    NULL-free columns of either type keep their native, typed `Series<T>`.
//! 5. Headerless CSV no longer silently drops row 0.
//! 6. Parquet write preserves NULLs (not a fabricated placeholder value in
//!    every NULL slot).
//! 7. Parquet read preserves NULLs (not 0 / NaN / false / "").
//! 8. Extended Arrow type support on Parquet read (Int32, Date32, Date64,
//!    Timestamp with/without a timezone) instead of an O(n^2) debug-dump
//!    fallback, plus an honest error for a genuinely unsupported type.
//! 9. `LargeUtf8` Parquet columns are read correctly (the downcast used to
//!    unconditionally target `StringArray`, which always fails for
//!    `LargeUtf8`'s `LargeStringArray`).
//! 10. Requesting Zstd/Lzo Parquet compression fails with an explicit,
//!     actionable error instead of an opaque codec error, and
//!     `pandrs::io::ParquetCompression` /
//!     `pandrs::optimized::split_dataframe::io::ParquetCompression` are the
//!     same type (they used to be independently-defined duplicates).
//!
//! One more bug was found and fixed alongside these: `optimize_dataframe`
//! used to fail with a length-mismatch error for *any* `DataFrame` that
//! never had `set_index` called on it explicitly (the common case), because
//! `DataFrame::get_index()` synthesizes a zero-length placeholder index in
//! that case and the old code propagated that 0 into a strict
//! `index.len() == row_count` check.

use pandrs::index::DataFrameIndex;
use pandrs::optimized::{optimize_dataframe, standard_dataframe};
use pandrs::{
    BooleanColumn, Column, DataFrame, Float64Column, Int64Column, MultiIndex, OptimizedDataFrame,
    Series, StringColumn,
};

#[cfg(feature = "parquet")]
use pandrs::optimized::split_dataframe::io::ParquetCompression;

/// A unique path under the OS temp dir, as required by project test policy.
fn unique_temp_path(name: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pandrs_optimized_convert_io_w1_{}_{}_{}",
        std::process::id(),
        nanos,
        name
    ));
    p
}

// ---------------------------------------------------------------------
// Items 1 (C1) and 3: typed columns and MultiIndex survive
// optimize_dataframe.
// ---------------------------------------------------------------------

#[test]
fn typed_columns_all_survive_optimize_dataframe() {
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1i64, 2, 3], Some("id".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "score".to_string(),
        Series::new(vec![1.5f64, 2.5, 3.5], Some("score".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "flag".to_string(),
        Series::new(vec![true, false, true], Some("flag".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "name".to_string(),
        Series::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            Some("name".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    // C1 regression: this used to produce an OptimizedDataFrame with 0
    // columns for anything but Series<String>.
    let odf = optimize_dataframe(&df).expect("a typed DataFrame must convert");
    assert_eq!(odf.column_count(), 4);
    assert_eq!(odf.row_count(), 3);

    match odf.column("id").unwrap().column() {
        Column::Int64(c) => {
            assert_eq!(c.get(0).unwrap(), Some(1));
            assert_eq!(c.get(1).unwrap(), Some(2));
            assert_eq!(c.get(2).unwrap(), Some(3));
        }
        other => panic!("expected an Int64 column, got {:?}", other),
    }
    match odf.column("score").unwrap().column() {
        Column::Float64(c) => assert_eq!(c.get(1).unwrap(), Some(2.5)),
        other => panic!("expected a Float64 column, got {:?}", other),
    }
    match odf.column("flag").unwrap().column() {
        Column::Boolean(c) => assert_eq!(c.get(1).unwrap(), Some(false)),
        other => panic!("expected a Boolean column, got {:?}", other),
    }
    match odf.column("name").unwrap().column() {
        Column::String(c) => assert_eq!(c.get(1).unwrap(), Some("b")),
        other => panic!("expected a String column, got {:?}", other),
    }
}

#[test]
fn optimize_dataframe_does_not_require_an_explicit_index() {
    // `DataFrame::get_index()` returns a zero-length placeholder index when
    // `set_index` was never called; `optimize_dataframe` used to propagate
    // that 0 straight into a length check against the real row count and
    // fail every such conversion (i.e. the common case).
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1i64, 2, 3], Some("id".to_string())).unwrap(),
    )
    .unwrap();

    let odf = optimize_dataframe(&df).expect("a DataFrame without an explicit index must convert");
    assert_eq!(odf.row_count(), 3);
    assert_eq!(odf.column_count(), 1);
}

#[test]
fn multi_index_preserved_through_optimize_dataframe() {
    let levels = vec![
        vec!["x".to_string(), "y".to_string()],
        vec!["p".to_string(), "q".to_string()],
    ];
    let codes = vec![vec![0, 1], vec![0, 1]];
    let mi = MultiIndex::new(levels, codes, None).expect("valid MultiIndex");

    let mut df = DataFrame::new();
    df.add_column(
        "v".to_string(),
        Series::new(vec![10i64, 20], Some("v".to_string())).unwrap(),
    )
    .unwrap();
    df.set_multi_index(mi).unwrap();

    let odf = optimize_dataframe(&df).expect("a MultiIndex DataFrame must convert");
    match odf.get_index() {
        Some(DataFrameIndex::Multi(_)) => {}
        other => panic!(
            "expected the MultiIndex to survive optimize_dataframe, got {:?}",
            other
        ),
    }
}

// ---------------------------------------------------------------------
// Item 2: empty-string type inference must NULL-track, not coerce to 0.
// ---------------------------------------------------------------------

#[test]
fn empty_string_in_numeric_series_becomes_null_not_zero() {
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        Series::new(
            vec!["10".to_string(), "".to_string(), "30".to_string()],
            Some("a".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let odf = optimize_dataframe(&df).unwrap();
    match odf.column("a").unwrap().column() {
        Column::Int64(c) => {
            assert_eq!(c.get(0).unwrap(), Some(10));
            assert_eq!(c.get(1).unwrap(), None, "a blank cell must be NULL, not 0");
            assert_eq!(c.get(2).unwrap(), Some(30));
        }
        other => panic!("expected an Int64 column, got {:?}", other),
    }
}

#[test]
fn all_blank_string_series_is_not_misclassified_as_numeric() {
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        Series::new(
            vec!["".to_string(), "".to_string(), "".to_string()],
            Some("a".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let odf = optimize_dataframe(&df).unwrap();
    match odf.column("a").unwrap().column() {
        Column::String(c) => {
            assert_eq!(c.get(0).unwrap(), None);
            assert_eq!(c.get(1).unwrap(), None);
            assert_eq!(c.get(2).unwrap(), None);
        }
        other => panic!(
            "an all-blank column must stay a fully-NULL String column, not {:?}",
            other
        ),
    }
}

#[test]
fn nan_in_float_series_becomes_null() {
    let mut df = DataFrame::new();
    df.add_column(
        "f".to_string(),
        Series::new(vec![1.0f64, f64::NAN, 3.0], Some("f".to_string())).unwrap(),
    )
    .unwrap();

    let odf = optimize_dataframe(&df).unwrap();
    match odf.column("f").unwrap().column() {
        Column::Float64(c) => {
            assert_eq!(c.get(0).unwrap(), Some(1.0));
            assert_eq!(c.get(1).unwrap(), None);
            assert_eq!(c.get(2).unwrap(), Some(3.0));
        }
        other => panic!("expected a Float64 column, got {:?}", other),
    }
}

// ---------------------------------------------------------------------
// Item 4: standard_dataframe's NULL policy.
// ---------------------------------------------------------------------

#[test]
fn int64_column_with_nulls_upcasts_to_float64_series() {
    let mut odf = OptimizedDataFrame::new();
    odf.add_column(
        "n".to_string(),
        Column::Int64(Int64Column::with_nulls(
            vec![1, 0, 3],
            vec![false, true, false],
        )),
    )
    .unwrap();

    let std_df = standard_dataframe(&odf).expect("must convert");
    let series = std_df
        .get_column::<f64>("n")
        .expect("an Int64Column with a NULL must become Series<f64>, not Series<i64>");
    assert_eq!(series.values()[0], 1.0);
    assert!(series.values()[1].is_nan(), "the NULL position must be NaN");
    assert_eq!(series.values()[2], 3.0);

    assert!(
        std_df.get_column::<i64>("n").is_err(),
        "a NULL-containing Int64Column must not also be readable as Series<i64>"
    );
}

#[test]
fn int64_column_without_nulls_stays_series_i64() {
    let mut odf = OptimizedDataFrame::new();
    odf.add_column(
        "n".to_string(),
        Column::Int64(Int64Column::new(vec![1, 2, 3])),
    )
    .unwrap();

    let std_df = standard_dataframe(&odf).expect("must convert");
    let series = std_df
        .get_column::<i64>("n")
        .expect("a NULL-free Int64Column must stay a native Series<i64>");
    assert_eq!(series.values(), &[1, 2, 3]);
}

#[test]
fn boolean_column_with_nulls_promotes_to_string_series() {
    let mut odf = OptimizedDataFrame::new();
    odf.add_column(
        "b".to_string(),
        Column::Boolean(BooleanColumn::with_nulls(
            vec![true, false, false],
            vec![false, true, false],
        )),
    )
    .unwrap();

    let std_df = standard_dataframe(&odf).expect("must convert");
    let series = std_df
        .get_column::<String>("b")
        .expect("a BooleanColumn with a NULL must become Series<String>");
    assert_eq!(
        series.values(),
        &["true".to_string(), "".to_string(), "false".to_string()]
    );
}

#[test]
fn boolean_column_without_nulls_stays_series_bool() {
    let mut odf = OptimizedDataFrame::new();
    odf.add_column(
        "b".to_string(),
        Column::Boolean(BooleanColumn::new(vec![true, false, true])),
    )
    .unwrap();

    let std_df = standard_dataframe(&odf).expect("must convert");
    let series = std_df
        .get_column::<bool>("b")
        .expect("a NULL-free BooleanColumn must stay a native Series<bool>");
    assert_eq!(series.values(), &[true, false, true]);
}

// ---------------------------------------------------------------------
// Item 5: headerless CSV must not drop row 0; blank cells are NULL.
// ---------------------------------------------------------------------

#[test]
fn headerless_csv_preserves_the_first_row() {
    let path = unique_temp_path("headerless.csv");
    std::fs::write(&path, "1,Alice,85.5\n2,Bob,92.0\n3,Charlie,78.3\n").unwrap();

    let df = OptimizedDataFrame::from_csv(&path, false).expect("headerless CSV must load");
    let _ = std::fs::remove_file(&path);

    assert_eq!(df.row_count(), 3, "row 0 must not be silently dropped");
    match df.column("column_0").unwrap().column() {
        Column::Int64(c) => {
            assert_eq!(c.get(0).unwrap(), Some(1));
            assert_eq!(c.get(1).unwrap(), Some(2));
            assert_eq!(c.get(2).unwrap(), Some(3));
        }
        other => panic!("expected an Int64 column, got {:?}", other),
    }
    match df.column("column_1").unwrap().column() {
        Column::String(c) => assert_eq!(c.get(0).unwrap(), Some("Alice")),
        other => panic!("expected a String column, got {:?}", other),
    }
}

#[test]
fn csv_blank_cells_become_null_in_numeric_and_string_columns() {
    let path = unique_temp_path("csv_nulls.csv");
    std::fs::write(&path, "a,b\n1,x\n,y\n3,\n").unwrap();

    let df = OptimizedDataFrame::from_csv(&path, true).expect("CSV must load");
    let _ = std::fs::remove_file(&path);

    match df.column("a").unwrap().column() {
        Column::Int64(c) => {
            assert_eq!(c.get(0).unwrap(), Some(1));
            assert_eq!(c.get(1).unwrap(), None, "a blank numeric cell must be NULL");
            assert_eq!(c.get(2).unwrap(), Some(3));
        }
        other => panic!("expected an Int64 column, got {:?}", other),
    }
    match df.column("b").unwrap().column() {
        Column::String(c) => {
            assert_eq!(c.get(0).unwrap(), Some("x"));
            assert_eq!(c.get(1).unwrap(), Some("y"));
            assert_eq!(c.get(2).unwrap(), None, "a blank string cell must be NULL");
        }
        other => panic!("expected a String column, got {:?}", other),
    }
}

// ---------------------------------------------------------------------
// Items 6/7/10: Parquet NULL round-trip and compression policy.
// ---------------------------------------------------------------------

#[cfg(feature = "parquet")]
#[test]
fn parquet_round_trip_preserves_nulls_in_every_column_type() {
    let mut src = OptimizedDataFrame::new();
    src.add_column(
        "i".to_string(),
        Column::Int64(Int64Column::with_nulls(
            vec![1, 0, 3],
            vec![false, true, false],
        )),
    )
    .unwrap();
    src.add_column(
        "f".to_string(),
        Column::Float64(Float64Column::with_nulls(
            vec![1.5, 0.0, 3.5],
            vec![false, true, false],
        )),
    )
    .unwrap();
    src.add_column(
        "bo".to_string(),
        Column::Boolean(BooleanColumn::with_nulls(
            vec![true, false, false],
            vec![false, true, false],
        )),
    )
    .unwrap();
    src.add_column(
        "s".to_string(),
        Column::String(StringColumn::with_nulls(
            vec!["x".to_string(), "".to_string(), "z".to_string()],
            vec![false, true, false],
        )),
    )
    .unwrap();

    let path = unique_temp_path("nulls.parquet");
    src.to_parquet(&path, Some(ParquetCompression::Snappy))
        .expect("write must succeed");
    let loaded = OptimizedDataFrame::from_parquet(&path).expect("read must succeed");
    let _ = std::fs::remove_file(&path);

    match loaded.column("i").unwrap().column() {
        Column::Int64(c) => {
            assert_eq!(c.get(0).unwrap(), Some(1));
            assert_eq!(c.get(1).unwrap(), None);
            assert_eq!(c.get(2).unwrap(), Some(3));
        }
        other => panic!("expected an Int64 column, got {:?}", other),
    }
    match loaded.column("f").unwrap().column() {
        Column::Float64(c) => {
            assert_eq!(c.get(0).unwrap(), Some(1.5));
            assert_eq!(c.get(1).unwrap(), None);
            assert_eq!(c.get(2).unwrap(), Some(3.5));
        }
        other => panic!("expected a Float64 column, got {:?}", other),
    }
    match loaded.column("bo").unwrap().column() {
        Column::Boolean(c) => {
            assert_eq!(c.get(0).unwrap(), Some(true));
            assert_eq!(c.get(1).unwrap(), None);
            assert_eq!(c.get(2).unwrap(), Some(false));
        }
        other => panic!("expected a Boolean column, got {:?}", other),
    }
    match loaded.column("s").unwrap().column() {
        Column::String(c) => {
            assert_eq!(c.get(0).unwrap(), Some("x"));
            assert_eq!(c.get(1).unwrap(), None);
            assert_eq!(c.get(2).unwrap(), Some("z"));
        }
        other => panic!("expected a String column, got {:?}", other),
    }
}

#[cfg(feature = "parquet")]
#[test]
fn zstd_compression_returns_explicit_policy_error() {
    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "n".to_string(),
        Column::Int64(Int64Column::new(vec![1, 2, 3])),
    )
    .unwrap();
    let path = unique_temp_path("zstd.parquet");

    let err = df
        .to_parquet(&path, Some(ParquetCompression::Zstd))
        .expect_err("Zstd must be rejected, not silently attempted");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("zstd") && msg.contains("pure rust"),
        "error should name the Pure Rust policy reason, got: {}",
        err
    );
    assert!(!path.exists(), "no partial file should be left behind");
}

#[cfg(feature = "parquet")]
#[test]
fn lzo_compression_returns_explicit_error() {
    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "n".to_string(),
        Column::Int64(Int64Column::new(vec![1, 2, 3])),
    )
    .unwrap();
    let path = unique_temp_path("lzo.parquet");

    let err = df
        .to_parquet(&path, Some(ParquetCompression::Lzo))
        .expect_err("Lzo must be rejected, not silently attempted");
    assert!(err.to_string().to_lowercase().contains("lzo"));
}

#[cfg(feature = "parquet")]
#[test]
fn parquet_compression_enum_is_shared_between_the_two_public_paths() {
    // Regression: `crate::optimized::split_dataframe::io::ParquetCompression`
    // used to be an independently-defined, identically-named-but-unrelated
    // enum from `pandrs::io::ParquetCompression`, so a value obtained from
    // one path did not type-check where the other was expected.
    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "n".to_string(),
        Column::Int64(Int64Column::new(vec![1, 2, 3])),
    )
    .unwrap();
    let path = unique_temp_path("shared_enum.parquet");
    df.to_parquet(&path, Some(pandrs::io::ParquetCompression::Snappy))
        .expect("pandrs::io::ParquetCompression must be directly usable here");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------
// Items 8/9: extended Arrow type support and the LargeUtf8 downcast fix.
// These need direct arrow/parquet crate access to synthesize column types
// this crate's own writer never produces, so they build the Parquet file
// with the `arrow`/`parquet` crates directly rather than round-tripping
// through `OptimizedDataFrame::to_parquet`.
// ---------------------------------------------------------------------

#[cfg(feature = "parquet")]
#[test]
fn parquet_extended_arrow_types_and_large_utf8_are_read_correctly() {
    use arrow::array::{
        ArrayRef, Date32Array, Date64Array, Int32Array, LargeStringArray, TimestampMillisecondArray,
    };
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_writer::ArrowWriter;
    use std::fs::File;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new("i32", DataType::Int32, true),
        Field::new("d32", DataType::Date32, true),
        Field::new("d64", DataType::Date64, true),
        Field::new(
            "ts_naive",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        Field::new(
            "ts_utc",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            true,
        ),
        Field::new("large_s", DataType::LargeUtf8, true),
    ]));

    // 1 day + 1h1m1s after the Unix epoch, so both the date and
    // time-of-day components are exercised (not trivially zero).
    const ONE_DAY_MS: i64 = 86_400_000;
    const OFFSET_MS: i64 = ONE_DAY_MS + 3_600_000 + 60_000 + 1_000; // + 1h 1m 1s

    let i32_arr: ArrayRef = Arc::new(Int32Array::from(vec![Some(42i32), None]));
    let d32_arr: ArrayRef = Arc::new(Date32Array::from(vec![Some(1i32), None]));
    let d64_arr: ArrayRef = Arc::new(Date64Array::from(vec![Some(OFFSET_MS), None]));
    let ts_naive_arr: ArrayRef =
        Arc::new(TimestampMillisecondArray::from(vec![Some(OFFSET_MS), None]));
    let ts_utc_arr: ArrayRef =
        Arc::new(TimestampMillisecondArray::from(vec![Some(OFFSET_MS), None]).with_timezone("UTC"));
    let large_s_arr: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("hello"), None]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            i32_arr,
            d32_arr,
            d64_arr,
            ts_naive_arr,
            ts_utc_arr,
            large_s_arr,
        ],
    )
    .unwrap();

    let path = unique_temp_path("extended_types.parquet");
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let df = OptimizedDataFrame::from_parquet(&path).expect("extended types must be readable");
    let _ = std::fs::remove_file(&path);

    match df.column("i32").unwrap().column() {
        Column::Int64(c) => {
            assert_eq!(c.get(0).unwrap(), Some(42));
            assert_eq!(c.get(1).unwrap(), None);
        }
        other => panic!("Int32 should widen to Int64, got {:?}", other),
    }
    match df.column("d32").unwrap().column() {
        Column::String(c) => {
            assert_eq!(c.get(0).unwrap(), Some("1970-01-02"));
            assert_eq!(c.get(1).unwrap(), None);
        }
        other => panic!("Date32 should become an ISO date string, got {:?}", other),
    }
    match df.column("d64").unwrap().column() {
        Column::String(c) => {
            assert_eq!(c.get(0).unwrap(), Some("1970-01-02T01:01:01"));
            assert_eq!(c.get(1).unwrap(), None);
        }
        other => panic!(
            "Date64 should become an ISO datetime string, got {:?}",
            other
        ),
    }
    match df.column("ts_naive").unwrap().column() {
        Column::String(c) => {
            assert_eq!(c.get(0).unwrap(), Some("1970-01-02T01:01:01"));
            assert_eq!(c.get(1).unwrap(), None);
        }
        other => panic!(
            "a timezone-naive Timestamp should become a bare ISO datetime string, got {:?}",
            other
        ),
    }
    match df.column("ts_utc").unwrap().column() {
        Column::String(c) => {
            assert_eq!(
                c.get(0).unwrap(),
                Some("1970-01-02T01:01:01Z"),
                "a timezone-aware Timestamp must render with a Z suffix"
            );
            assert_eq!(c.get(1).unwrap(), None);
        }
        other => panic!(
            "Timestamp should become an ISO datetime string, got {:?}",
            other
        ),
    }
    match df.column("large_s").unwrap().column() {
        Column::String(c) => {
            assert_eq!(c.get(0).unwrap(), Some("hello"));
            assert_eq!(c.get(1).unwrap(), None);
        }
        other => panic!("LargeUtf8 should downcast correctly, got {:?}", other),
    }
}

#[cfg(feature = "parquet")]
#[test]
fn parquet_genuinely_unsupported_arrow_type_errors_instead_of_fabricating_data() {
    use arrow::array::{ArrayRef, BinaryArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_writer::ArrowWriter;
    use std::fs::File;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![Field::new("bin", DataType::Binary, true)]));
    let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(&b"abc"[..]), None]));
    let batch = RecordBatch::try_new(schema.clone(), vec![arr]).unwrap();

    let path = unique_temp_path("unsupported_type.parquet");
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let result = OptimizedDataFrame::from_parquet(&path);
    let _ = std::fs::remove_file(&path);

    assert!(
        result.is_err(),
        "a genuinely unsupported Arrow type must error, not fabricate a Debug-dump string"
    );
}
