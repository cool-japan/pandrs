//! Wave 4 regression tests for the clippy-narrowing task.
//!
//! The one behaviour-affecting change made while removing the crate-wide
//! `#![allow(clippy::all)]` was fixing `OrderedFloat` so its `PartialOrd` is
//! consistent with its manual total `Ord`. Previously `PartialOrd` was derived
//! (returning `None` for NaN) while `Ord` returned `Equal`, which violated the
//! `Ord`/`PartialOrd` agreement contract (clippy::derive_ord_xor_partial_ord).
//! These tests lock in the corrected, consistent behaviour and prove the map /
//! sort ordering real code relies on is unchanged.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use pandrs::core::advanced_multi_index::OrderedFloat;
use pandrs::core::IndexValue;

/// `partial_cmp(a, b)` must equal `Some(cmp(a, b))` for every pair — exactly the
/// invariant `derive_ord_xor_partial_ord` protects. Covers normal values, the
/// two zeroes, the infinities and NaN.
#[test]
fn ordered_float_partial_cmp_agrees_with_cmp() {
    let samples = [
        0.0_f64,
        -0.0,
        1.0,
        -1.0,
        f64::MIN,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
    ];
    for &a in &samples {
        for &b in &samples {
            let oa = OrderedFloat(a);
            let ob = OrderedFloat(b);
            assert_eq!(
                oa.partial_cmp(&ob),
                Some(oa.cmp(&ob)),
                "partial_cmp/cmp disagree for ({a}, {b})"
            );
        }
    }
}

/// The corrected `PartialOrd` returns `Some(Equal)` for NaN (matching the total
/// `Ord`), not the `None` the derived impl used to produce. This is the concrete
/// behaviour change the fix introduces.
#[test]
fn ordered_float_nan_is_total() {
    let nan = OrderedFloat(f64::NAN);
    assert_eq!(nan.partial_cmp(&nan), Some(Ordering::Equal));
    assert_eq!(nan.cmp(&nan), Ordering::Equal);
    // NaN vs a finite value is also defined (Equal), never `None`.
    assert_eq!(nan.partial_cmp(&OrderedFloat(1.0)), Some(Ordering::Equal));
}

/// `Ord` ordering of finite values is unchanged — the fix must not perturb the
/// map/sort behaviour real code relies on.
#[test]
fn ordered_float_ord_unchanged_for_finite() {
    assert_eq!(OrderedFloat(1.0).cmp(&OrderedFloat(2.0)), Ordering::Less);
    assert_eq!(OrderedFloat(2.0).cmp(&OrderedFloat(2.0)), Ordering::Equal);
    assert_eq!(OrderedFloat(3.0).cmp(&OrderedFloat(2.0)), Ordering::Greater);

    let mut v = [OrderedFloat(3.0), OrderedFloat(1.0), OrderedFloat(2.0)];
    v.sort();
    assert_eq!(v, [OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(3.0)]);
}

/// `IndexValue::Float` is used as a `BTreeMap` key, which orders via `Ord`; the
/// fix keeps that ordering intact so multi-index cross-section lookups are
/// unaffected.
#[test]
fn index_value_float_btreemap_roundtrip() {
    let mut map: BTreeMap<IndexValue, usize> = BTreeMap::new();
    map.insert(IndexValue::Float(OrderedFloat(2.5)), 2);
    map.insert(IndexValue::Float(OrderedFloat(1.5)), 1);
    map.insert(IndexValue::Float(OrderedFloat(3.5)), 3);

    // BTreeMap iterates in `Ord` order.
    let ordered: Vec<usize> = map.values().copied().collect();
    assert_eq!(ordered, vec![1, 2, 3]);

    assert_eq!(map.get(&IndexValue::Float(OrderedFloat(2.5))), Some(&2));
}

/// Documents *why* `#![allow(clippy::result_large_err)]` exists in `src/lib.rs`:
/// the public `Error` enum exceeds clippy's 128-byte `result_large_err`
/// threshold (dominated by the `Enhanced` variant's inline `ErrorContext`).
/// Shrinking it is a public-API change deferred to 0.5.0. If a future change
/// drops it below the threshold this test fails — a signal to remove the
/// crate-wide allow. Gated to 64-bit targets, where the threshold applies.
#[test]
#[cfg(target_pointer_width = "64")]
fn error_enum_exceeds_large_err_threshold() {
    assert!(
        std::mem::size_of::<pandrs::core::Error>() > 128,
        "Error shrank below clippy's result_large_err threshold ({} bytes) — \
         reconsider the crate-wide allow in src/lib.rs",
        std::mem::size_of::<pandrs::core::Error>()
    );
}
