//! Full polynomial-basis helpers shared by [`super::AutoFeatureEngineer`]'s
//! `generate_polynomial_features` and by
//! [`pipeline_extended`](crate::ml::pipeline_extended)'s
//! `FeatureEngineeringStage::create_polynomial_features`, so both modules'
//! polynomial-feature generators produce the identical basis and naming
//! convention.

/// All non-decreasing index sequences of length `k` drawn from `0..n`
/// (combinations of `0..n` with replacement) -- the index multisets of
/// every degree-`k` monomial in a full polynomial expansion of `n`
/// variables. `k=0` yields a single empty combination; `n=0` (with `k>0`)
/// yields none.
pub(crate) fn combinations_with_replacement(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if n == 0 {
        return Vec::new();
    }

    fn recurse(
        start: usize,
        n: usize,
        k: usize,
        pos: usize,
        combo: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if pos == k {
            result.push(combo.clone());
            return;
        }
        for i in start..n {
            combo[pos] = i;
            recurse(i, n, k, pos + 1, combo, result);
        }
    }

    let mut result = Vec::new();
    let mut combo = vec![0usize; k];
    recurse(0, n, k, 0, &mut combo, &mut result);
    result
}

/// Render a monomial's name from its (non-decreasing) index combination
/// over `names`, e.g. `[0,0,1]` over `["a","b"]` -> `"a^2*b"`. A run of a
/// repeated index collapses to a single `name^count` term, matching the
/// pre-existing naming convention for pure powers (`x^2`) and pairwise
/// cross terms (`x*y`).
pub(crate) fn monomial_name(combo: &[usize], names: &[String]) -> String {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < combo.len() {
        let idx = combo[i];
        let mut count = 1;
        while i + count < combo.len() && combo[i + count] == idx {
            count += 1;
        }
        if count > 1 {
            parts.push(format!("{}^{}", names[idx], count));
        } else {
            parts.push(names[idx].clone());
        }
        i += count;
    }
    parts.join("*")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_with_replacement_counts_match_multiset_coefficient() {
        // C(n+k-1, k): degree-2 monomials over 3 variables = C(4,2) = 6;
        // degree-3 monomials over 3 variables = C(5,3) = 10.
        assert_eq!(combinations_with_replacement(3, 2).len(), 6);
        assert_eq!(combinations_with_replacement(3, 3).len(), 10);
    }

    #[test]
    fn monomial_name_collapses_repeated_indices() {
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(monomial_name(&[0, 0], &names), "a^2");
        assert_eq!(monomial_name(&[0, 1], &names), "a*b");
        assert_eq!(monomial_name(&[0, 0, 1], &names), "a^2*b");
        assert_eq!(monomial_name(&[0, 1, 2], &names), "a*b*c");
    }
}
