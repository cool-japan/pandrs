//! String pattern analysis driving adaptive storage/compression selection.
//!
//! Two of the detectors here used to panic on any non-ASCII input:
//! `PrefixSuffixDetector` byte-sliced `&s[..len]` for `len` in `1..=5`, and the
//! numeric detector did `&s[1..]` after `starts_with('€')`. Both are reachable
//! from `create_storage`/`write_chunk`, so a single Japanese or euro-prefixed
//! value aborted the process. All slicing is now character-aware.

use crate::core::error::Result;
use crate::storage::adaptive_string_pool::codec::StringCompressionAlgorithm;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// String pool configuration
#[derive(Debug, Clone)]
pub struct StringPoolConfig {
    /// Maximum pool size in bytes
    pub max_pool_size: usize,
    /// Enable pattern analysis
    pub enable_pattern_analysis: bool,
    /// Enable adaptive compression
    pub enable_adaptive_compression: bool,
    /// Deduplication threshold (minimum string length to deduplicate)
    pub deduplication_threshold: usize,
    /// Analysis window size for pattern detection
    pub analysis_window_size: usize,
    /// Compression threshold (minimum savings to enable compression)
    pub compression_threshold: f64,
    /// Enable dictionary encoding
    pub enable_dictionary_encoding: bool,
    /// Maximum dictionary size
    pub max_dictionary_size: usize,
}

impl Default for StringPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 100 * 1024 * 1024,
            enable_pattern_analysis: true,
            enable_adaptive_compression: true,
            deduplication_threshold: 4,
            analysis_window_size: 10000,
            compression_threshold: 0.1,
            enable_dictionary_encoding: true,
            max_dictionary_size: 1024 * 1024,
        }
    }
}

/// String characteristics for optimization
#[derive(Debug, Clone, Default)]
pub struct StringCharacteristics {
    /// Average string length in bytes
    pub avg_length: f64,
    /// String length distribution
    pub length_distribution: HashMap<usize, u32>,
    /// Character frequency distribution
    pub char_frequency: HashMap<char, u32>,
    /// Common prefixes and suffixes
    pub common_patterns: PatternAnalysis,
    /// Duplication ratio
    pub duplication_ratio: f64,
    /// Compression potential estimate (0.0 to 1.0)
    pub compression_potential: f64,
    /// Shannon entropy in **bits per byte** (0.0 to 8.0)
    pub entropy: f64,
}

impl StringCharacteristics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Pattern analysis for string optimization
#[derive(Debug, Clone, Default)]
pub struct PatternAnalysis {
    /// Common prefixes with their frequencies
    pub common_prefixes: HashMap<String, u32>,
    /// Common suffixes with their frequencies
    pub common_suffixes: HashMap<String, u32>,
    /// Common substrings
    pub common_substrings: HashMap<String, u32>,
    /// Numeric pattern detection
    pub numeric_patterns: NumericPatterns,
    /// Date/time pattern detection
    pub datetime_patterns: DateTimePatterns,
    /// URL/email pattern detection
    pub structured_patterns: StructuredPatterns,
}

impl PatternAnalysis {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Numeric pattern analysis
#[derive(Debug, Clone, Default)]
pub struct NumericPatterns {
    pub integer_frequency: u32,
    pub float_frequency: u32,
    pub scientific_frequency: u32,
    pub currency_frequency: u32,
}

impl NumericPatterns {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Date/time pattern analysis
#[derive(Debug, Clone, Default)]
pub struct DateTimePatterns {
    pub iso_date_frequency: u32,
    pub us_date_frequency: u32,
    pub eu_date_frequency: u32,
    pub timestamp_frequency: u32,
}

impl DateTimePatterns {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Structured pattern analysis
#[derive(Debug, Clone, Default)]
pub struct StructuredPatterns {
    pub url_frequency: u32,
    pub email_frequency: u32,
    pub uuid_frequency: u32,
    pub json_frequency: u32,
}

impl StructuredPatterns {
    pub fn new() -> Self {
        Self::default()
    }
}

/// String storage strategy enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringStorageStrategy {
    /// Raw string storage
    Raw,
    /// Deduplicated storage with reference counting
    Deduplicated,
    /// Dictionary encoded storage
    DictionaryEncoded,
    /// Compressed storage
    Compressed,
    /// Hybrid strategy combining multiple approaches
    Hybrid,
}

/// Strategy recommendations based on analysis
#[derive(Debug, Clone)]
pub struct StrategyRecommendations {
    /// Recommended storage strategy
    pub recommended_strategy: StringStorageStrategy,
    /// Recommended compression algorithm
    pub recommended_compression: StringCompressionAlgorithm,
    /// Confidence in the recommendation (0.0 to 1.0)
    pub confidence: f64,
    /// Expected space savings
    pub expected_savings: f64,
}

impl Default for StrategyRecommendations {
    fn default() -> Self {
        Self {
            recommended_strategy: StringStorageStrategy::Raw,
            recommended_compression: StringCompressionAlgorithm::None,
            confidence: 0.5,
            expected_savings: 0.0,
        }
    }
}

/// String pattern analyzer
pub struct StringPatternAnalyzer {
    config: StringPoolConfig,
    pattern_engines: Vec<Box<dyn PatternDetector>>,
    analysis_cache: Arc<Mutex<HashMap<u64, StringCharacteristics>>>,
}

impl StringPatternAnalyzer {
    pub fn new(config: StringPoolConfig) -> Self {
        let pattern_engines: Vec<Box<dyn PatternDetector>> = vec![
            Box::new(LengthPatternDetector::new()),
            Box::new(CharacterFrequencyDetector::new()),
            Box::new(PrefixSuffixDetector::new()),
            Box::new(NumericPatternDetector::new()),
            Box::new(DateTimePatternDetector::new()),
            Box::new(StructuredPatternDetector::new()),
        ];

        Self {
            config,
            pattern_engines,
            analysis_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Configuration this analyzer was built with.
    pub fn config(&self) -> &StringPoolConfig {
        &self.config
    }

    /// Number of cached analyses.
    pub fn cached_analyses(&self) -> usize {
        self.analysis_cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    pub fn analyze_strings(&self, strings: &[String]) -> Result<StringCharacteristics> {
        let mut characteristics = StringCharacteristics::new();

        for detector in &self.pattern_engines {
            detector.analyze(strings, &mut characteristics)?;
        }

        self.calculate_derived_metrics(&mut characteristics, strings)?;

        if let Ok(mut cache) = self.analysis_cache.lock() {
            // Keep the cache bounded; it is a diagnostic aid, not storage.
            if cache.len() < self.config.analysis_window_size {
                cache.insert(sample_fingerprint(strings), characteristics.clone());
            }
        }

        Ok(characteristics)
    }

    fn calculate_derived_metrics(
        &self,
        characteristics: &mut StringCharacteristics,
        strings: &[String],
    ) -> Result<()> {
        if strings.is_empty() {
            return Ok(());
        }

        let total_length: usize = strings.iter().map(|s| s.len()).sum();
        characteristics.avg_length = total_length as f64 / strings.len() as f64;

        let unique_strings: std::collections::HashSet<&String> = strings.iter().collect();
        characteristics.duplication_ratio =
            1.0 - (unique_strings.len() as f64 / strings.len() as f64);

        characteristics.entropy = calculate_entropy(strings);
        // Entropy is bits per byte, so 8.0 is the maximum; dividing by 8 now
        // really normalises to 0..1. It used to be computed over Unicode scalar
        // values, where CJK text exceeds 8 bits/symbol and every such column was
        // scored as having zero compression potential.
        characteristics.compression_potential =
            1.0 - (characteristics.entropy / 8.0).clamp(0.0, 1.0);

        Ok(())
    }

    pub fn recommend_strategy(
        &self,
        characteristics: &StringCharacteristics,
    ) -> StrategyRecommendations {
        let confidence;

        let recommended_strategy = if characteristics.duplication_ratio > 0.3 {
            confidence = 0.9;
            StringStorageStrategy::Deduplicated
        } else if characteristics.avg_length < 10.0 && characteristics.compression_potential > 0.5 {
            confidence = 0.85;
            StringStorageStrategy::DictionaryEncoded
        } else if characteristics.compression_potential > 0.4 {
            confidence = 0.8;
            StringStorageStrategy::Compressed
        } else if characteristics.avg_length > 100.0 || characteristics.duplication_ratio > 0.1 {
            confidence = 0.75;
            StringStorageStrategy::Hybrid
        } else {
            confidence = 0.6;
            StringStorageStrategy::Raw
        };

        let mut recommended_compression = if characteristics
            .common_patterns
            .numeric_patterns
            .integer_frequency
            > 50
        {
            StringCompressionAlgorithm::Dictionary
        } else if characteristics.entropy < 4.0 {
            StringCompressionAlgorithm::RunLength
        } else if characteristics.avg_length > 50.0 {
            StringCompressionAlgorithm::Zstd
        } else {
            StringCompressionAlgorithm::Lz4
        };

        if !self.config.enable_adaptive_compression {
            recommended_compression = StringCompressionAlgorithm::None;
        } else if recommended_compression == StringCompressionAlgorithm::Dictionary
            && !self.config.enable_dictionary_encoding
        {
            recommended_compression = StringCompressionAlgorithm::Zstd;
        }

        let expected_savings = match recommended_strategy {
            StringStorageStrategy::Deduplicated => characteristics.duplication_ratio * 0.8,
            StringStorageStrategy::DictionaryEncoded => characteristics.compression_potential * 0.6,
            StringStorageStrategy::Compressed => characteristics.compression_potential * 0.7,
            StringStorageStrategy::Hybrid => {
                characteristics.compression_potential * 0.8
                    + characteristics.duplication_ratio * 0.5
            }
            _ => 0.0,
        };

        StrategyRecommendations {
            recommended_strategy,
            recommended_compression,
            confidence,
            expected_savings,
        }
    }
}

/// Shannon entropy of the UTF-8 **bytes**, in bits per byte (0.0 to 8.0).
fn calculate_entropy(strings: &[String]) -> f64 {
    let mut byte_counts = [0u64; 256];
    let mut total_bytes = 0u64;

    for s in strings {
        for &b in s.as_bytes() {
            byte_counts[b as usize] += 1;
            total_bytes += 1;
        }
    }

    if total_bytes == 0 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &count in byte_counts.iter() {
        if count == 0 {
            continue;
        }
        let probability = count as f64 / total_bytes as f64;
        entropy -= probability * probability.log2();
    }
    entropy
}

fn sample_fingerprint(strings: &[String]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    strings.len().hash(&mut hasher);
    for s in strings.iter().take(16) {
        s.hash(&mut hasher);
    }
    hasher.finish()
}

/// Pattern detector trait
pub trait PatternDetector: Send + Sync {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()>;
}

/// Length pattern detector
pub struct LengthPatternDetector;

impl LengthPatternDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LengthPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector for LengthPatternDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            *characteristics
                .length_distribution
                .entry(s.len())
                .or_insert(0) += 1;
        }
        Ok(())
    }
}

/// Character frequency detector
pub struct CharacterFrequencyDetector;

impl CharacterFrequencyDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CharacterFrequencyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector for CharacterFrequencyDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            for c in s.chars() {
                *characteristics.char_frequency.entry(c).or_insert(0) += 1;
            }
        }
        Ok(())
    }
}

/// Prefix/suffix pattern detector
pub struct PrefixSuffixDetector;

impl PrefixSuffixDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PrefixSuffixDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Longest prefix of `s` containing at most `chars` characters.
fn char_prefix(s: &str, chars: usize) -> &str {
    match s.char_indices().nth(chars) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

/// Longest suffix of `s` containing at most `chars` characters.
fn char_suffix(s: &str, chars: usize) -> &str {
    let total = s.chars().count();
    if chars >= total {
        return s;
    }
    match s.char_indices().nth(total - chars) {
        Some((idx, _)) => &s[idx..],
        None => s,
    }
}

impl PatternDetector for PrefixSuffixDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            let char_count = s.chars().count();
            let limit = char_count.min(5);
            for len in 1..=limit {
                // Character-indexed, so a multi-byte value can never be sliced
                // through the middle of a code point.
                let prefix = char_prefix(s, len);
                *characteristics
                    .common_patterns
                    .common_prefixes
                    .entry(prefix.to_string())
                    .or_insert(0) += 1;

                let suffix = char_suffix(s, len);
                *characteristics
                    .common_patterns
                    .common_suffixes
                    .entry(suffix.to_string())
                    .or_insert(0) += 1;
            }
        }
        Ok(())
    }
}

/// Numeric pattern detector
pub struct NumericPatternDetector;

impl NumericPatternDetector {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NumericPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector for NumericPatternDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            let numeric = &mut characteristics.common_patterns.numeric_patterns;
            if s.parse::<i64>().is_ok() {
                numeric.integer_frequency += 1;
            } else if s.parse::<f64>().is_ok() {
                numeric.float_frequency += 1;
            } else if s.contains('e') || s.contains('E') {
                if s.replace(['e', 'E', '+', '-'], "").parse::<f64>().is_ok() {
                    numeric.scientific_frequency += 1;
                }
            } else if let Some(rest) = s
                .strip_prefix('$')
                .or_else(|| s.strip_prefix('€'))
                .or_else(|| s.strip_prefix('£'))
            {
                // `strip_prefix` returns the remainder at a char boundary; the
                // old `&s[1..]` panicked for the multi-byte currency symbols.
                if rest.replace(',', "").parse::<f64>().is_ok() {
                    numeric.currency_frequency += 1;
                }
            }
        }
        Ok(())
    }
}

/// Date/time pattern detector
pub struct DateTimePatternDetector;

impl DateTimePatternDetector {
    pub fn new() -> Self {
        Self
    }

    fn matches_iso_date(&self, s: &str) -> bool {
        // ASCII-only shape check first, so the byte slices below are safe.
        let bytes = s.as_bytes();
        bytes.len() == 10
            && s.is_ascii()
            && bytes[4] == b'-'
            && bytes[7] == b'-'
            && s[..4].parse::<u16>().is_ok()
            && s[5..7].parse::<u8>().is_ok()
            && s[8..10].parse::<u8>().is_ok()
    }

    fn matches_slash_date(&self, s: &str) -> bool {
        let parts: Vec<&str> = s.split('/').collect();
        parts.len() == 3
            && parts[0].parse::<u8>().is_ok()
            && parts[1].parse::<u8>().is_ok()
            && parts[2].parse::<u16>().is_ok()
    }
}

impl Default for DateTimePatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector for DateTimePatternDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            let datetime = &mut characteristics.common_patterns.datetime_patterns;
            if self.matches_iso_date(s) {
                datetime.iso_date_frequency += 1;
            } else if self.matches_slash_date(s) {
                // `MM/DD/YYYY` and `DD/MM/YYYY` are indistinguishable without a
                // locale, so both counters are incremented rather than pretending
                // the first branch identified a US date.
                datetime.us_date_frequency += 1;
                datetime.eu_date_frequency += 1;
            } else if s.len() == 10 && s.parse::<u64>().is_ok() {
                datetime.timestamp_frequency += 1;
            }
        }
        Ok(())
    }
}

/// Structured pattern detector (URLs, emails, etc.)
pub struct StructuredPatternDetector;

impl StructuredPatternDetector {
    pub fn new() -> Self {
        Self
    }

    fn matches_uuid(&self, s: &str) -> bool {
        let bytes = s.as_bytes();
        bytes.len() == 36
            && s.is_ascii()
            && bytes[8] == b'-'
            && bytes[13] == b'-'
            && bytes[18] == b'-'
            && bytes[23] == b'-'
            && bytes
                .iter()
                .enumerate()
                .all(|(i, b)| matches!(i, 8 | 13 | 18 | 23) || b.is_ascii_hexdigit())
    }
}

impl Default for StructuredPatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector for StructuredPatternDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            let structured = &mut characteristics.common_patterns.structured_patterns;
            if s.starts_with("http://") || s.starts_with("https://") {
                structured.url_frequency += 1;
            } else if s.contains('@') && s.contains('.') {
                structured.email_frequency += 1;
            } else if self.matches_uuid(s) {
                structured.uuid_frequency += 1;
            } else if (s.starts_with('{') && s.ends_with('}'))
                || (s.starts_with('[') && s.ends_with(']'))
            {
                structured.json_frequency += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_ascii_strings_do_not_panic() {
        // Each of these panicked in the old detectors.
        let strings = vec![
            "日本語".to_string(),
            "€100".to_string(),
            "£42.5".to_string(),
            "🐟🍣".to_string(),
            "aé".to_string(),
            "ß".to_string(),
        ];
        let analyzer = StringPatternAnalyzer::new(StringPoolConfig::default());
        let characteristics = analyzer.analyze_strings(&strings).expect("analyze");
        assert!(!characteristics.common_patterns.common_prefixes.is_empty());
        assert_eq!(
            characteristics
                .common_patterns
                .numeric_patterns
                .currency_frequency,
            2,
            "euro and pound amounts should be detected as currency"
        );
    }

    #[test]
    fn prefix_and_suffix_are_character_aligned() {
        let strings = vec!["日本語テキスト".to_string()];
        let analyzer = StringPatternAnalyzer::new(StringPoolConfig::default());
        let characteristics = analyzer.analyze_strings(&strings).expect("analyze");
        assert!(characteristics
            .common_patterns
            .common_prefixes
            .contains_key("日"));
        assert!(characteristics
            .common_patterns
            .common_suffixes
            .contains_key("ト"));
    }

    #[test]
    fn entropy_is_bits_per_byte() {
        // Uniformly random bytes approach 8 bits/byte; a constant string is 0.
        let constant = vec!["aaaaaaaaaaaaaaaaaaaa".to_string()];
        assert!(calculate_entropy(&constant) < 0.001);

        let cjk = vec!["日本語のテキストです、これは長い文章です。".to_string()];
        let entropy = calculate_entropy(&cjk);
        assert!(
            (0.0..=8.0).contains(&entropy),
            "entropy {} is outside the bits-per-byte range",
            entropy
        );
    }

    #[test]
    fn cjk_columns_are_not_forced_to_raw() {
        // Byte-entropy normalisation keeps compression_potential > 0 for CJK
        // text; the old scalar-based entropy exceeded 8 and zeroed it out.
        let strings: Vec<String> = (0..64)
            .map(|i| format!("日本語のテキスト {}", i % 8))
            .collect();
        let analyzer = StringPatternAnalyzer::new(StringPoolConfig::default());
        let characteristics = analyzer.analyze_strings(&strings).expect("analyze");
        assert!(
            characteristics.compression_potential > 0.0,
            "CJK column scored zero compression potential"
        );
    }

    #[test]
    fn numeric_and_pattern_detection_still_works() {
        let strings = vec![
            "123".to_string(),
            "45.67".to_string(),
            "not_a_number".to_string(),
            "2024-01-31".to_string(),
            "https://example.com".to_string(),
            "user@example.com".to_string(),
        ];
        let analyzer = StringPatternAnalyzer::new(StringPoolConfig::default());
        let c = analyzer.analyze_strings(&strings).expect("analyze");
        assert_eq!(c.common_patterns.numeric_patterns.integer_frequency, 1);
        assert_eq!(c.common_patterns.numeric_patterns.float_frequency, 1);
        assert_eq!(c.common_patterns.datetime_patterns.iso_date_frequency, 1);
        assert_eq!(c.common_patterns.structured_patterns.url_frequency, 1);
        assert_eq!(c.common_patterns.structured_patterns.email_frequency, 1);
    }

    #[test]
    fn adaptive_compression_flag_is_honoured() {
        let mut config = StringPoolConfig::default();
        config.enable_adaptive_compression = false;
        let analyzer = StringPatternAnalyzer::new(config);
        let characteristics = analyzer
            .analyze_strings(&["some text".to_string()])
            .expect("analyze");
        let recommendation = analyzer.recommend_strategy(&characteristics);
        assert_eq!(
            recommendation.recommended_compression,
            StringCompressionAlgorithm::None
        );
    }
}
