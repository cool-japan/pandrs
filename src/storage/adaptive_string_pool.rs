//! Adaptive String Pool Strategy Implementation
//!
//! This module provides the AdaptiveStringPoolStrategy as specified in the
//! memory management unification strategy document, with intelligent string
//! pattern analysis and adaptive optimization.

use crate::core::error::{Error, Result};
use crate::storage::unified_memory::*;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

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
            max_pool_size: 100 * 1024 * 1024, // 100MB
            enable_pattern_analysis: true,
            enable_adaptive_compression: true,
            deduplication_threshold: 4, // Deduplicate strings >= 4 characters
            analysis_window_size: 10000, // Analyze last 10k strings
            compression_threshold: 0.1, // 10% minimum savings
            enable_dictionary_encoding: true,
            max_dictionary_size: 1024 * 1024, // 1MB dictionary
        }
    }
}

/// String characteristics for optimization
#[derive(Debug, Clone)]
pub struct StringCharacteristics {
    /// Average string length
    pub avg_length: f64,
    /// String length distribution
    pub length_distribution: HashMap<usize, u32>,
    /// Character frequency distribution
    pub char_frequency: HashMap<char, u32>,
    /// Common prefixes and suffixes
    pub common_patterns: PatternAnalysis,
    /// Duplication ratio
    pub duplication_ratio: f64,
    /// Compression potential estimate
    pub compression_potential: f64,
    /// Entropy measure
    pub entropy: f64,
}

impl StringCharacteristics {
    pub fn new() -> Self {
        Self {
            avg_length: 0.0,
            length_distribution: HashMap::new(),
            char_frequency: HashMap::new(),
            common_patterns: PatternAnalysis::new(),
            duplication_ratio: 0.0,
            compression_potential: 0.0,
            entropy: 0.0,
        }
    }
}

/// Pattern analysis for string optimization
#[derive(Debug, Clone)]
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
        Self {
            common_prefixes: HashMap::new(),
            common_suffixes: HashMap::new(),
            common_substrings: HashMap::new(),
            numeric_patterns: NumericPatterns::new(),
            datetime_patterns: DateTimePatterns::new(),
            structured_patterns: StructuredPatterns::new(),
        }
    }
}

/// Numeric pattern analysis
#[derive(Debug, Clone)]
pub struct NumericPatterns {
    /// Integer pattern frequency
    pub integer_frequency: u32,
    /// Float pattern frequency
    pub float_frequency: u32,
    /// Scientific notation frequency
    pub scientific_frequency: u32,
    /// Currency pattern frequency
    pub currency_frequency: u32,
}

impl NumericPatterns {
    pub fn new() -> Self {
        Self {
            integer_frequency: 0,
            float_frequency: 0,
            scientific_frequency: 0,
            currency_frequency: 0,
        }
    }
}

/// Date/time pattern analysis
#[derive(Debug, Clone)]
pub struct DateTimePatterns {
    /// ISO date format frequency
    pub iso_date_frequency: u32,
    /// US date format frequency
    pub us_date_frequency: u32,
    /// European date format frequency
    pub eu_date_frequency: u32,
    /// Timestamp frequency
    pub timestamp_frequency: u32,
}

impl DateTimePatterns {
    pub fn new() -> Self {
        Self {
            iso_date_frequency: 0,
            us_date_frequency: 0,
            eu_date_frequency: 0,
            timestamp_frequency: 0,
        }
    }
}

/// Structured pattern analysis
#[derive(Debug, Clone)]
pub struct StructuredPatterns {
    /// URL pattern frequency
    pub url_frequency: u32,
    /// Email pattern frequency
    pub email_frequency: u32,
    /// UUID pattern frequency
    pub uuid_frequency: u32,
    /// JSON pattern frequency
    pub json_frequency: u32,
}

impl StructuredPatterns {
    pub fn new() -> Self {
        Self {
            url_frequency: 0,
            email_frequency: 0,
            uuid_frequency: 0,
            json_frequency: 0,
        }
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

/// String compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StringCompressionAlgorithm {
    /// No compression
    None,
    /// Run-length encoding for repetitive strings
    RunLength,
    /// Dictionary compression
    Dictionary,
    /// LZ4 compression
    Lz4,
    /// ZSTD compression
    Zstd,
    /// Custom string-optimized compression
    StringOptimized,
}

/// String entry in the pool
#[derive(Debug, Clone)]
pub struct StringEntry {
    /// Unique identifier
    pub id: StringId,
    /// Storage strategy used
    pub strategy: StringStorageStrategy,
    /// Compressed/encoded data
    pub data: Vec<u8>,
    /// Original string length
    pub original_length: usize,
    /// Reference count for deduplication
    pub ref_count: u32,
    /// First access timestamp
    pub first_accessed: Instant,
    /// Last access timestamp
    pub last_accessed: Instant,
    /// Access frequency
    pub access_count: u64,
    /// Compression algorithm used
    pub compression: StringCompressionAlgorithm,
    /// Metadata for reconstruction
    pub metadata: StringMetadata,
}

/// String identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StringId(pub u64);

/// String metadata for reconstruction
#[derive(Debug, Clone)]
pub struct StringMetadata {
    /// Encoding type
    pub encoding: StringEncoding,
    /// Dictionary references if applicable
    pub dictionary_refs: Vec<u32>,
    /// Additional reconstruction data
    pub reconstruction_data: HashMap<String, String>,
}

/// String encoding type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringEncoding {
    /// UTF-8 encoding
    Utf8,
    /// ASCII encoding
    Ascii,
    /// Latin-1 encoding
    Latin1,
    /// Custom encoding based on pattern analysis
    Custom,
}

/// String pool handle
#[derive(Debug)]
pub struct StringPoolHandle {
    /// Pool configuration
    pub config: StringPoolConfig,
    /// Current storage strategy
    pub current_strategy: StringStorageStrategy,
    /// Active string entries
    pub string_count: usize,
    /// Pool statistics
    pub statistics: StringPoolStatistics,
    /// Pattern analyzer state
    pub analyzer_state: PatternAnalyzerState,
}

/// String pool statistics
#[derive(Debug, Clone)]
pub struct StringPoolStatistics {
    /// Total strings stored
    pub total_strings: u64,
    /// Unique strings (after deduplication)
    pub unique_strings: u64,
    /// Total storage space used
    pub storage_used: usize,
    /// Total original size
    pub original_size: usize,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Deduplication savings
    pub deduplication_savings: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average access time
    pub avg_access_time: Duration,
}

impl StringPoolStatistics {
    pub fn new() -> Self {
        Self {
            total_strings: 0,
            unique_strings: 0,
            storage_used: 0,
            original_size: 0,
            compression_ratio: 1.0,
            deduplication_savings: 0.0,
            cache_hit_rate: 0.0,
            avg_access_time: Duration::ZERO,
        }
    }

    pub fn space_savings(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            1.0 - (self.storage_used as f64 / self.original_size as f64)
        }
    }
}

/// Pattern analyzer state
#[derive(Debug)]
pub struct PatternAnalyzerState {
    /// Recent strings for analysis
    pub recent_strings: VecDeque<String>,
    /// Current characteristics
    pub characteristics: StringCharacteristics,
    /// Analysis window
    pub analysis_window: usize,
    /// Last analysis timestamp
    pub last_analysis: Instant,
    /// Strategy recommendations
    pub strategy_recommendations: StrategyRecommendations,
}

/// Strategy recommendations based on analysis
#[derive(Debug, Clone)]
pub struct StrategyRecommendations {
    /// Recommended storage strategy
    pub recommended_strategy: StringStorageStrategy,
    /// Recommended compression algorithm
    pub recommended_compression: StringCompressionAlgorithm,
    /// Confidence in recommendation (0.0 to 1.0)
    pub confidence: f64,
    /// Expected space savings
    pub expected_savings: f64,
}

/// String pattern analyzer
pub struct StringPatternAnalyzer {
    /// Configuration
    config: StringPoolConfig,
    /// Pattern detection engines
    pattern_engines: Vec<Box<dyn PatternDetector>>,
    /// Analysis cache
    analysis_cache: Arc<Mutex<HashMap<u64, StringCharacteristics>>>,
}

impl StringPatternAnalyzer {
    pub fn new(config: StringPoolConfig) -> Self {
        let mut pattern_engines: Vec<Box<dyn PatternDetector>> = Vec::new();
        pattern_engines.push(Box::new(LengthPatternDetector::new()));
        pattern_engines.push(Box::new(CharacterFrequencyDetector::new()));
        pattern_engines.push(Box::new(PrefixSuffixDetector::new()));
        pattern_engines.push(Box::new(NumericPatternDetector::new()));
        pattern_engines.push(Box::new(DateTimePatternDetector::new()));
        pattern_engines.push(Box::new(StructuredPatternDetector::new()));

        Self {
            config,
            pattern_engines,
            analysis_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn analyze_strings(&self, strings: &[String]) -> Result<StringCharacteristics> {
        let mut characteristics = StringCharacteristics::new();

        // Run all pattern detectors
        for detector in &self.pattern_engines {
            detector.analyze(strings, &mut characteristics)?;
        }

        // Calculate derived metrics
        self.calculate_derived_metrics(&mut characteristics, strings)?;

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

        // Calculate average length
        let total_length: usize = strings.iter().map(|s| s.len()).sum();
        characteristics.avg_length = total_length as f64 / strings.len() as f64;

        // Calculate duplication ratio
        let mut unique_strings = std::collections::HashSet::new();
        for s in strings {
            unique_strings.insert(s.clone());
        }
        characteristics.duplication_ratio =
            1.0 - (unique_strings.len() as f64 / strings.len() as f64);

        // Estimate compression potential based on entropy
        characteristics.entropy = self.calculate_entropy(strings);
        characteristics.compression_potential = 1.0 - (characteristics.entropy / 8.0).min(1.0);

        Ok(())
    }

    fn calculate_entropy(&self, strings: &[String]) -> f64 {
        let mut char_counts = HashMap::new();
        let mut total_chars = 0;

        for s in strings {
            for c in s.chars() {
                *char_counts.entry(c).or_insert(0) += 1;
                total_chars += 1;
            }
        }

        if total_chars == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in char_counts.values() {
            let probability = count as f64 / total_chars as f64;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }

        entropy
    }

    pub fn recommend_strategy(
        &self,
        characteristics: &StringCharacteristics,
    ) -> StrategyRecommendations {
        let confidence;

        // Strategy selection logic
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

        // Compression algorithm selection
        let recommended_compression = if characteristics
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

impl PatternDetector for LengthPatternDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            let len = s.len();
            *characteristics.length_distribution.entry(len).or_insert(0) += 1;
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

impl PatternDetector for PrefixSuffixDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            // Analyze prefixes (up to 5 characters)
            for len in 1..=5.min(s.len()) {
                let prefix = &s[..len];
                *characteristics
                    .common_patterns
                    .common_prefixes
                    .entry(prefix.to_string())
                    .or_insert(0) += 1;
            }

            // Analyze suffixes (up to 5 characters)
            for len in 1..=5.min(s.len()) {
                let suffix = &s[s.len() - len..];
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

impl PatternDetector for NumericPatternDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            if s.parse::<i64>().is_ok() {
                characteristics
                    .common_patterns
                    .numeric_patterns
                    .integer_frequency += 1;
            } else if s.parse::<f64>().is_ok() {
                characteristics
                    .common_patterns
                    .numeric_patterns
                    .float_frequency += 1;
            } else if s.contains('e') || s.contains('E') {
                if s.replace('e', "")
                    .replace('E', "")
                    .replace('+', "")
                    .replace('-', "")
                    .parse::<f64>()
                    .is_ok()
                {
                    characteristics
                        .common_patterns
                        .numeric_patterns
                        .scientific_frequency += 1;
                }
            } else if s.starts_with('$') || s.starts_with('€') || s.starts_with('£') {
                if s[1..].replace(',', "").parse::<f64>().is_ok() {
                    characteristics
                        .common_patterns
                        .numeric_patterns
                        .currency_frequency += 1;
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
}

impl PatternDetector for DateTimePatternDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            // ISO date pattern (YYYY-MM-DD)
            if self.matches_iso_date(s) {
                characteristics
                    .common_patterns
                    .datetime_patterns
                    .iso_date_frequency += 1;
            }
            // US date pattern (MM/DD/YYYY)
            else if self.matches_us_date(s) {
                characteristics
                    .common_patterns
                    .datetime_patterns
                    .us_date_frequency += 1;
            }
            // European date pattern (DD/MM/YYYY)
            else if self.matches_eu_date(s) {
                characteristics
                    .common_patterns
                    .datetime_patterns
                    .eu_date_frequency += 1;
            }
            // Unix timestamp
            else if s.parse::<u64>().is_ok() && s.len() == 10 {
                characteristics
                    .common_patterns
                    .datetime_patterns
                    .timestamp_frequency += 1;
            }
        }
        Ok(())
    }
}

impl DateTimePatternDetector {
    fn matches_iso_date(&self, s: &str) -> bool {
        s.len() == 10
            && s.chars().nth(4) == Some('-')
            && s.chars().nth(7) == Some('-')
            && s[..4].parse::<u16>().is_ok()
            && s[5..7].parse::<u8>().is_ok()
            && s[8..10].parse::<u8>().is_ok()
    }

    fn matches_us_date(&self, s: &str) -> bool {
        let parts: Vec<&str> = s.split('/').collect();
        parts.len() == 3
            && parts[0].parse::<u8>().is_ok()
            && parts[1].parse::<u8>().is_ok()
            && parts[2].parse::<u16>().is_ok()
    }

    fn matches_eu_date(&self, s: &str) -> bool {
        let parts: Vec<&str> = s.split('/').collect();
        parts.len() == 3
            && parts[0].parse::<u8>().is_ok()
            && parts[1].parse::<u8>().is_ok()
            && parts[2].parse::<u16>().is_ok()
    }
}

/// Structured pattern detector (URLs, emails, etc.)
pub struct StructuredPatternDetector;

impl StructuredPatternDetector {
    pub fn new() -> Self {
        Self
    }
}

impl PatternDetector for StructuredPatternDetector {
    fn analyze(
        &self,
        strings: &[String],
        characteristics: &mut StringCharacteristics,
    ) -> Result<()> {
        for s in strings {
            if s.starts_with("http://") || s.starts_with("https://") {
                characteristics
                    .common_patterns
                    .structured_patterns
                    .url_frequency += 1;
            } else if s.contains('@') && s.contains('.') {
                characteristics
                    .common_patterns
                    .structured_patterns
                    .email_frequency += 1;
            } else if self.matches_uuid(s) {
                characteristics
                    .common_patterns
                    .structured_patterns
                    .uuid_frequency += 1;
            } else if (s.starts_with('{') && s.ends_with('}'))
                || (s.starts_with('[') && s.ends_with(']'))
            {
                characteristics
                    .common_patterns
                    .structured_patterns
                    .json_frequency += 1;
            }
        }
        Ok(())
    }
}

impl StructuredPatternDetector {
    fn matches_uuid(&self, s: &str) -> bool {
        s.len() == 36
            && s.chars().nth(8) == Some('-')
            && s.chars().nth(13) == Some('-')
            && s.chars().nth(18) == Some('-')
            && s.chars().nth(23) == Some('-')
    }
}

/// String compression engine
pub struct StringCompressionEngine {
    algorithm: StringCompressionAlgorithm,
    dictionary: Option<Arc<CompressionDictionary>>,
}

impl StringCompressionEngine {
    pub fn new(algorithm: StringCompressionAlgorithm) -> Self {
        Self {
            algorithm,
            dictionary: None,
        }
    }

    pub fn with_dictionary(
        algorithm: StringCompressionAlgorithm,
        dictionary: Arc<CompressionDictionary>,
    ) -> Self {
        Self {
            algorithm,
            dictionary: Some(dictionary),
        }
    }

    pub fn compress(&self, data: &str) -> Result<Vec<u8>> {
        match self.algorithm {
            StringCompressionAlgorithm::None => Ok(data.as_bytes().to_vec()),
            StringCompressionAlgorithm::RunLength => self.run_length_encode(data),
            StringCompressionAlgorithm::Dictionary => self.dictionary_compress(data),
            StringCompressionAlgorithm::Lz4 => self.lz4_compress(data),
            StringCompressionAlgorithm::Zstd => self.zstd_compress(data),
            StringCompressionAlgorithm::StringOptimized => self.string_optimized_compress(data),
        }
    }

    pub fn decompress(&self, data: &[u8]) -> Result<String> {
        match self.algorithm {
            StringCompressionAlgorithm::None => String::from_utf8(data.to_vec())
                .map_err(|e| Error::InvalidOperation(format!("UTF-8 decode error: {}", e))),
            StringCompressionAlgorithm::RunLength => self.run_length_decode(data),
            StringCompressionAlgorithm::Dictionary => self.dictionary_decompress(data),
            StringCompressionAlgorithm::Lz4 => self.lz4_decompress(data),
            StringCompressionAlgorithm::Zstd => self.zstd_decompress(data),
            StringCompressionAlgorithm::StringOptimized => self.string_optimized_decompress(data),
        }
    }

    fn run_length_encode(&self, data: &str) -> Result<Vec<u8>> {
        let mut encoded = Vec::new();
        let bytes = data.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            let current_byte = bytes[i];
            let mut count = 1u8;

            while i + (count as usize) < bytes.len()
                && bytes[i + (count as usize)] == current_byte
                && count < 255
            {
                count += 1;
            }

            encoded.push(count);
            encoded.push(current_byte);
            i += count as usize;
        }

        Ok(encoded)
    }

    fn run_length_decode(&self, data: &[u8]) -> Result<String> {
        let mut decoded = Vec::new();
        let mut i = 0;

        while i + 1 < data.len() {
            let count = data[i];
            let byte_val = data[i + 1];

            for _ in 0..count {
                decoded.push(byte_val);
            }

            i += 2;
        }

        String::from_utf8(decoded)
            .map_err(|e| Error::InvalidOperation(format!("UTF-8 decode error: {}", e)))
    }

    fn dictionary_compress(&self, data: &str) -> Result<Vec<u8>> {
        if let Some(ref dict) = self.dictionary {
            dict.compress(data)
        } else {
            // Simple dictionary compression without external dictionary
            Ok(data.as_bytes().to_vec())
        }
    }

    fn dictionary_decompress(&self, data: &[u8]) -> Result<String> {
        if let Some(ref dict) = self.dictionary {
            dict.decompress(data)
        } else {
            String::from_utf8(data.to_vec())
                .map_err(|e| Error::InvalidOperation(format!("UTF-8 decode error: {}", e)))
        }
    }

    fn lz4_compress(&self, data: &str) -> Result<Vec<u8>> {
        // Placeholder implementation - would use actual LZ4
        let mut compressed = vec![data.len() as u8];
        compressed.extend(data.as_bytes());
        Ok(compressed)
    }

    fn lz4_decompress(&self, data: &[u8]) -> Result<String> {
        if data.is_empty() {
            return Ok(String::new());
        }

        String::from_utf8(data[1..].to_vec())
            .map_err(|e| Error::InvalidOperation(format!("UTF-8 decode error: {}", e)))
    }

    fn zstd_compress(&self, data: &str) -> Result<Vec<u8>> {
        // Placeholder implementation - would use actual ZSTD
        let mut compressed = vec![(data.len() >> 8) as u8, data.len() as u8];
        compressed.extend(data.as_bytes());
        Ok(compressed)
    }

    fn zstd_decompress(&self, data: &[u8]) -> Result<String> {
        if data.len() < 2 {
            return Ok(String::new());
        }

        String::from_utf8(data[2..].to_vec())
            .map_err(|e| Error::InvalidOperation(format!("UTF-8 decode error: {}", e)))
    }

    fn string_optimized_compress(&self, data: &str) -> Result<Vec<u8>> {
        // Custom string-optimized compression combining multiple techniques
        if data.len() < 4 {
            return Ok(data.as_bytes().to_vec());
        }

        // Try run-length encoding first
        let rle_result = self.run_length_encode(data)?;

        // If RLE doesn't provide good compression, use dictionary or LZ4
        if rle_result.len() < data.len() * 8 / 10 {
            Ok(rle_result)
        } else {
            self.lz4_compress(data)
        }
    }

    fn string_optimized_decompress(&self, data: &[u8]) -> Result<String> {
        // Try run-length decoding first, fall back to LZ4
        if let Ok(result) = self.run_length_decode(data) {
            Ok(result)
        } else {
            self.lz4_decompress(data)
        }
    }
}

/// Compression dictionary for string optimization
pub struct CompressionDictionary {
    /// Word to ID mapping
    word_to_id: HashMap<String, u32>,
    /// ID to word mapping
    id_to_word: Vec<String>,
    /// Next available ID
    next_id: u32,
}

impl CompressionDictionary {
    pub fn new() -> Self {
        Self {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
            next_id: 0,
        }
    }

    pub fn build_from_strings(&mut self, strings: &[String]) -> Result<()> {
        let mut word_counts = HashMap::new();

        // Count word frequencies
        for s in strings {
            for word in s.split_whitespace() {
                *word_counts.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        // Add most frequent words to dictionary
        let mut word_freq: Vec<_> = word_counts.into_iter().collect();
        word_freq.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        for (word, _) in word_freq.into_iter().take(1000) {
            // Top 1000 words
            self.add_word(word);
        }

        Ok(())
    }

    fn add_word(&mut self, word: String) -> u32 {
        if let Some(&id) = self.word_to_id.get(&word) {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.word_to_id.insert(word.clone(), id);
            self.id_to_word.push(word);
            id
        }
    }

    pub fn compress(&self, data: &str) -> Result<Vec<u8>> {
        let mut compressed = Vec::new();

        for word in data.split_whitespace() {
            if let Some(&id) = self.word_to_id.get(word) {
                // Use dictionary reference
                compressed.push(0xFF); // Dictionary marker
                compressed.extend_from_slice(&id.to_le_bytes());
            } else {
                // Store word literally
                compressed.push(word.len() as u8);
                compressed.extend_from_slice(word.as_bytes());
            }
        }

        Ok(compressed)
    }

    pub fn decompress(&self, data: &[u8]) -> Result<String> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < data.len() {
            if data[i] == 0xFF {
                // Dictionary reference
                if i + 4 < data.len() {
                    let id =
                        u32::from_le_bytes([data[i + 1], data[i + 2], data[i + 3], data[i + 4]]);
                    if let Some(word) = self.id_to_word.get(id as usize) {
                        result.extend_from_slice(word.as_bytes());
                        result.push(b' ');
                    }
                    i += 5;
                } else {
                    break;
                }
            } else {
                // Literal word
                let len = data[i] as usize;
                if i + 1 + len <= data.len() {
                    result.extend_from_slice(&data[i + 1..i + 1 + len]);
                    result.push(b' ');
                    i += 1 + len;
                } else {
                    break;
                }
            }
        }

        // Remove trailing space
        if result.last() == Some(&b' ') {
            result.pop();
        }

        String::from_utf8(result)
            .map_err(|e| Error::InvalidOperation(format!("UTF-8 decode error: {}", e)))
    }
}

/// Adaptive String Pool Strategy Implementation
pub struct AdaptiveStringPoolStrategy {
    /// Configuration
    config: StringPoolConfig,

    /// String storage
    string_storage: Arc<Mutex<HashMap<StringId, StringEntry>>>,

    /// String lookup for deduplication
    string_lookup: Arc<Mutex<HashMap<u64, StringId>>>,

    /// Pattern analyzer
    pattern_analyzer: StringPatternAnalyzer,

    /// Compression engines for different algorithms
    compression_engines: HashMap<StringCompressionAlgorithm, StringCompressionEngine>,

    /// Dictionary for dictionary compression
    dictionary: Arc<CompressionDictionary>,

    /// Next string ID
    next_string_id: std::sync::atomic::AtomicU64,

    /// Pool statistics
    statistics: Arc<Mutex<StringPoolStatistics>>,
}

impl AdaptiveStringPoolStrategy {
    pub fn new(config: StringPoolConfig) -> Self {
        let mut compression_engines = HashMap::new();
        compression_engines.insert(
            StringCompressionAlgorithm::None,
            StringCompressionEngine::new(StringCompressionAlgorithm::None),
        );
        compression_engines.insert(
            StringCompressionAlgorithm::RunLength,
            StringCompressionEngine::new(StringCompressionAlgorithm::RunLength),
        );
        compression_engines.insert(
            StringCompressionAlgorithm::Lz4,
            StringCompressionEngine::new(StringCompressionAlgorithm::Lz4),
        );
        compression_engines.insert(
            StringCompressionAlgorithm::Zstd,
            StringCompressionEngine::new(StringCompressionAlgorithm::Zstd),
        );
        compression_engines.insert(
            StringCompressionAlgorithm::StringOptimized,
            StringCompressionEngine::new(StringCompressionAlgorithm::StringOptimized),
        );

        let dictionary = Arc::new(CompressionDictionary::new());
        compression_engines.insert(
            StringCompressionAlgorithm::Dictionary,
            StringCompressionEngine::with_dictionary(
                StringCompressionAlgorithm::Dictionary,
                dictionary.clone(),
            ),
        );

        Self {
            pattern_analyzer: StringPatternAnalyzer::new(config.clone()),
            config,
            string_storage: Arc::new(Mutex::new(HashMap::new())),
            string_lookup: Arc::new(Mutex::new(HashMap::new())),
            compression_engines,
            dictionary,
            next_string_id: std::sync::atomic::AtomicU64::new(1),
            statistics: Arc::new(Mutex::new(StringPoolStatistics::new())),
        }
    }

    fn compute_string_hash(&self, s: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    fn store_string_with_strategy(
        &self,
        s: &str,
        strategy: StringStorageStrategy,
        compression: StringCompressionAlgorithm,
    ) -> Result<StringId> {
        let string_hash = self.compute_string_hash(s);

        // Check for existing string (deduplication)
        if let Ok(lookup) = self.string_lookup.lock() {
            if let Some(&existing_id) = lookup.get(&string_hash) {
                // Update reference count
                if let Ok(mut storage) = self.string_storage.lock() {
                    if let Some(entry) = storage.get_mut(&existing_id) {
                        entry.ref_count += 1;
                        entry.last_accessed = Instant::now();
                        entry.access_count += 1;
                        return Ok(existing_id);
                    }
                }
            }
        }

        // Create new string entry
        let string_id = StringId(
            self.next_string_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
        );

        let compression_engine = self.compression_engines.get(&compression).ok_or_else(|| {
            Error::InvalidOperation(format!("Compression algorithm {:?} not found", compression))
        })?;

        let compressed_data = compression_engine.compress(s)?;
        let compressed_size = compressed_data.len();

        let entry = StringEntry {
            id: string_id,
            strategy,
            data: compressed_data,
            original_length: s.len(),
            ref_count: 1,
            first_accessed: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
            compression,
            metadata: StringMetadata {
                encoding: StringEncoding::Utf8,
                dictionary_refs: Vec::new(),
                reconstruction_data: HashMap::new(),
            },
        };

        // Store the entry
        if let Ok(mut storage) = self.string_storage.lock() {
            storage.insert(string_id, entry);
        }

        if let Ok(mut lookup) = self.string_lookup.lock() {
            lookup.insert(string_hash, string_id);
        }

        // Update statistics
        if let Ok(mut stats) = self.statistics.lock() {
            stats.total_strings += 1;
            stats.original_size += s.len();
            stats.storage_used += compressed_size;

            if stats.original_size > 0 {
                stats.compression_ratio = stats.storage_used as f64 / stats.original_size as f64;
            }
        }

        Ok(string_id)
    }

    fn retrieve_string(&self, string_id: StringId) -> Result<String> {
        let entry = {
            if let Ok(mut storage) = self.string_storage.lock() {
                if let Some(entry) = storage.get_mut(&string_id) {
                    entry.last_accessed = Instant::now();
                    entry.access_count += 1;
                    entry.clone()
                } else {
                    return Err(Error::InvalidOperation(format!(
                        "String ID {:?} not found",
                        string_id
                    )));
                }
            } else {
                return Err(Error::InvalidOperation(
                    "Failed to acquire storage lock".to_string(),
                ));
            }
        };

        let compression_engine = self
            .compression_engines
            .get(&entry.compression)
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Compression algorithm {:?} not found",
                    entry.compression
                ))
            })?;

        compression_engine.decompress(&entry.data)
    }

    fn analyze_and_optimize(&mut self, sample_strings: &[String]) -> Result<()> {
        if sample_strings.is_empty() {
            return Ok(());
        }

        let characteristics = self.pattern_analyzer.analyze_strings(sample_strings)?;
        let recommendations = self.pattern_analyzer.recommend_strategy(&characteristics);

        // Build dictionary if dictionary encoding is recommended
        if recommendations.recommended_strategy == StringStorageStrategy::DictionaryEncoded {
            if let Some(dict) = Arc::get_mut(&mut self.dictionary) {
                dict.build_from_strings(sample_strings)?;
            }
        }

        Ok(())
    }
}

impl StorageStrategy for AdaptiveStringPoolStrategy {
    type Handle = StringPoolHandle;
    type Error = Error;
    type Metadata = StringPoolStatistics;

    fn name(&self) -> &'static str {
        "AdaptiveStringPool"
    }

    fn create_storage(&mut self, config: &StorageConfig) -> Result<Self::Handle> {
        // Analyze data sample if provided
        if let Some(ref data_sample) = config.data_sample {
            if let Ok(sample_data) = String::from_utf8(data_sample.clone()) {
                let sample_strings: Vec<String> =
                    sample_data.lines().map(|s| s.to_string()).collect();
                self.analyze_and_optimize(&sample_strings)?;
            }
        }

        let handle = StringPoolHandle {
            config: self.config.clone(),
            current_strategy: StringStorageStrategy::Hybrid,
            string_count: 0,
            statistics: StringPoolStatistics::new(),
            analyzer_state: PatternAnalyzerState {
                recent_strings: VecDeque::new(),
                characteristics: StringCharacteristics::new(),
                analysis_window: self.config.analysis_window_size,
                last_analysis: Instant::now(),
                strategy_recommendations: StrategyRecommendations {
                    recommended_strategy: StringStorageStrategy::Raw,
                    recommended_compression: StringCompressionAlgorithm::None,
                    confidence: 0.5,
                    expected_savings: 0.0,
                },
            },
        };

        Ok(handle)
    }

    fn read_chunk(&self, handle: &Self::Handle, range: ChunkRange) -> Result<DataChunk> {
        // For string pool, we interpret the range as string IDs
        let start_id = StringId(range.start as u64);
        let end_id = StringId(range.end as u64);

        let mut strings = Vec::new();
        for id_val in start_id.0..end_id.0 {
            let string_id = StringId(id_val);
            if let Ok(s) = self.retrieve_string(string_id) {
                strings.push(s);
            }
        }

        Ok(DataChunk::from_strings(strings))
    }

    fn write_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        let strings = chunk.as_strings()?;

        // Analyze patterns if enabled
        if self.config.enable_pattern_analysis && !strings.is_empty() {
            self.analyze_and_optimize(&strings)?;
        }

        // Store strings using optimal strategy
        for s in strings {
            let strategy = handle
                .analyzer_state
                .strategy_recommendations
                .recommended_strategy;
            let compression = handle
                .analyzer_state
                .strategy_recommendations
                .recommended_compression;
            self.store_string_with_strategy(&s, strategy, compression)?;
        }

        Ok(())
    }

    fn append_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        self.write_chunk(handle, chunk)
    }

    fn flush(&mut self, _handle: &Self::Handle) -> Result<()> {
        // String pool is always in-memory, no flushing needed
        Ok(())
    }

    fn delete_storage(&mut self, _handle: &Self::Handle) -> Result<()> {
        // Clear all storage
        if let Ok(mut storage) = self.string_storage.lock() {
            storage.clear();
        }
        if let Ok(mut lookup) = self.string_lookup.lock() {
            lookup.clear();
        }

        Ok(())
    }

    fn can_handle(&self, requirements: &StorageRequirements) -> StrategyCapability {
        let can_handle = match requirements.data_characteristics {
            DataCharacteristics::Text => true,
            DataCharacteristics::Categorical => true,
            DataCharacteristics::Mixed => requirements.estimated_size < 1024 * 1024 * 1024, // < 1GB
            _ => false,
        };

        let confidence = if can_handle { 0.9 } else { 0.1 };

        let performance_score = match requirements.performance_priority {
            PerformancePriority::Memory => 0.95, // Excellent memory efficiency
            PerformancePriority::Speed => 0.8,   // Good speed with deduplication
            PerformancePriority::Balanced => 0.9,
            _ => 0.7,
        };

        StrategyCapability {
            can_handle,
            confidence,
            performance_score,
            resource_cost: ResourceCost {
                memory: requirements.estimated_size / 3, // Good compression expected
                cpu: 10.0,                               // Moderate CPU usage for analysis
                disk: 0,                                 // In-memory storage
                network: 0,
            },
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            read_speed: Speed::VeryFast,
            write_speed: Speed::Fast,
            memory_efficiency: Efficiency::Excellent,
            compression_ratio: 3.5, // Expected ratio with deduplication
            query_optimization: QueryOptimization::Good,
            parallel_scalability: ParallelScalability::Good,
        }
    }

    fn storage_stats(&self) -> StorageStats {
        if let Ok(stats) = self.statistics.lock() {
            StorageStats {
                total_size: stats.storage_used,
                used_size: stats.storage_used,
                read_operations: stats.total_strings,
                write_operations: stats.total_strings,
                avg_read_latency_ns: stats.avg_access_time.as_nanos() as u64,
                avg_write_latency_ns: stats.avg_access_time.as_nanos() as u64,
                cache_hit_rate: stats.cache_hit_rate,
            }
        } else {
            StorageStats::default()
        }
    }

    fn optimize_for_pattern(&mut self, pattern: AccessPattern) -> Result<()> {
        match pattern {
            AccessPattern::HighDuplication => {
                self.config.deduplication_threshold = 1; // Deduplicate all strings
                self.config.enable_dictionary_encoding = true;
            }
            AccessPattern::LongStrings => {
                self.config.compression_threshold = 0.05; // Lower threshold for long strings
                self.config.enable_adaptive_compression = true;
            }
            AccessPattern::ShortStrings => {
                self.config.enable_dictionary_encoding = true;
                self.config.max_dictionary_size = 2 * 1024 * 1024; // Larger dictionary
            }
            _ => {
                // Use default settings
            }
        }

        Ok(())
    }

    fn compact(&mut self, _handle: &Self::Handle) -> Result<CompactionResult> {
        let start_time = Instant::now();
        let size_before = if let Ok(stats) = self.statistics.lock() {
            stats.storage_used
        } else {
            0
        };

        // Remove unreferenced strings
        let mut removed_count = 0;
        if let Ok(mut storage) = self.string_storage.lock() {
            storage.retain(|_, entry| {
                if entry.ref_count == 0 {
                    removed_count += 1;
                    false
                } else {
                    true
                }
            });
        }

        let size_after = if let Ok(stats) = self.statistics.lock() {
            stats.storage_used
        } else {
            0
        };

        Ok(CompactionResult {
            size_before,
            size_after,
            duration: start_time.elapsed(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_pattern_analyzer() {
        let config = StringPoolConfig::default();
        let analyzer = StringPatternAnalyzer::new(config);

        let strings = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "goodbye world".to_string(),
            "123".to_string(),
            "456.78".to_string(),
        ];

        let characteristics = analyzer.analyze_strings(&strings).unwrap();
        assert!(characteristics.common_patterns.common_prefixes.len() > 0);
        assert!(
            characteristics
                .common_patterns
                .numeric_patterns
                .integer_frequency
                > 0
        );
    }

    #[test]
    fn test_string_compression_engine() {
        let engine = StringCompressionEngine::new(StringCompressionAlgorithm::RunLength);
        let test_string = "aaaaaabbbbbbcccccc";

        let compressed = engine.compress(test_string).unwrap();
        let decompressed = engine.decompress(&compressed).unwrap();

        assert_eq!(test_string, decompressed);
        assert!(compressed.len() < test_string.len());
    }

    #[test]
    fn test_compression_dictionary() {
        let mut dict = CompressionDictionary::new();
        let strings = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "world peace".to_string(),
        ];

        dict.build_from_strings(&strings).unwrap();

        let test_string = "hello world";
        let compressed = dict.compress(test_string).unwrap();
        let decompressed = dict.decompress(&compressed).unwrap();

        assert_eq!(test_string, decompressed);
    }

    #[test]
    fn test_adaptive_string_pool_strategy() {
        let config = StringPoolConfig::default();
        let mut strategy = AdaptiveStringPoolStrategy::new(config);

        let storage_config = StorageConfig {
            requirements: StorageRequirements {
                estimated_size: 1024,
                data_characteristics: DataCharacteristics::Text,
                ..Default::default()
            },
            ..Default::default()
        };

        let handle = strategy.create_storage(&storage_config).unwrap();

        let strings = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello".to_string(),
        ];
        let chunk = DataChunk::from_strings(strings);

        strategy.write_chunk(&handle, chunk).unwrap();

        let stats = strategy.storage_stats();
        assert!(stats.total_size > 0);
    }

    #[test]
    fn test_pattern_detectors() {
        let detector = NumericPatternDetector::new();
        let mut characteristics = StringCharacteristics::new();
        let strings = vec![
            "123".to_string(),
            "45.67".to_string(),
            "not_a_number".to_string(),
        ];

        detector.analyze(&strings, &mut characteristics).unwrap();

        assert_eq!(
            characteristics
                .common_patterns
                .numeric_patterns
                .integer_frequency,
            1
        );
        assert_eq!(
            characteristics
                .common_patterns
                .numeric_patterns
                .float_frequency,
            1
        );
    }

    #[test]
    fn test_strategy_capability_assessment() {
        let config = StringPoolConfig::default();
        let strategy = AdaptiveStringPoolStrategy::new(config);

        let requirements = StorageRequirements {
            estimated_size: 10 * 1024, // 10KB
            data_characteristics: DataCharacteristics::Text,
            performance_priority: PerformancePriority::Memory,
            ..Default::default()
        };

        let capability = strategy.can_handle(&requirements);
        assert!(capability.can_handle);
        assert!(capability.confidence > 0.8);
        assert!(capability.performance_score > 0.9);
    }
}
