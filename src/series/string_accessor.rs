use crate::core::error::Error as PandrsError;
use crate::series::base::Series;
use regex::Regex;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Regex cache for performance optimization
/// Caches compiled regex patterns to avoid recompilation
static REGEX_CACHE: OnceLock<Arc<Mutex<HashMap<String, Regex>>>> = OnceLock::new();

/// Get or compile a regex pattern with caching
fn get_or_compile_regex(pattern: &str, case_insensitive: bool) -> Result<Regex, PandrsError> {
    let cache_key = if case_insensitive {
        format!("(?i){}", pattern)
    } else {
        pattern.to_string()
    };

    // Initialize cache if needed
    let cache = REGEX_CACHE.get_or_init(|| Arc::new(Mutex::new(HashMap::new())));

    // Try to get from cache first
    {
        let cache_guard = cache.lock().unwrap();
        if let Some(regex) = cache_guard.get(&cache_key) {
            return Ok(regex.clone());
        }
    }

    // Compile new regex
    let regex = if case_insensitive {
        Regex::new(&format!("(?i){}", pattern))
    } else {
        Regex::new(pattern)
    }
    .map_err(|e| {
        PandrsError::InvalidValue(format!("Invalid regex pattern '{}': {}", pattern, e))
    })?;

    // Store in cache
    {
        let mut cache_guard = cache.lock().unwrap();
        cache_guard.insert(cache_key, regex.clone());
    }

    Ok(regex)
}

/// String accessor for Series containing string data
/// Provides pandas-like string operations through .str accessor
#[derive(Clone)]
pub struct StringAccessor {
    series: Series<String>,
}

impl StringAccessor {
    /// Create a new StringAccessor
    pub fn new(series: Series<String>) -> Result<Self, PandrsError> {
        Ok(StringAccessor { series })
    }

    /// Convert all strings to uppercase
    pub fn upper(&self) -> Result<Series<String>, PandrsError> {
        let upper_values: Vec<String> = self
            .series
            .values()
            .iter()
            .map(|s| s.to_uppercase())
            .collect();

        Series::new(upper_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Convert all strings to lowercase
    pub fn lower(&self) -> Result<Series<String>, PandrsError> {
        let lower_values: Vec<String> = self
            .series
            .values()
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        Series::new(lower_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Convert strings to title case
    pub fn title(&self) -> Result<Series<String>, PandrsError> {
        let title_values: Vec<String> =
            self.series.values().iter().map(|s| title_case(s)).collect();

        Series::new(title_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Capitalize first character of each string
    pub fn capitalize(&self) -> Result<Series<String>, PandrsError> {
        let cap_values: Vec<String> = self
            .series
            .values()
            .iter()
            .map(|s| capitalize_string(s))
            .collect();

        Series::new(cap_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Check if strings contain a pattern
    ///
    /// # Arguments
    /// * `pattern` - Pattern to search for
    /// * `case` - If true, perform case-sensitive search
    /// * `regex` - If true, treat pattern as regex
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["hello world".to_string(), "HELLO".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().contains("hello", false, false).unwrap();
    /// assert_eq!(result.values(), &[true, true]); // Case insensitive
    /// ```
    pub fn contains(
        &self,
        pattern: &str,
        case: bool,
        regex: bool,
    ) -> Result<Series<bool>, PandrsError> {
        if regex {
            let re = get_or_compile_regex(pattern, !case)?;

            let bool_values: Vec<bool> = self
                .series
                .values()
                .iter()
                .map(|s| re.is_match(s))
                .collect();

            Series::new(bool_values, self.series.name().cloned())
                .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
        } else {
            let bool_values: Vec<bool> = if case {
                self.series
                    .values()
                    .iter()
                    .map(|s| s.contains(pattern))
                    .collect()
            } else {
                let pattern_lower = pattern.to_lowercase();
                self.series
                    .values()
                    .iter()
                    .map(|s| s.to_lowercase().contains(&pattern_lower))
                    .collect()
            };

            Series::new(bool_values, self.series.name().cloned())
                .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
        }
    }

    /// Check if strings start with a pattern
    pub fn startswith(&self, pattern: &str, case: bool) -> Result<Series<bool>, PandrsError> {
        let bool_values: Vec<bool> = if case {
            self.series
                .values()
                .iter()
                .map(|s| s.starts_with(pattern))
                .collect()
        } else {
            let pattern_lower = pattern.to_lowercase();
            self.series
                .values()
                .iter()
                .map(|s| s.to_lowercase().starts_with(&pattern_lower))
                .collect()
        };

        Series::new(bool_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Check if strings end with a pattern
    pub fn endswith(&self, pattern: &str, case: bool) -> Result<Series<bool>, PandrsError> {
        let bool_values: Vec<bool> = if case {
            self.series
                .values()
                .iter()
                .map(|s| s.ends_with(pattern))
                .collect()
        } else {
            let pattern_lower = pattern.to_lowercase();
            self.series
                .values()
                .iter()
                .map(|s| s.to_lowercase().ends_with(&pattern_lower))
                .collect()
        };

        Series::new(bool_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Replace occurrences of a pattern
    ///
    /// # Arguments
    /// * `pattern` - Pattern to replace
    /// * `replacement` - Replacement string
    /// * `regex` - If true, treat pattern as regex
    /// * `case` - If true, perform case-sensitive replacement
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["hello world".to_string(), "test".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().replace("world", "rust", false, true).unwrap();
    /// assert_eq!(result.values()[0], "hello rust");
    /// ```
    pub fn replace(
        &self,
        pattern: &str,
        replacement: &str,
        regex: bool,
        case: bool,
    ) -> Result<Series<String>, PandrsError> {
        if regex {
            let re = get_or_compile_regex(pattern, !case)?;

            let replaced_values: Vec<String> = self
                .series
                .values()
                .iter()
                .map(|s| re.replace_all(s, replacement).to_string())
                .collect();

            Series::new(replaced_values, self.series.name().cloned())
                .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
        } else {
            let replaced_values: Vec<String> = if case {
                self.series
                    .values()
                    .iter()
                    .map(|s| s.replace(pattern, replacement))
                    .collect()
            } else {
                // Case-insensitive replacement without regex
                self.series
                    .values()
                    .iter()
                    .map(|s| case_insensitive_replace(s, pattern, replacement))
                    .collect()
            };

            Series::new(replaced_values, self.series.name().cloned())
                .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
        }
    }

    /// Split strings by delimiter
    pub fn split(
        &self,
        delimiter: &str,
        n: Option<usize>,
        expand: bool,
    ) -> Result<Series<String>, PandrsError> {
        if expand {
            // Return multiple columns (not implemented yet, return error)
            return Err(PandrsError::NotImplemented(
                "split with expand=true not yet implemented".to_string(),
            ));
        }

        let split_values: Vec<Vec<String>> = self
            .series
            .values()
            .iter()
            .map(|s| {
                if let Some(max_splits) = n {
                    s.splitn(max_splits + 1, delimiter)
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    s.split(delimiter).map(|s| s.to_string()).collect()
                }
            })
            .collect();

        // Convert Vec<Vec<String>> to appropriate Series representation
        // For now, convert to strings representation
        let result_strings: Vec<String> = split_values
            .iter()
            .map(|parts| format!("[{}]", parts.join(", ")))
            .collect();

        Series::new(result_strings, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Get string length (character count, not byte count)
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["hello".to_string(), "café".to_string(), "🦀".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let lengths = series.str().unwrap().len().unwrap();
    /// assert_eq!(lengths.values(), &[5i64, 4i64, 1i64]); // Character count, not bytes
    /// ```
    pub fn len(&self) -> Result<Series<i64>, PandrsError> {
        let lengths: Vec<i64> = self.series.values()
            .iter()
            .map(|s| s.chars().count() as i64)  // Use character count for Unicode safety
            .collect();

        Series::new(lengths, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Strip whitespace from both ends
    pub fn strip(&self, chars: Option<&str>) -> Result<Series<String>, PandrsError> {
        let stripped_values: Vec<String> = if let Some(chars_to_strip) = chars {
            self.series
                .values()
                .iter()
                .map(|s| strip_chars(s, chars_to_strip))
                .collect()
        } else {
            self.series
                .values()
                .iter()
                .map(|s| s.trim().to_string())
                .collect()
        };

        Series::new(stripped_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Strip whitespace from left end
    pub fn lstrip(&self, chars: Option<&str>) -> Result<Series<String>, PandrsError> {
        let stripped_values: Vec<String> = if let Some(chars_to_strip) = chars {
            self.series
                .values()
                .iter()
                .map(|s| lstrip_chars(s, chars_to_strip))
                .collect()
        } else {
            self.series
                .values()
                .iter()
                .map(|s| s.trim_start().to_string())
                .collect()
        };

        Series::new(stripped_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Strip whitespace from right end
    pub fn rstrip(&self, chars: Option<&str>) -> Result<Series<String>, PandrsError> {
        let stripped_values: Vec<String> = if let Some(chars_to_strip) = chars {
            self.series
                .values()
                .iter()
                .map(|s| rstrip_chars(s, chars_to_strip))
                .collect()
        } else {
            self.series
                .values()
                .iter()
                .map(|s| s.trim_end().to_string())
                .collect()
        };

        Series::new(stripped_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract substring using regex groups
    ///
    /// # Arguments
    /// * `pattern` - Regex pattern with capture groups
    /// * `flags` - Optional regex flags (e.g., "i" for case insensitive)
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["abc123def".to_string(), "xyz456ghi".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().extract(r"(\d+)", None).unwrap();
    /// assert_eq!(result.values(), &["123".to_string(), "456".to_string()]);
    /// ```
    pub fn extract(
        &self,
        pattern: &str,
        flags: Option<&str>,
    ) -> Result<Series<String>, PandrsError> {
        let case_insensitive = flags.map_or(false, |f| f.contains('i'));
        let re = get_or_compile_regex(pattern, case_insensitive)?;

        let extracted_values: Vec<String> = self
            .series
            .values()
            .iter()
            .map(|s| {
                if let Some(caps) = re.captures(s) {
                    if caps.len() > 1 {
                        caps.get(1)
                            .map(|m| m.as_str().to_string())
                            .unwrap_or_else(|| "".to_string())
                    } else {
                        caps.get(0)
                            .map(|m| m.as_str().to_string())
                            .unwrap_or_else(|| "".to_string())
                    }
                } else {
                    "".to_string()
                }
            })
            .collect();

        Series::new(extracted_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Find all matches of pattern
    ///
    /// # Arguments
    /// * `pattern` - Regex pattern to find
    /// * `flags` - Optional regex flags (e.g., "i" for case insensitive)
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["abc123def456".to_string(), "nodigits".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().findall(r"\d+", None).unwrap();
    /// assert!(result.values()[0].contains("123"));
    /// assert!(result.values()[0].contains("456"));
    /// ```
    pub fn findall(
        &self,
        pattern: &str,
        flags: Option<&str>,
    ) -> Result<Series<String>, PandrsError> {
        let case_insensitive = flags.map_or(false, |f| f.contains('i'));
        let re = get_or_compile_regex(pattern, case_insensitive)?;

        let found_values: Vec<String> = self
            .series
            .values()
            .iter()
            .map(|s| {
                let matches: Vec<String> =
                    re.find_iter(s).map(|m| m.as_str().to_string()).collect();
                format!("[{}]", matches.join(", "))
            })
            .collect();

        Series::new(found_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Count occurrences of pattern
    ///
    /// # Arguments
    /// * `pattern` - Regex pattern to count
    /// * `flags` - Optional regex flags (e.g., "i" for case insensitive)
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["abc123def456".to_string(), "nodigits".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().count(r"\d", None).unwrap();
    /// assert_eq!(result.values(), &[6i64, 0i64]); // 6 digits in first, 0 in second
    /// ```
    pub fn count(&self, pattern: &str, flags: Option<&str>) -> Result<Series<i64>, PandrsError> {
        let case_insensitive = flags.map_or(false, |f| f.contains('i'));
        let re = get_or_compile_regex(pattern, case_insensitive)?;

        let counts: Vec<i64> = self
            .series
            .values()
            .iter()
            .map(|s| re.find_iter(s).count() as i64)
            .collect();

        Series::new(counts, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Check if all characters are alphabetic
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["hello".to_string(), "world123".to_string(), "".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().isalpha().unwrap();
    /// assert_eq!(result.values(), &[true, false, false]);
    /// ```
    pub fn isalpha(&self) -> Result<Series<bool>, PandrsError> {
        let bool_values: Vec<bool> = self
            .series
            .values()
            .iter()
            .map(|s| !s.is_empty() && s.chars().all(|c| c.is_alphabetic()))
            .collect();

        Series::new(bool_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create boolean series: {:?}", e)))
    }

    /// Check if all characters are digits
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["123".to_string(), "12.3".to_string(), "".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().isdigit().unwrap();
    /// assert_eq!(result.values(), &[true, false, false]);
    /// ```
    pub fn isdigit(&self) -> Result<Series<bool>, PandrsError> {
        let bool_values: Vec<bool> = self
            .series
            .values()
            .iter()
            .map(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
            .collect();

        Series::new(bool_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create boolean series: {:?}", e)))
    }

    /// Check if all characters are alphanumeric
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["hello123".to_string(), "hello-world".to_string(), "".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().isalnum().unwrap();
    /// assert_eq!(result.values(), &[true, false, false]);
    /// ```
    pub fn isalnum(&self) -> Result<Series<bool>, PandrsError> {
        let bool_values: Vec<bool> = self
            .series
            .values()
            .iter()
            .map(|s| !s.is_empty() && s.chars().all(|c| c.is_alphanumeric()))
            .collect();

        Series::new(bool_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create boolean series: {:?}", e)))
    }

    /// Check if all characters are whitespace
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["   ".to_string(), "\t\n".to_string(), "hello".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().isspace().unwrap();
    /// assert_eq!(result.values(), &[true, true, false]);
    /// ```
    pub fn isspace(&self) -> Result<Series<bool>, PandrsError> {
        let bool_values: Vec<bool> = self
            .series
            .values()
            .iter()
            .map(|s| !s.is_empty() && s.chars().all(|c| c.is_whitespace()))
            .collect();

        Series::new(bool_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create boolean series: {:?}", e)))
    }

    /// Check if string is lowercase
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["hello".to_string(), "Hello".to_string(), "123".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().islower().unwrap();
    /// assert_eq!(result.values(), &[true, false, false]);
    /// ```
    pub fn islower(&self) -> Result<Series<bool>, PandrsError> {
        let bool_values: Vec<bool> = self
            .series
            .values()
            .iter()
            .map(|s| {
                let has_cased = s.chars().any(|c| c.is_alphabetic());
                has_cased
                    && s.chars()
                        .filter(|c| c.is_alphabetic())
                        .all(|c| c.is_lowercase())
            })
            .collect();

        Series::new(bool_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create boolean series: {:?}", e)))
    }

    /// Check if string is uppercase
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["HELLO".to_string(), "Hello".to_string(), "123".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().isupper().unwrap();
    /// assert_eq!(result.values(), &[true, false, false]);
    /// ```
    pub fn isupper(&self) -> Result<Series<bool>, PandrsError> {
        let bool_values: Vec<bool> = self
            .series
            .values()
            .iter()
            .map(|s| {
                let has_cased = s.chars().any(|c| c.is_alphabetic());
                has_cased
                    && s.chars()
                        .filter(|c| c.is_alphabetic())
                        .all(|c| c.is_uppercase())
            })
            .collect();

        Series::new(bool_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create boolean series: {:?}", e)))
    }

    /// Swap case of alphabetic characters
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["Hello World".to_string(), "RUST".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let result = series.str().unwrap().swapcase().unwrap();
    /// assert_eq!(result.values(), &["hELLO wORLD".to_string(), "rust".to_string()]);
    /// ```
    pub fn swapcase(&self) -> Result<Series<String>, PandrsError> {
        let swapped_values: Vec<String> = self
            .series
            .values()
            .iter()
            .map(|s| {
                s.chars()
                    .map(|c| {
                        if c.is_uppercase() {
                            c.to_lowercase().collect::<String>()
                        } else if c.is_lowercase() {
                            c.to_uppercase().collect::<String>()
                        } else {
                            c.to_string()
                        }
                    })
                    .collect::<String>()
            })
            .collect();

        Series::new(swapped_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Pad strings to specified width
    ///
    /// # Arguments
    /// * `width` - Target width for padding
    /// * `side` - Padding side: "left", "right", or "both"
    /// * `fillchar` - Character to use for padding
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// let data = vec!["hi".to_string(), "world".to_string()];
    /// let series = Series::new(data, None).unwrap();
    /// let padded = series.str().unwrap().pad(8, "left", '*').unwrap();
    /// assert_eq!(padded.values()[0], "******hi");
    /// ```
    ///
    /// # Errors
    /// Returns error if `side` is not one of "left", "right", or "both"
    pub fn pad(
        &self,
        width: usize,
        side: &str,
        fillchar: char,
    ) -> Result<Series<String>, PandrsError> {
        // Input validation
        if !matches!(side, "left" | "right" | "both") {
            return Err(PandrsError::InvalidValue(format!(
                "Invalid side parameter '{}'. Must be 'left', 'right', or 'both'",
                side
            )));
        }

        let padded_values: Vec<String> = self
            .series
            .values()
            .iter()
            .map(|s| {
                let char_count = s.chars().count(); // Use character count for Unicode safety
                if char_count >= width {
                    s.clone()
                } else {
                    let padding_needed = width - char_count;
                    match side {
                        "left" => format!("{}{}", fillchar.to_string().repeat(padding_needed), s),
                        "right" => format!("{}{}", s, fillchar.to_string().repeat(padding_needed)),
                        "both" => {
                            let left_pad = padding_needed / 2;
                            let right_pad = padding_needed - left_pad;
                            format!(
                                "{}{}{}",
                                fillchar.to_string().repeat(left_pad),
                                s,
                                fillchar.to_string().repeat(right_pad)
                            )
                        }
                        _ => unreachable!(), // Already validated above
                    }
                }
            })
            .collect();

        Series::new(padded_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }
}

// Helper functions

/// Convert string to title case
fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Capitalize first character of string
fn capitalize_string(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Case-insensitive string replacement
fn case_insensitive_replace(text: &str, pattern: &str, replacement: &str) -> String {
    let lower_text = text.to_lowercase();
    let lower_pattern = pattern.to_lowercase();

    if !lower_text.contains(&lower_pattern) {
        return text.to_string();
    }

    let mut result = String::new();
    let mut start = 0;

    while let Some(pos) = lower_text[start..].find(&lower_pattern) {
        let actual_pos = start + pos;
        result.push_str(&text[start..actual_pos]);
        result.push_str(replacement);
        start = actual_pos + pattern.len();
    }

    result.push_str(&text[start..]);
    result
}

/// Strip specific characters from both ends
fn strip_chars(s: &str, chars: &str) -> String {
    let chars_set: std::collections::HashSet<char> = chars.chars().collect();
    s.trim_matches(|c| chars_set.contains(&c)).to_string()
}

/// Strip specific characters from left end
fn lstrip_chars(s: &str, chars: &str) -> String {
    let chars_set: std::collections::HashSet<char> = chars.chars().collect();
    s.trim_start_matches(|c| chars_set.contains(&c)).to_string()
}

/// Strip specific characters from right end
fn rstrip_chars(s: &str, chars: &str) -> String {
    let chars_set: std::collections::HashSet<char> = chars.chars().collect();
    s.trim_end_matches(|c| chars_set.contains(&c)).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_upper() {
        let data = vec!["hello".to_string(), "world".to_string(), "RUST".to_string()];
        let series = Series::new(data, Some("test".to_string())).unwrap();
        let str_accessor = StringAccessor::new(series).unwrap();

        let result = str_accessor.upper().unwrap();
        let values = result.values();

        assert_eq!(
            values,
            &["HELLO".to_string(), "WORLD".to_string(), "RUST".to_string()]
        );
    }

    #[test]
    fn test_string_lower() {
        let data = vec!["HELLO".to_string(), "World".to_string(), "rust".to_string()];
        let series = Series::new(data, Some("test".to_string())).unwrap();
        let str_accessor = StringAccessor::new(series).unwrap();

        let result = str_accessor.lower().unwrap();
        let values = result.values();

        assert_eq!(
            values,
            &["hello".to_string(), "world".to_string(), "rust".to_string()]
        );
    }

    #[test]
    fn test_string_contains() {
        let data = vec![
            "hello world".to_string(),
            "rust programming".to_string(),
            "python data".to_string(),
        ];
        let series = Series::new(data, Some("test".to_string())).unwrap();
        let str_accessor = StringAccessor::new(series).unwrap();

        let result = str_accessor.contains("rust", true, false).unwrap();
        let values = result.values();

        assert_eq!(values, &[false, true, false]);
    }

    #[test]
    fn test_string_startswith() {
        let data = vec![
            "hello world".to_string(),
            "hello rust".to_string(),
            "goodbye".to_string(),
        ];
        let series = Series::new(data, Some("test".to_string())).unwrap();
        let str_accessor = StringAccessor::new(series).unwrap();

        let result = str_accessor.startswith("hello", true).unwrap();
        let values = result.values();

        assert_eq!(values, &[true, true, false]);
    }

    #[test]
    fn test_string_len() {
        let data = vec!["a".to_string(), "hello".to_string(), "world".to_string()];
        let series = Series::new(data, Some("test".to_string())).unwrap();
        let str_accessor = StringAccessor::new(series).unwrap();

        let result = str_accessor.len().unwrap();
        let values = result.values();

        assert_eq!(values, &[1i64, 5i64, 5i64]);
    }

    #[test]
    fn test_string_strip() {
        let data = vec![
            "  hello  ".to_string(),
            "\tworld\n".to_string(),
            " rust ".to_string(),
        ];
        let series = Series::new(data, Some("test".to_string())).unwrap();
        let str_accessor = StringAccessor::new(series).unwrap();

        let result = str_accessor.strip(None).unwrap();
        let values = result.values();

        assert_eq!(
            values,
            &["hello".to_string(), "world".to_string(), "rust".to_string()]
        );
    }
}
