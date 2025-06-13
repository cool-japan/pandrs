use crate::core::error::Error as PandrsError;
use crate::series::base::Series;
use regex::Regex;

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
        let upper_values: Vec<String> = self.series.values()
            .iter()
            .map(|s| s.to_uppercase())
            .collect();
        
        Series::new(upper_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Convert all strings to lowercase
    pub fn lower(&self) -> Result<Series<String>, PandrsError> {
        let lower_values: Vec<String> = self.series.values()
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        
        Series::new(lower_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Convert strings to title case
    pub fn title(&self) -> Result<Series<String>, PandrsError> {
        let title_values: Vec<String> = self.series.values()
            .iter()
            .map(|s| title_case(s))
            .collect();
        
        Series::new(title_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Capitalize first character of each string
    pub fn capitalize(&self) -> Result<Series<String>, PandrsError> {
        let cap_values: Vec<String> = self.series.values()
            .iter()
            .map(|s| capitalize_string(s))
            .collect();
        
        Series::new(cap_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Check if strings contain a pattern
    pub fn contains(&self, pattern: &str, case: bool, regex: bool) -> Result<Series<bool>, PandrsError> {
        if regex {
            let re = if case {
                Regex::new(pattern)
            } else {
                Regex::new(&format!("(?i){}", pattern))
            }.map_err(|e| PandrsError::InvalidValue(format!("Invalid regex pattern: {}", e)))?;
            
            let bool_values: Vec<bool> = self.series.values()
                .iter()
                .map(|s| re.is_match(s))
                .collect();
            
            Series::new(bool_values, self.series.name().cloned())
                .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
        } else {
            let bool_values: Vec<bool> = if case {
                self.series.values()
                    .iter()
                    .map(|s| s.contains(pattern))
                    .collect()
            } else {
                let pattern_lower = pattern.to_lowercase();
                self.series.values()
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
            self.series.values()
                .iter()
                .map(|s| s.starts_with(pattern))
                .collect()
        } else {
            let pattern_lower = pattern.to_lowercase();
            self.series.values()
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
            self.series.values()
                .iter()
                .map(|s| s.ends_with(pattern))
                .collect()
        } else {
            let pattern_lower = pattern.to_lowercase();
            self.series.values()
                .iter()
                .map(|s| s.to_lowercase().ends_with(&pattern_lower))
                .collect()
        };
        
        Series::new(bool_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Replace occurrences of a pattern
    pub fn replace(&self, pattern: &str, replacement: &str, regex: bool, case: bool) -> Result<Series<String>, PandrsError> {
        if regex {
            let re = if case {
                Regex::new(pattern)
            } else {
                Regex::new(&format!("(?i){}", pattern))
            }.map_err(|e| PandrsError::InvalidValue(format!("Invalid regex pattern: {}", e)))?;
            
            let replaced_values: Vec<String> = self.series.values()
                .iter()
                .map(|s| re.replace_all(s, replacement).to_string())
                .collect();
            
            Series::new(replaced_values, self.series.name().cloned())
                .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
        } else {
            let replaced_values: Vec<String> = if case {
                self.series.values()
                    .iter()
                    .map(|s| s.replace(pattern, replacement))
                    .collect()
            } else {
                // Case-insensitive replacement without regex
                self.series.values()
                    .iter()
                    .map(|s| case_insensitive_replace(s, pattern, replacement))
                    .collect()
            };
            
            Series::new(replaced_values, self.series.name().cloned())
                .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
        }
    }

    /// Split strings by delimiter
    pub fn split(&self, delimiter: &str, n: Option<usize>, expand: bool) -> Result<Series<String>, PandrsError> {
        if expand {
            // Return multiple columns (not implemented yet, return error)
            return Err(PandrsError::NotImplemented(
                "split with expand=true not yet implemented".to_string()
            ));
        }
        
        let split_values: Vec<Vec<String>> = self.series.values()
            .iter()
            .map(|s| {
                if let Some(max_splits) = n {
                    s.splitn(max_splits + 1, delimiter).map(|s| s.to_string()).collect()
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

    /// Get string length
    pub fn len(&self) -> Result<Series<i64>, PandrsError> {
        let lengths: Vec<i64> = self.series.values()
            .iter()
            .map(|s| s.len() as i64)
            .collect();
        
        Series::new(lengths, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Strip whitespace from both ends
    pub fn strip(&self, chars: Option<&str>) -> Result<Series<String>, PandrsError> {
        let stripped_values: Vec<String> = if let Some(chars_to_strip) = chars {
            self.series.values()
                .iter()
                .map(|s| strip_chars(s, chars_to_strip))
                .collect()
        } else {
            self.series.values()
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
            self.series.values()
                .iter()
                .map(|s| lstrip_chars(s, chars_to_strip))
                .collect()
        } else {
            self.series.values()
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
            self.series.values()
                .iter()
                .map(|s| rstrip_chars(s, chars_to_strip))
                .collect()
        } else {
            self.series.values()
                .iter()
                .map(|s| s.trim_end().to_string())
                .collect()
        };
        
        Series::new(stripped_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract substring using regex groups
    pub fn extract(&self, pattern: &str, flags: Option<&str>) -> Result<Series<String>, PandrsError> {
        let regex_pattern = if let Some(f) = flags {
            if f.contains('i') {
                format!("(?i){}", pattern)
            } else {
                pattern.to_string()
            }
        } else {
            pattern.to_string()
        };
        
        let re = Regex::new(&regex_pattern)
            .map_err(|e| PandrsError::InvalidValue(format!("Invalid regex pattern: {}", e)))?;
        
        let extracted_values: Vec<String> = self.series.values()
            .iter()
            .map(|s| {
                if let Some(caps) = re.captures(s) {
                    if caps.len() > 1 {
                        caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_else(|| "".to_string())
                    } else {
                        caps.get(0).map(|m| m.as_str().to_string()).unwrap_or_else(|| "".to_string())
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
    pub fn findall(&self, pattern: &str, flags: Option<&str>) -> Result<Series<String>, PandrsError> {
        let regex_pattern = if let Some(f) = flags {
            if f.contains('i') {
                format!("(?i){}", pattern)
            } else {
                pattern.to_string()
            }
        } else {
            pattern.to_string()
        };
        
        let re = Regex::new(&regex_pattern)
            .map_err(|e| PandrsError::InvalidValue(format!("Invalid regex pattern: {}", e)))?;
        
        let found_values: Vec<String> = self.series.values()
            .iter()
            .map(|s| {
                let matches: Vec<String> = re.find_iter(s)
                    .map(|m| m.as_str().to_string())
                    .collect();
                format!("[{}]", matches.join(", "))
            })
            .collect();
        
        Series::new(found_values, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Count occurrences of pattern
    pub fn count(&self, pattern: &str, flags: Option<&str>) -> Result<Series<i64>, PandrsError> {
        let regex_pattern = if let Some(f) = flags {
            if f.contains('i') {
                format!("(?i){}", pattern)
            } else {
                pattern.to_string()
            }
        } else {
            pattern.to_string()
        };
        
        let re = Regex::new(&regex_pattern)
            .map_err(|e| PandrsError::InvalidValue(format!("Invalid regex pattern: {}", e)))?;
        
        let counts: Vec<i64> = self.series.values()
            .iter()
            .map(|s| re.find_iter(s).count() as i64)
            .collect();
        
        Series::new(counts, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Pad strings to specified width
    pub fn pad(&self, width: usize, side: &str, fillchar: char) -> Result<Series<String>, PandrsError> {
        let padded_values: Vec<String> = self.series.values()
            .iter()
            .map(|s| {
                if s.len() >= width {
                    s.clone()
                } else {
                    let padding_needed = width - s.len();
                    match side {
                        "left" => format!("{}{}", fillchar.to_string().repeat(padding_needed), s),
                        "right" => format!("{}{}", s, fillchar.to_string().repeat(padding_needed)),
                        "both" => {
                            let left_pad = padding_needed / 2;
                            let right_pad = padding_needed - left_pad;
                            format!("{}{}{}", 
                                fillchar.to_string().repeat(left_pad),
                                s,
                                fillchar.to_string().repeat(right_pad))
                        }
                        _ => s.clone() // Invalid side, return original
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
                Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
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
        
        assert_eq!(values, &["HELLO".to_string(), "WORLD".to_string(), "RUST".to_string()]);
    }

    #[test]
    fn test_string_lower() {
        let data = vec!["HELLO".to_string(), "World".to_string(), "rust".to_string()];
        let series = Series::new(data, Some("test".to_string())).unwrap();
        let str_accessor = StringAccessor::new(series).unwrap();
        
        let result = str_accessor.lower().unwrap();
        let values = result.values();
        
        assert_eq!(values, &["hello".to_string(), "world".to_string(), "rust".to_string()]);
    }

    #[test]
    fn test_string_contains() {
        let data = vec!["hello world".to_string(), "rust programming".to_string(), "python data".to_string()];
        let series = Series::new(data, Some("test".to_string())).unwrap();
        let str_accessor = StringAccessor::new(series).unwrap();
        
        let result = str_accessor.contains("rust", true, false).unwrap();
        let values = result.values();
        
        assert_eq!(values, &[false, true, false]);
    }

    #[test]
    fn test_string_startswith() {
        let data = vec!["hello world".to_string(), "hello rust".to_string(), "goodbye".to_string()];
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
        let data = vec!["  hello  ".to_string(), "\tworld\n".to_string(), " rust ".to_string()];
        let series = Series::new(data, Some("test".to_string())).unwrap();
        let str_accessor = StringAccessor::new(series).unwrap();
        
        let result = str_accessor.strip(None).unwrap();
        let values = result.values();
        
        assert_eq!(values, &["hello".to_string(), "world".to_string(), "rust".to_string()]);
    }
}