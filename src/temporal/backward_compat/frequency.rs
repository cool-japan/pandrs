use chrono::Duration;
use std::fmt;

/// Enumeration representing frequency (period) of time series data
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frequency {
    /// Every second
    Secondly,
    /// Every minute
    Minutely,
    /// Every hour
    Hourly,
    /// Every day
    Daily,
    /// Every week
    Weekly,
    /// Every month
    Monthly,
    /// Every quarter (3 months)
    Quarterly,
    /// Every year
    Yearly,
    /// Custom period
    Custom(Duration),
}

impl Frequency {
    /// Parse frequency from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "S" | "SEC" | "SECOND" | "SECONDS" => Some(Frequency::Secondly),
            "T" | "MIN" | "MINUTE" | "MINUTES" => Some(Frequency::Minutely),
            "H" | "HOUR" | "HOURS" => Some(Frequency::Hourly),
            "D" | "DAY" | "DAYS" | "DAILY" => Some(Frequency::Daily),
            "W" | "WEEK" | "WEEKS" | "WEEKLY" => Some(Frequency::Weekly),
            "M" | "MONTH" | "MONTHS" | "MONTHLY" => Some(Frequency::Monthly),
            "Q" | "QUARTER" | "QUARTERS" | "QUARTERLY" => Some(Frequency::Quarterly),
            "Y" | "YEAR" | "YEARS" | "A" | "ANNUAL" | "ANNUALLY" | "YEARLY" => {
                Some(Frequency::Yearly)
            }
            _ => {
                // Try to parse a custom period
                parse_custom_frequency(s)
            }
        }
    }

    /// Get approximate number of seconds for this frequency
    /// Estimates for months and years
    pub fn to_seconds(&self) -> i64 {
        match self {
            Frequency::Secondly => 1,
            Frequency::Minutely => 60,
            Frequency::Hourly => 3600,
            Frequency::Daily => 86400,
            Frequency::Weekly => 604800,
            Frequency::Monthly => 2592000,   // Estimated as 30 days
            Frequency::Quarterly => 7776000, // Estimated as 90 days
            Frequency::Yearly => 31536000,   // Estimated as 365 days
            Frequency::Custom(duration) => duration.num_seconds(),
        }
    }
}

impl fmt::Display for Frequency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Frequency::Secondly => write!(f, "S"),
            Frequency::Minutely => write!(f, "T"),
            Frequency::Hourly => write!(f, "H"),
            Frequency::Daily => write!(f, "D"),
            Frequency::Weekly => write!(f, "W"),
            Frequency::Monthly => write!(f, "M"),
            Frequency::Quarterly => write!(f, "Q"),
            Frequency::Yearly => write!(f, "Y"),
            Frequency::Custom(duration) => write!(f, "{}s", duration.num_seconds()),
        }
    }
}

/// Parse a custom frequency string
fn parse_custom_frequency(s: &str) -> Option<Frequency> {
    // Parse formats like "3D" (3 days) or "2H" (2 hours)

    // Split into numeric and unit parts
    let mut num_chars = String::new();
    let mut unit_chars = String::new();
    let mut found_digit = false;

    for c in s.chars() {
        if c.is_digit(10) {
            found_digit = true;
            num_chars.push(c);
        } else if found_digit {
            unit_chars.push(c);
        } else {
            // Numbers must come first
            return None;
        }
    }

    if num_chars.is_empty() || unit_chars.is_empty() {
        return None;
    }

    // Parse the number
    let num: i64 = match num_chars.parse() {
        Ok(n) => n,
        Err(_) => return None,
    };

    // Parse the unit and create appropriate Duration
    match unit_chars.to_uppercase().as_str() {
        "S" | "SEC" | "SECOND" | "SECONDS" => Some(Frequency::Custom(Duration::seconds(num))),
        "T" | "MIN" | "MINUTE" | "MINUTES" => Some(Frequency::Custom(Duration::minutes(num))),
        "H" | "HOUR" | "HOURS" => Some(Frequency::Custom(Duration::hours(num))),
        "D" | "DAY" | "DAYS" => Some(Frequency::Custom(Duration::days(num))),
        "W" | "WEEK" | "WEEKS" => Some(Frequency::Custom(Duration::weeks(num))),
        _ => None,
    }
}
