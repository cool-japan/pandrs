use chrono::Duration;
use std::fmt;

/// 時系列データの頻度（周期）を表す列挙型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frequency {
    /// 1秒ごと
    Secondly,
    /// 1分ごと
    Minutely,
    /// 1時間ごと
    Hourly,
    /// 1日ごと
    Daily,
    /// 1週間ごと
    Weekly,
    /// 1ヶ月ごと
    Monthly,
    /// 3ヶ月（四半期）ごと
    Quarterly,
    /// 1年ごと
    Yearly,
    /// カスタム周期
    Custom(Duration),
}

impl Frequency {
    /// 文字列から周波数を解析
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
                // カスタム期間の解析を試みる
                parse_custom_frequency(s)
            }
        }
    }

    /// この頻度に対応するおおよその秒数を取得
    /// 月や年などは概算値
    pub fn to_seconds(&self) -> i64 {
        match self {
            Frequency::Secondly => 1,
            Frequency::Minutely => 60,
            Frequency::Hourly => 3600,
            Frequency::Daily => 86400,
            Frequency::Weekly => 604800,
            Frequency::Monthly => 2592000,   // 30日として概算
            Frequency::Quarterly => 7776000, // 90日として概算
            Frequency::Yearly => 31536000,   // 365日として概算
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

/// カスタム周期の文字列を解析する
fn parse_custom_frequency(s: &str) -> Option<Frequency> {
    // "3D" (3日) や "2H" (2時間) のような形式を解析

    // 数値部分と単位部分に分割
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
            // 数字が先に来ないといけない
            return None;
        }
    }

    if num_chars.is_empty() || unit_chars.is_empty() {
        return None;
    }

    // 数値を解析
    let num: i64 = match num_chars.parse() {
        Ok(n) => n,
        Err(_) => return None,
    };

    // 単位を解析して適切なDurationを作成
    match unit_chars.to_uppercase().as_str() {
        "S" | "SEC" | "SECOND" | "SECONDS" => Some(Frequency::Custom(Duration::seconds(num))),
        "T" | "MIN" | "MINUTE" | "MINUTES" => Some(Frequency::Custom(Duration::minutes(num))),
        "H" | "HOUR" | "HOURS" => Some(Frequency::Custom(Duration::hours(num))),
        "D" | "DAY" | "DAYS" => Some(Frequency::Custom(Duration::days(num))),
        "W" | "WEEK" | "WEEKS" => Some(Frequency::Custom(Duration::weeks(num))),
        _ => None,
    }
}
