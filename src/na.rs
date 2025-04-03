use std::cmp::Ordering;
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};

/// 欠損値（NA, Not Available）を表現する型
///
/// Rustでは欠損値を型システムで表現するため、OptionではなくNA型を定義します。
/// NAは値が存在しないことを表します。
#[derive(Clone, Copy)]
pub enum NA<T> {
    /// 値が存在する場合
    Value(T),
    /// 値が存在しない場合
    NA,
}

impl<T> NA<T> {
    /// 値があるかどうかをチェック
    pub fn is_na(&self) -> bool {
        match self {
            NA::Value(_) => false,
            NA::NA => true,
        }
    }

    /// 値があるかどうかをチェック
    pub fn is_value(&self) -> bool {
        !self.is_na()
    }

    /// 値を取得（存在する場合）
    pub fn value(&self) -> Option<&T> {
        match self {
            NA::Value(v) => Some(v),
            NA::NA => None,
        }
    }

    /// 値を取得（存在する場合）、存在しない場合はデフォルト値を返す
    pub fn value_or<'a>(&'a self, default: &'a T) -> &'a T {
        match self {
            NA::Value(v) => v,
            NA::NA => default,
        }
    }

    /// 値を変換する
    pub fn map<U, F>(&self, f: F) -> NA<U>
    where
        F: FnOnce(&T) -> U,
    {
        match self {
            NA::Value(v) => NA::Value(f(v)),
            NA::NA => NA::NA,
        }
    }
}

// From実装：T型からNA<T>への自動変換
impl<T> From<T> for NA<T> {
    fn from(value: T) -> Self {
        NA::Value(value)
    }
}

// From実装：Option<T>からNA<T>への自動変換
impl<T> From<Option<T>> for NA<T> {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => NA::Value(v),
            None => NA::NA,
        }
    }
}

// Into実装：NA<T>からOption<T>への自動変換
impl<T> From<NA<T>> for Option<T> {
    fn from(na: NA<T>) -> Self {
        match na {
            NA::Value(v) => Some(v),
            NA::NA => None,
        }
    }
}

// Debug実装
impl<T: Debug> Debug for NA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NA::Value(v) => write!(f, "{:?}", v),
            NA::NA => write!(f, "NA"),
        }
    }
}

// Display実装
impl<T: Display> Display for NA<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NA::Value(v) => write!(f, "{}", v),
            NA::NA => write!(f, "NA"),
        }
    }
}

// PartialEq実装
impl<T: PartialEq> PartialEq for NA<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => a == b,
            (NA::NA, NA::NA) => true,
            _ => false,
        }
    }
}

// Eq実装（T: Eqの場合）
impl<T: Eq> Eq for NA<T> {}

// PartialOrd実装
impl<T: PartialOrd> PartialOrd for NA<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => a.partial_cmp(b),
            (NA::NA, NA::NA) => Some(Ordering::Equal),
            (NA::NA, _) => Some(Ordering::Less), // NAは常に他の値より小さいと定義
            (_, NA::NA) => Some(Ordering::Greater),
        }
    }
}

// Ord実装（T: Ordの場合）
impl<T: Ord> Ord for NA<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => a.cmp(b),
            (NA::NA, NA::NA) => Ordering::Equal,
            (NA::NA, _) => Ordering::Less,
            (_, NA::NA) => Ordering::Greater,
        }
    }
}

// Hash実装
impl<T: Hash> Hash for NA<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            NA::Value(v) => {
                0.hash(state); // タグ値
                v.hash(state);
            }
            NA::NA => {
                1.hash(state); // タグ値
            }
        }
    }
}

// 数値演算の実装（Add）
impl<T: Add<Output = T>> Add for NA<T> {
    type Output = NA<T>;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => NA::Value(a + b),
            _ => NA::NA, // どちらかがNAならNAを返す
        }
    }
}

// 数値演算の実装（Sub）
impl<T: Sub<Output = T>> Sub for NA<T> {
    type Output = NA<T>;

    fn sub(self, other: Self) -> Self::Output {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => NA::Value(a - b),
            _ => NA::NA,
        }
    }
}

// 数値演算の実装（Mul）
impl<T: Mul<Output = T>> Mul for NA<T> {
    type Output = NA<T>;

    fn mul(self, other: Self) -> Self::Output {
        match (self, other) {
            (NA::Value(a), NA::Value(b)) => NA::Value(a * b),
            _ => NA::NA,
        }
    }
}

// 数値演算の実装（Div）
impl<T: Div<Output = T> + std::cmp::PartialEq + NumericCast> Div for NA<T> {
    type Output = NA<T>;

    fn div(self, other: Self) -> Self::Output {
        match (self, other) {
            (NA::Value(_), NA::Value(b)) if b == T::from(0) => NA::NA, // ゼロ除算はNA
            (NA::Value(a), NA::Value(b)) => NA::Value(a / b),
            _ => NA::NA,
        }
    }
}

// 型変換ヘルパー関数（FromをT::from(0)で使うため）
trait NumericCast {
    fn from(val: i32) -> Self;
}

// i32, f64などの基本型に対して実装
macro_rules! impl_numeric_cast {
    ($($t:ty),*) => {
        $(
            impl NumericCast for $t {
                fn from(val: i32) -> Self {
                    val as $t
                }
            }
        )*
    };
}

// 数値型に対して実装
impl_numeric_cast!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64);
