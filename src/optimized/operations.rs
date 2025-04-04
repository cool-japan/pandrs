/// 集計操作を表す列挙型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOp {
    /// 合計
    Sum,
    /// 平均
    Mean,
    /// 最小値
    Min,
    /// 最大値
    Max,
    /// 件数
    Count,
}

/// 結合タイプを表す列挙型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// 内部結合（両方の表に存在する行のみ）
    Inner,
    /// 左結合（左側の表の全ての行と、それに一致する右側の表の行）
    Left,
    /// 右結合（右側の表の全ての行と、それに一致する左側の表の行）
    Right,
    /// 外部結合（両方の表の全ての行）
    Outer,
}