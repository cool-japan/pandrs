//! OptimizedDataFrameのインデックス操作関連機能

use super::core::OptimizedDataFrame;
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, Index, IndexTrait};

impl OptimizedDataFrame {
    /// インデックスを設定
    ///
    /// # Arguments
    /// * `index` - 新しいインデックス
    ///
    /// # Returns
    /// * `Result<()>` - 成功した場合はOk、失敗した場合はエラー
    pub fn set_index(&mut self, index: DataFrameIndex<String>) -> Result<()> {
        // インデックス長がデータフレームの行数と一致するか確認
        if index.len() != self.row_count {
            return Err(Error::Index(format!(
                "インデックスの長さ ({}) がデータフレームの行数 ({}) と一致しません",
                index.len(),
                self.row_count
            )));
        }

        self.index = Some(index);
        Ok(())
    }

    /// インデックスを取得
    ///
    /// # Returns
    /// * `Option<&DataFrameIndex<String>>` - インデックスが存在する場合はSome、存在しない場合はNone
    pub fn get_index(&self) -> Option<&DataFrameIndex<String>> {
        self.index.as_ref()
    }

    /// デフォルトのインデックスを作成して設定
    ///
    /// # Returns
    /// * `Result<()>` - 成功した場合はOk、失敗した場合はエラー
    pub fn set_default_index(&mut self) -> Result<()> {
        if self.row_count == 0 {
            self.index = None;
            return Ok(());
        }

        let index = DataFrameIndex::<String>::default_with_len(self.row_count)?;
        self.index = Some(index);
        Ok(())
    }

    /// 列をインデックスに設定
    ///
    /// # Arguments
    /// * `column_name` - インデックスに設定する列の名前
    /// * `drop` - 元の列を削除するかどうか
    ///
    /// # Returns
    /// * `Result<()>` - 成功した場合はOk、失敗した場合はエラー
    pub fn set_index_from_column(&mut self, column_name: &str, drop: bool) -> Result<()> {
        // 列の存在確認
        let col_idx = self.column_indices.get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        
        let col = &self.columns[*col_idx];
        
        // 列の値をString型に変換してベクトルとして取得
        let values: Vec<String> = (0..self.row_count)
            .filter_map(|i| {
                match col {
                    crate::column::Column::Int64(c) => {
                        if let Ok(Some(v)) = c.get(i) {
                            Some(v.to_string())
                        } else {
                            Some("NULL".to_string())
                        }
                    },
                    crate::column::Column::Float64(c) => {
                        if let Ok(Some(v)) = c.get(i) {
                            Some(v.to_string())
                        } else {
                            Some("NULL".to_string())
                        }
                    },
                    crate::column::Column::String(c) => {
                        if let Ok(Some(v)) = c.get(i) {
                            Some(v.to_string())
                        } else {
                            Some("NULL".to_string())
                        }
                    },
                    crate::column::Column::Boolean(c) => {
                        if let Ok(Some(v)) = c.get(i) {
                            Some(v.to_string())
                        } else {
                            Some("NULL".to_string())
                        }
                    },
                }
            })
            .collect();
        
        // 重複を許可するためにマップを使わないインデックスを作成
        let name = Some(column_name.to_string());
        let index = Index::with_name(values, name)?;
        self.index = Some(DataFrameIndex::Simple(index));
        
        // 元の列を削除するかどうか
        if drop {
            // 列の削除機能を実装する必要があります
            // self.drop_column(column_name)?;
            return Err(Error::NotImplemented("列の削除機能はまだ実装されていません".to_string()));
        }
        
        Ok(())
    }

    /// インデックスを列として追加
    ///
    /// # Arguments
    /// * `name` - 新しい列の名前
    /// * `drop_index` - インデックスを削除するかどうか
    ///
    /// # Returns
    /// * `Result<()>` - 成功した場合はOk、失敗した場合はエラー
    pub fn reset_index(&mut self, name: &str, drop_index: bool) -> Result<()> {
        if let Some(ref index) = self.index {
            let values = match index {
                DataFrameIndex::Simple(idx) => {
                    idx.values().iter().map(|v| v.clone()).collect()
                },
                DataFrameIndex::Multi(midx) => {
                    // マルチインデックスの場合は、文字列として連結
                    midx.tuples().iter()
                        .map(|tuple| tuple.join(", "))
                        .collect()
                }
            };
            
            // 新しい列としてインデックスを追加
            let col = crate::column::Column::String(crate::column::StringColumn::new(values));
            self.add_column(name.to_string(), col)?;
            
            // インデックスを削除するかどうか
            if drop_index {
                self.index = None;
            }
            
            Ok(())
        } else {
            // インデックスが存在しない場合はデフォルトのインデックスを列として追加
            let values: Vec<String> = (0..self.row_count)
                .map(|i| i.to_string())
                .collect();
            
            let col = crate::column::Column::String(crate::column::StringColumn::new(values));
            self.add_column(name.to_string(), col)?;
            
            Ok(())
        }
    }

    /// インデックスを使って行を取得
    ///
    /// # Arguments
    /// * `key` - 検索するインデックスの値
    ///
    /// # Returns
    /// * `Result<Self>` - 該当する行を含む新しいDataFrame
    pub fn get_row_by_index(&self, key: &str) -> Result<Self> {
        if let Some(ref index) = self.index {
            let pos = match index {
                DataFrameIndex::Simple(idx) => idx.get_loc(&key.to_string()),
                DataFrameIndex::Multi(_) => None, // マルチインデックスは現在サポートしない
            };
            
            if let Some(row_idx) = pos {
                // 行インデックスを使用して行を抽出
                // 1行を含むDataFrameを作成
                let indices = vec![row_idx];
                self.filter_by_indices(&indices)
            } else {
                Err(Error::Index(format!(
                    "インデックス '{}' が見つかりません",
                    key
                )))
            }
        } else {
            Err(Error::Index("インデックスが設定されていません".to_string()))
        }
    }

    /// インデックスを使って行を選択
    ///
    /// # Arguments
    /// * `keys` - 選択するインデックス値のリスト
    ///
    /// # Returns
    /// * `Result<Self>` - 選択された行を含む新しいDataFrame
    pub fn select_by_index<I, S>(&self, keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if self.index.is_none() {
            return Err(Error::Index("インデックスが設定されていません".to_string()));
        }
        
        let index = self.index.as_ref().unwrap();
        
        // インデックスから行番号を取得
        let mut indices = Vec::new();
        
        for key in keys {
            let key_str = key.as_ref().to_string();
            let pos = match index {
                DataFrameIndex::Simple(idx) => idx.get_loc(&key_str),
                DataFrameIndex::Multi(_) => None, // マルチインデックスは現在サポートしない
            };
            
            if let Some(idx) = pos {
                indices.push(idx);
            } else {
                return Err(Error::Index(format!(
                    "インデックス '{}' が見つかりません",
                    key.as_ref()
                )));
            }
        }
        
        // 行インデックスを使用して行を選択
        self.filter_by_indices(&indices)
    }
}
