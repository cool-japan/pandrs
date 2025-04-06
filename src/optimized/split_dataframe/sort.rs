use std::cmp::Ordering;
use std::collections::HashMap;
use rayon::prelude::*;

use crate::column::{Column, ColumnTrait, ColumnType};
use crate::error::{Error, Result};
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

impl OptimizedDataFrame {
    /// �U�_gDataFrame����W~Y
    ///
    /// # Arguments
    /// * `by` - ��ȭ�hj�
    /// * `ascending` - k���Y�KiFK
    ///
    /// # Returns
    /// * `Result<Self>` - ���P�n�WDDataFrame
    pub fn sort_by(&self, by: &str, ascending: bool) -> Result<Self> {
        // �aLX(Y�K��
        let column_idx = self.column_indices.get(by)
            .ok_or_else(|| Error::ColumnNotFound(by.to_string()))?;
        
        let column = &self.columns[*column_idx];
        
        // L���ï�nٯ��\0K�Lp-1~gn#j	
        let mut indices: Vec<usize> = (0..self.row_count()).collect();
        
        // n�k�X_���
        match column.column_type() {
            ColumnType::Int64 => {
                let col = column.as_int64().unwrap();
                // ���hCn���ï�nڢ�\
                let mut pairs: Vec<(usize, Option<i64>)> = indices.iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten()))
                    .collect();
                
                // ���: NULLo8k �kMn
                pairs.sort_by(|a, b| {
                    match (&a.1, &b.1) {
                        (None, None) => Ordering::Equal,
                        (None, _) => Ordering::Greater,
                        (_, None) => Ordering::Less,
                        (Some(val_a), Some(val_b)) => {
                            if ascending {
                                val_a.cmp(val_b)
                            } else {
                                val_b.cmp(val_a)
                            }
                        }
                    }
                });
                
                // ���U�_���ï��֗
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            },
            ColumnType::Float64 => {
                let col = column.as_float64().unwrap();
                // ���hCn���ï�nڢ�\
                let mut pairs: Vec<(usize, Option<f64>)> = indices.iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten()))
                    .collect();
                
                // ���: NULLo8k �kMn
                pairs.sort_by(|a, b| {
                    match (&a.1, &b.1) {
                        (None, None) => Ordering::Equal,
                        (None, _) => Ordering::Greater,
                        (_, None) => Ordering::Less,
                        (Some(val_a), Some(val_b)) => {
                            if ascending {
                                val_a.partial_cmp(val_b).unwrap_or(Ordering::Equal)
                            } else {
                                val_b.partial_cmp(val_a).unwrap_or(Ordering::Equal)
                            }
                        }
                    }
                });
                
                // ���U�_���ï��֗
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            },
            ColumnType::String => {
                let col = column.as_string().unwrap();
                // ���hCn���ï�nڢ�\
                let mut pairs: Vec<(usize, Option<String>)> = indices.iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten().map(|s| s.to_string())))
                    .collect();
                
                // ���: NULLo8k �kMn
                pairs.sort_by(|a, b| {
                    match (&a.1, &b.1) {
                        (None, None) => Ordering::Equal,
                        (None, _) => Ordering::Greater,
                        (_, None) => Ordering::Less,
                        (Some(val_a), Some(val_b)) => {
                            if ascending {
                                val_a.cmp(val_b)
                            } else {
                                val_b.cmp(val_a)
                            }
                        }
                    }
                });
                
                // ���U�_���ï��֗
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            },
            ColumnType::Boolean => {
                let col = column.as_boolean().unwrap();
                // ���hCn���ï�nڢ�\
                let mut pairs: Vec<(usize, Option<bool>)> = indices.iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten()))
                    .collect();
                
                // ���: NULLo8k �kMn
                pairs.sort_by(|a, b| {
                    match (&a.1, &b.1) {
                        (None, None) => Ordering::Equal,
                        (None, _) => Ordering::Greater,
                        (_, None) => Ordering::Less,
                        (Some(val_a), Some(val_b)) => {
                            if ascending {
                                val_a.cmp(val_b)
                            } else {
                                val_b.cmp(val_a)
                            }
                        }
                    }
                });
                
                // ���U�_���ï��֗
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            },
        }
        
        // ���U�_L���ï��(Wf�WDDataFrame�\
        self.select_rows_by_indices(&indices)
    }
    
    /// pngDataFrame����W~Y
    ///
    /// # Arguments
    /// * `by` - ��ȭ�hj�nM
    /// * `ascending` - ��k��Y�/M��nM�WjD4oYyf	
    ///
    /// # Returns
    /// * `Result<Self>` - ���P�n�WDDataFrame
    pub fn sort_by_columns(&self, by: &[&str], ascending: Option<&[bool]>) -> Result<Self> {
        if by.is_empty() {
            return Err(Error::EmptyColumnList);
        }
        
        // YyfnLX(Y�K��
        for &col_name in by {
            if !self.column_indices.contains_key(col_name) {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }
        }
        
        // /M�鰒��
        let is_ascending: Vec<bool> = match ascending {
            Some(asc) => {
                if asc.len() != by.len() {
                    return Err(Error::InconsistentArrayLengths {
                        expected: by.len(),
                        found: asc.len(),
                    });
                }
                asc.to_vec()
            },
            None => vec![true; by.len()], // �թ��oYyf
        };
        
        // L���ï�nٯ��\0K�Lp-1~gn#j	
        let mut indices: Vec<usize> = (0..self.row_count()).collect();
        
        // pk�eO��������	
        indices.sort_by(|&a, &b| {
            // �U�_�k�
            for (col_idx, (&col_name, &asc)) in by.iter().zip(is_ascending.iter()).enumerate() {
                let column_idx = self.column_indices[col_name];
                let column = &self.columns[column_idx];
                
                let cmp = match column.column_type() {
                    ColumnType::Int64 => {
                        let col = column.as_int64().unwrap();
                        let val_a = col.get(a).ok().flatten();
                        let val_b = col.get(b).ok().flatten();
                        
                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc { v_a.cmp(&v_b) } else { v_b.cmp(&v_a) }
                            }
                        }
                    },
                    ColumnType::Float64 => {
                        let col = column.as_float64().unwrap();
                        let val_a = col.get(a).ok().flatten();
                        let val_b = col.get(b).ok().flatten();
                        
                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc { 
                                    v_a.partial_cmp(&v_b).unwrap_or(Ordering::Equal) 
                                } else { 
                                    v_b.partial_cmp(&v_a).unwrap_or(Ordering::Equal) 
                                }
                            }
                        }
                    },
                    ColumnType::String => {
                        let col = column.as_string().unwrap();
                        let val_a = col.get(a).ok().flatten().map(|s| s.to_string());
                        let val_b = col.get(b).ok().flatten().map(|s| s.to_string());
                        
                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc { v_a.cmp(&v_b) } else { v_b.cmp(&v_a) }
                            }
                        }
                    },
                    ColumnType::Boolean => {
                        let col = column.as_boolean().unwrap();
                        let val_a = col.get(a).ok().flatten();
                        let val_b = col.get(b).ok().flatten();
                        
                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc { v_a.cmp(&v_b) } else { v_b.cmp(&v_a) }
                            }
                        }
                    },
                };
                
                // Sng�Lz~�p]���Y
                if cmp != Ordering::Equal {
                    return cmp;
                }
                
                // $n4o!n��
            }
            
            // Yyfng$n4oCn���
            Ordering::Equal
        });
        
        // ���U�_L���ï��(Wf�WDDataFrame�\
        self.select_rows_by_indices(&indices)
    }
    
    /// �U�_L���ï��(WfL�x�
    fn select_rows_by_indices(&self, indices: &[usize]) -> Result<Self> {
        // LpL0n4ozn���������Y
        if indices.is_empty() {
            return Ok(Self::new());
        }
        
        let mut result = Self::new();
        
        // k�Wf�
        for (name, &column_idx) in &self.column_indices {
            let column = &self.columns[column_idx];
            
            // n�k�Xf���ï��i(
            let selected_col = match column {
                Column::Int64(col) => {
                    let selected_data: Vec<i64> = indices.iter()
                        .map(|&idx| col.get(idx).ok().flatten().unwrap_or_default())
                        .collect();
                    Column::Int64(crate::column::Int64Column::new(selected_data))
                },
                Column::Float64(col) => {
                    let selected_data: Vec<f64> = indices.iter()
                        .map(|&idx| col.get(idx).ok().flatten().unwrap_or_default())
                        .collect();
                    Column::Float64(crate::column::Float64Column::new(selected_data))
                },
                Column::String(col) => {
                    let selected_data: Vec<String> = indices.iter()
                        .map(|&idx| col.get(idx).ok().flatten().map(|s| s.to_string()).unwrap_or_default())
                        .collect();
                    Column::String(crate::column::StringColumn::new(selected_data))
                },
                Column::Boolean(col) => {
                    let selected_data: Vec<bool> = indices.iter()
                        .map(|&idx| col.get(idx).ok().flatten().unwrap_or_default())
                        .collect();
                    Column::Boolean(crate::column::BooleanColumn::new(selected_data))
                },
            };
            
            result.add_column(name.clone(), selected_col)?;
        }
        
        // ���ï�LB�pi(
        if let Some(ref idx) = self.index {
            // TODO: ���ï�n��Ȃ��
            result.index = Some(idx.clone());
        }
        
        Ok(result)
    }
}