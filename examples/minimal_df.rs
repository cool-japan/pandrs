use pandrs::optimized::dataframe::OptimizedDataFrame;
use pandrs::column::{Column, Int64Column};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut df = OptimizedDataFrame::new();
    
    // 数値データ
    let values = vec\![100, 200, 300];
    
    // 数値列を追加
    df.add_column("値", Column::Int64(Int64Column::new(values)))?;
    
    println\!("{:?}", df);
    
    Ok(())
}
