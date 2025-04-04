use pandrs::optimized::dataframe::OptimizedDataFrame;
use pandrs::column::{Column, StringColumn, Int64Column};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut df = OptimizedDataFrame::new();
    
    let cities = vec\![
        "東京".to_string(), "大阪".to_string(), "名古屋".to_string()
    ];
    
    let values = vec\![100, 200, 300];
    
    df.add_column("都市", Column::String(StringColumn::new(cities)))?;
    df.add_column("値", Column::Int64(Int64Column::new(values)))?;
    
    println\!("{:?}", df);
    
    Ok(())
}
