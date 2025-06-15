//! Jupyter Integration Example for PandRS Alpha 4
//!
//! This example demonstrates the comprehensive Jupyter notebook integration including:
//! - Rich HTML display for DataFrames and Series
//! - Interactive widgets and visualizations  
//! - Custom display formatting and styling
//! - Configuration management
//! - Data explorer and summary views

use pandrs::core::error::Result;
use pandrs::dataframe::DataFrame;
use pandrs::jupyter::{
    init_jupyter, jupyter_dark_mode, jupyter_light_mode, set_jupyter_config, JupyterColorScheme,
    JupyterConfig, JupyterDisplay, JupyterMagics, TableStyle, TableWidth,
};
use pandrs::series::Series;
use std::collections::HashMap;
use std::fs;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("📔 Jupyter Integration Example for PandRS Alpha 4");
    println!("=================================================\n");

    // 1. Initialize Jupyter Integration
    println!("1️⃣ Initializing Jupyter Integration");
    println!("------------------------------------");

    init_jupyter()?;

    // Create sample data for demonstration
    let df = create_sample_data()?;
    let series = create_sample_series()?;

    println!(
        "📊 Created sample DataFrame: {} rows × {} columns",
        df.row_count(),
        df.column_names().len()
    );
    println!("📈 Created sample Series: {} values\n", series.len());

    // 2. Basic HTML Display
    println!("2️⃣ Basic HTML Display");
    println!("---------------------");

    let default_config = JupyterConfig::default();

    // Generate DataFrame HTML
    let df_html = df.to_jupyter_html(&default_config)?;
    fs::write("jupyter_output/dataframe_display.html", &df_html)?;
    println!("✅ Generated DataFrame HTML: jupyter_output/dataframe_display.html");

    // Generate Series HTML
    let series_html = series.to_jupyter_html(&default_config)?;
    fs::write("jupyter_output/series_display.html", &series_html)?;
    println!("✅ Generated Series HTML: jupyter_output/series_display.html");

    // 3. Custom Styling and Themes
    println!("\n3️⃣ Custom Styling and Themes");
    println!("-----------------------------");

    // Dark mode configuration
    let dark_config = jupyter_dark_mode();
    let dark_html = df.to_jupyter_html(&dark_config)?;
    fs::write("jupyter_output/dataframe_dark.html", &dark_html)?;
    println!("✅ Generated dark theme HTML: jupyter_output/dataframe_dark.html");

    // Light mode configuration
    let light_config = jupyter_light_mode();
    let light_html = df.to_jupyter_html(&light_config)?;
    fs::write("jupyter_output/dataframe_light.html", &light_html)?;
    println!("✅ Generated light theme HTML: jupyter_output/dataframe_light.html");

    // Custom color scheme
    let custom_config = JupyterConfig {
        color_scheme: JupyterColorScheme::Custom {
            header_bg: "#4a90e2".to_string(),
            header_text: "#ffffff".to_string(),
            row_bg: "#f8f9fa".to_string(),
            alt_row_bg: "#e9ecef".to_string(),
            text_color: "#333333".to_string(),
            border_color: "#dee2e6".to_string(),
        },
        table_style: TableStyle {
            width: TableWidth::Percentage(95),
            cell_padding: "10px 15px".to_string(),
            font_family: "'Roboto', sans-serif".to_string(),
            show_borders: true,
            show_index: true,
            show_dtypes: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let custom_html = df.to_jupyter_html(&custom_config)?;
    fs::write("jupyter_output/dataframe_custom.html", &custom_html)?;
    println!("✅ Generated custom styled HTML: jupyter_output/dataframe_custom.html");

    // 4. Interactive Widgets
    println!("\n4️⃣ Interactive Widgets");
    println!("----------------------");

    let interactive_widget = df.to_interactive_widget(&default_config)?;
    fs::write("jupyter_output/dataframe_widget.html", &interactive_widget)?;
    println!("✅ Generated interactive widget: jupyter_output/dataframe_widget.html");

    let series_widget = series.to_interactive_widget(&default_config)?;
    fs::write("jupyter_output/series_widget.html", &series_widget)?;
    println!("✅ Generated Series widget: jupyter_output/series_widget.html");

    // 5. Summary Views
    println!("\n5️⃣ Summary Views");
    println!("----------------");

    let df_summary = df.to_summary_html(&default_config)?;
    fs::write("jupyter_output/dataframe_summary.html", &df_summary)?;
    println!("✅ Generated DataFrame summary: jupyter_output/dataframe_summary.html");

    let series_summary = series.to_summary_html(&default_config)?;
    fs::write("jupyter_output/series_summary.html", &series_summary)?;
    println!("✅ Generated Series summary: jupyter_output/series_summary.html");

    // 6. Data Explorer
    println!("\n6️⃣ Data Explorer");
    println!("----------------");

    let data_explorer = df.to_data_explorer(&default_config)?;
    fs::write("jupyter_output/data_explorer.html", &data_explorer)?;
    println!("✅ Generated data explorer: jupyter_output/data_explorer.html");

    // 7. Configuration Management
    println!("\n7️⃣ Configuration Management");
    println!("---------------------------");

    // Set global configuration
    let global_config = JupyterConfig {
        max_rows: 50,
        max_columns: 15,
        precision: 3,
        interactive: true,
        syntax_highlighting: true,
        table_style: TableStyle {
            show_summary: true,
            show_dtypes: true,
            font_size: "14px".to_string(),
            ..Default::default()
        },
        color_scheme: JupyterColorScheme::Dark,
    };

    set_jupyter_config(global_config.clone());
    println!("✅ Set global Jupyter configuration");
    println!("   📊 Max rows: {}", global_config.max_rows);
    println!("   📋 Max columns: {}", global_config.max_columns);
    println!("   🎨 Color scheme: {:?}", global_config.color_scheme);

    // Generate output with global config
    let global_html = df.to_jupyter_html(&global_config)?;
    fs::write("jupyter_output/dataframe_global_config.html", &global_html)?;
    println!("✅ Generated HTML with global config: jupyter_output/dataframe_global_config.html");

    // 8. Magic Commands Support
    println!("\n8️⃣ Magic Commands Support");
    println!("-------------------------");

    let magic_commands = JupyterMagics::register_magics();
    fs::write("jupyter_output/pandrs_magics.py", &magic_commands)?;
    println!("✅ Generated magic commands script: jupyter_output/pandrs_magics.py");
    println!("   📝 Available magics:");
    println!("      %pandrs_info - Display PandRS version info");
    println!("      %pandrs_config - Configure display settings");
    println!("      %%pandrs_sql - Execute SQL-like queries");

    // 9. Advanced Display Features
    println!("\n9️⃣ Advanced Display Features");
    println!("----------------------------");

    // Large DataFrame with truncation
    let large_df = create_large_sample_data()?;
    println!(
        "📊 Created large DataFrame: {} rows × {} columns",
        large_df.row_count(),
        large_df.column_names().len()
    );

    let truncated_config = JupyterConfig {
        max_rows: 20,
        max_columns: 8,
        table_style: TableStyle {
            show_summary: true,
            show_dtypes: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let large_html = large_df.to_jupyter_html(&truncated_config)?;
    fs::write("jupyter_output/large_dataframe.html", &large_html)?;
    println!("✅ Generated truncated large DataFrame: jupyter_output/large_dataframe.html");

    // 10. Complete Jupyter Notebook Template
    println!("\n🔟 Complete Jupyter Notebook Template");
    println!("--------------------------------------");

    let notebook_template = create_jupyter_notebook_template(&df, &series)?;
    fs::write("jupyter_output/pandrs_demo.html", &notebook_template)?;
    println!("✅ Generated complete demo notebook: jupyter_output/pandrs_demo.html");

    println!("\n🎉 Jupyter Integration Example Completed!");
    println!("📁 All outputs saved to 'jupyter_output/' directory");
    println!("🌐 Open the HTML files in a web browser to see the rich displays");
    println!("📔 Copy pandrs_magics.py to your Jupyter environment to enable magic commands");

    Ok(())
}

#[allow(clippy::result_large_err)]
fn create_sample_data() -> Result<DataFrame> {
    let mut data = HashMap::new();

    data.insert(
        "product_id".to_string(),
        vec![
            "P001".to_string(),
            "P002".to_string(),
            "P003".to_string(),
            "P004".to_string(),
            "P005".to_string(),
        ],
    );

    data.insert(
        "name".to_string(),
        vec![
            "Laptop".to_string(),
            "Mouse".to_string(),
            "Keyboard".to_string(),
            "Monitor".to_string(),
            "Speakers".to_string(),
        ],
    );

    data.insert(
        "price".to_string(),
        vec![
            "1299.99".to_string(),
            "29.99".to_string(),
            "79.99".to_string(),
            "449.99".to_string(),
            "129.99".to_string(),
        ],
    );

    data.insert(
        "category".to_string(),
        vec![
            "Computers".to_string(),
            "Accessories".to_string(),
            "Accessories".to_string(),
            "Displays".to_string(),
            "Audio".to_string(),
        ],
    );

    data.insert(
        "in_stock".to_string(),
        vec![
            "true".to_string(),
            "true".to_string(),
            "false".to_string(),
            "true".to_string(),
            "true".to_string(),
        ],
    );

    DataFrame::from_map(data, None)
}

#[allow(clippy::result_large_err)]
fn create_sample_series() -> Result<Series<i32>> {
    let data = vec![10, 25, 15, 30, 20, 35, 18, 28];
    Series::new(data, Some("sales_count".to_string()))
}

#[allow(clippy::result_large_err)]
fn create_large_sample_data() -> Result<DataFrame> {
    let mut data = HashMap::new();

    // Generate larger dataset for truncation demonstration
    let size = 100;

    let ids: Vec<String> = (1..=size).map(|i| format!("ID{:03}", i)).collect();
    let values: Vec<String> = (1..=size).map(|i| (i as f64 * 1.5).to_string()).collect();
    let categories: Vec<String> = (1..=size).map(|i| format!("Cat{}", i % 5 + 1)).collect();
    let flags: Vec<String> = (1..=size).map(|i| (i % 2 == 0).to_string()).collect();
    let scores: Vec<String> = (1..=size)
        .map(|i| (i as f64 * 0.8 + 10.0).to_string())
        .collect();
    let status: Vec<String> = (1..=size)
        .map(|i| {
            if i % 3 == 0 {
                "Active"
            } else if i % 3 == 1 {
                "Pending"
            } else {
                "Inactive"
            }
            .to_string()
        })
        .collect();

    data.insert("id".to_string(), ids);
    data.insert("value".to_string(), values);
    data.insert("category".to_string(), categories);
    data.insert("flag".to_string(), flags);
    data.insert("score".to_string(), scores);
    data.insert("status".to_string(), status);
    data.insert(
        "extra_col1".to_string(),
        (1..=size).map(|i| format!("Extra{}", i)).collect(),
    );
    data.insert(
        "extra_col2".to_string(),
        (1..=size).map(|i| format!("Data{}", i)).collect(),
    );
    data.insert(
        "extra_col3".to_string(),
        (1..=size).map(|i| (i as f64).to_string()).collect(),
    );
    data.insert(
        "extra_col4".to_string(),
        (1..=size).map(|i| format!("Info{}", i)).collect(),
    );
    data.insert(
        "extra_col5".to_string(),
        (1..=size).map(|i| format!("Field{}", i)).collect(),
    );

    DataFrame::from_map(data, None)
}

#[allow(clippy::result_large_err)]
fn create_jupyter_notebook_template(df: &DataFrame, series: &Series<i32>) -> Result<String> {
    let config = JupyterConfig::default();

    let template = format!(
        r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PandRS Jupyter Integration Demo</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .notebook-cell {{
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 15px 0;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .cell-header {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
            padding: 8px 12px;
            background-color: #e9ecef;
            border-radius: 4px;
            font-size: 14px;
        }}
        .code-cell {{
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 13px;
            overflow-x: auto;
        }}
        h1, h2, h3 {{
            color: #343a40;
        }}
        .demo-title {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
    <div class="demo-title">
        <h1>🚀 PandRS Jupyter Integration Demo</h1>
        <p>Comprehensive DataFrame and Series display in Jupyter notebooks</p>
    </div>

    <div class="notebook-cell">
        <div class="cell-header">📋 Cell 1: Import and Setup</div>
        <div class="code-cell">
use pandrs::{{DataFrame, Series, JupyterDisplay, init_jupyter}};<br>
init_jupyter().unwrap();<br>
println!("PandRS Jupyter integration initialized!");
        </div>
        <p>✅ <strong>Output:</strong> PandRS Jupyter integration initialized!</p>
    </div>

    <div class="notebook-cell">
        <div class="cell-header">📊 Cell 2: DataFrame Display</div>
        <div class="code-cell">
// Sample DataFrame with product data<br>
let df = create_sample_dataframe();<br>
df.to_jupyter_html(&JupyterConfig::default())
        </div>
        <div style="margin-top: 15px;">
            {}
        </div>
    </div>

    <div class="notebook-cell">
        <div class="cell-header">📈 Cell 3: Series Display</div>
        <div class="code-cell">
// Sample Series with sales data<br>
let series = create_sample_series();<br>
series.to_jupyter_html(&JupyterConfig::default())
        </div>
        <div style="margin-top: 15px;">
            {}
        </div>
    </div>

    <div class="notebook-cell">
        <div class="cell-header">🎨 Cell 4: Custom Styling</div>
        <div class="code-cell">
// Apply dark theme styling<br>
let dark_config = jupyter_dark_mode();<br>
df.to_jupyter_html(&dark_config)
        </div>
        <div style="margin-top: 15px;">
            {}
        </div>
    </div>

    <div class="notebook-cell">
        <div class="cell-header">🔍 Cell 5: Data Explorer Widget</div>
        <div class="code-cell">
// Interactive data explorer<br>
df.to_data_explorer(&JupyterConfig::default())
        </div>
        <div style="margin-top: 15px;">
            {}
        </div>
    </div>

    <div class="notebook-cell">
        <div class="cell-header">📝 Cell 6: Summary Information</div>
        <div class="code-cell">
// DataFrame summary and statistics<br>
df.to_summary_html(&JupyterConfig::default())
        </div>
        <div style="margin-top: 15px;">
            {}
        </div>
    </div>

    <div class="notebook-cell">
        <div class="cell-header">⚙️ Cell 7: Configuration Options</div>
        <div class="code-cell">
// Custom configuration example<br>
let config = JupyterConfig {{<br>
&nbsp;&nbsp;&nbsp;&nbsp;max_rows: 50,<br>
&nbsp;&nbsp;&nbsp;&nbsp;max_columns: 10,<br>
&nbsp;&nbsp;&nbsp;&nbsp;precision: 3,<br>
&nbsp;&nbsp;&nbsp;&nbsp;interactive: true,<br>
&nbsp;&nbsp;&nbsp;&nbsp;color_scheme: JupyterColorScheme::Light,<br>
&nbsp;&nbsp;&nbsp;&nbsp;..Default::default()<br>
}};<br>
set_jupyter_config(config);
        </div>
        <p>✅ <strong>Output:</strong> Jupyter configuration updated successfully!</p>
    </div>

    <footer style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
        <p>🔬 <strong>PandRS Alpha 4</strong> - Enhanced DataFrame Library for Rust</p>
        <p>📔 Rich Jupyter integration with interactive widgets and customizable displays</p>
    </footer>
</body>
</html>
    "#,
        df.to_jupyter_html(&config)?,
        series.to_jupyter_html(&config)?,
        df.to_jupyter_html(&jupyter_dark_mode())?,
        df.to_data_explorer(&config)?,
        df.to_summary_html(&config)?
    );

    Ok(template)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jupyter_integration_example() {
        // Create output directory
        std::fs::create_dir_all("jupyter_output").unwrap();

        let result = main();
        assert!(
            result.is_ok(),
            "Jupyter integration example should complete successfully"
        );
    }

    #[test]
    fn test_sample_data_creation() {
        let df = create_sample_data();
        assert!(df.is_ok(), "Sample data creation should succeed");

        let df = df.unwrap();
        assert!(df.row_count() > 0, "Sample data should have rows");
        assert!(
            df.column_names().len() >= 5,
            "Sample data should have at least 5 columns"
        );

        let series = create_sample_series();
        assert!(series.is_ok(), "Sample series creation should succeed");
    }

    #[test]
    fn test_jupyter_html_generation() {
        let df = create_sample_data().unwrap();
        let config = JupyterConfig::default();

        let html = df.to_jupyter_html(&config);
        assert!(html.is_ok(), "HTML generation should succeed");

        let html_content = html.unwrap();
        assert!(
            html_content.contains("pandrs-table"),
            "HTML should contain table class"
        );
        assert!(
            html_content.contains("product_id"),
            "HTML should contain column names"
        );
    }

    #[test]
    fn test_jupyter_config_modes() {
        let dark_config = jupyter_dark_mode();
        assert!(matches!(dark_config.color_scheme, JupyterColorScheme::Dark));

        let light_config = jupyter_light_mode();
        assert!(matches!(
            light_config.color_scheme,
            JupyterColorScheme::Light
        ));
    }
}
