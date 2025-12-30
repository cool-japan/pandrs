use wasm_bindgen::prelude::*;

// Re-export all wasm-related types from pandrs
pub use pandrs::web::{WebVisualization, WebVisualizationConfig, ColorTheme, VisualizationType};

#[wasm_bindgen(start)]
pub fn wasm_init() {
    // Enable better error messages in development
    console_error_panic_hook::set_once();
}

// Library initialization
#[wasm_bindgen]
pub fn init() {
    // Any additional initialization can go here
}