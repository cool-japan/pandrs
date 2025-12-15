// Build script for pandrs Python bindings
// Sets cuda_available cfg flag on non-macOS when cuda feature is enabled

fn main() {
    // Emit cfg flag for CUDA availability
    // CUDA is only available on non-macOS platforms
    #[cfg(not(target_os = "macos"))]
    {
        #[cfg(feature = "cuda")]
        {
            println!("cargo:rustc-cfg=cuda_available");
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}
