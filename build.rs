// Build script for PandRS
// Handles conditional compilation for optional features like CUDA

fn main() {
    // Register cuda_available as a valid cfg for check-cfg
    // Using cargo: syntax for MSRV 1.70.0 compatibility
    println!("cargo:rustc-check-cfg=cfg(cuda_available)");

    // Emit cfg flag for CUDA availability
    // CUDA is only available on non-macOS platforms
    #[cfg(not(target_os = "macos"))]
    {
        // On non-macOS, emit cuda_available cfg if cuda feature is enabled
        #[cfg(feature = "cuda")]
        {
            println!("cargo:rustc-cfg=cuda_available");
            println!(
                "cargo:warning=CUDA feature enabled - this requires NVIDIA CUDA toolkit installation"
            );

            // Check if CUDA is available and handle gracefully
            if std::env::var("CUDA_ROOT").is_err() {
                // Try common CUDA installation paths
                let cuda_paths = [
                    "/usr/local/cuda",
                    "/opt/cuda",
                    "/usr/lib/cuda",
                    "/usr/local/cuda-11.0",
                    "/usr/local/cuda-11.1",
                    "/usr/local/cuda-11.2",
                    "/usr/local/cuda-11.3",
                    "/usr/local/cuda-11.4",
                    "/usr/local/cuda-11.5",
                    "/usr/local/cuda-11.6",
                    "/usr/local/cuda-11.7",
                    "/usr/local/cuda-11.8",
                    "/usr/local/cuda-12.0",
                    "/usr/local/cuda-12.1",
                    "/usr/local/cuda-12.2",
                    "/usr/local/cuda-12.3",
                    "/usr/local/cuda-12.4",
                    "/usr/local/cuda-12.5",
                ];

                let mut found = false;
                for path in &cuda_paths {
                    let cuda_h_path = format!("{}/include/cuda.h", path);
                    if std::path::Path::new(&cuda_h_path).exists() {
                        println!("cargo:rustc-env=CUDA_ROOT={}", path);
                        println!("cargo:warning=Found CUDA at {}", path);
                        found = true;
                        break;
                    }
                }

                if !found {
                    println!("cargo:warning=CUDA toolkit not found. Please install NVIDIA CUDA toolkit or disable 'cuda' feature");
                    println!("cargo:warning=Use: cargo build --features \"all-safe\" instead of --all-features");
                }
            }

            // Tell cargo to rerun this build script if CUDA environment changes
            println!("cargo:rerun-if-env-changed=CUDA_ROOT");
            println!("cargo:rerun-if-env-changed=CUDA_PATH");
            println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");
        }
    }

    // On macOS, cuda feature is ignored (CUDA not supported)
    #[cfg(target_os = "macos")]
    {
        #[cfg(feature = "cuda")]
        {
            println!("cargo:warning=CUDA feature requested but macOS does not support CUDA - feature ignored");
        }
    }

    // Handle other conditional compilation requirements
    println!("cargo:rerun-if-changed=build.rs");
}
