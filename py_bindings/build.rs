// Build script for pandrs Python bindings
// Sets cuda_available cfg flag on non-macOS when cuda feature is enabled

fn main() {
    println!("cargo:rustc-check-cfg=cfg(cuda_available)");

    #[cfg(not(target_os = "macos"))]
    {
        #[cfg(feature = "cuda")]
        {
            println!("cargo:rustc-cfg=cuda_available");
        }
    }

    // Link libpython so `cargo test` (which builds a standalone test binary,
    // not a cdylib) can resolve all PyO3 symbols.
    // On Linux, for production cdylib builds the dynamic linker deduplicates
    // libpython via SONAME — the interpreter already holds libpython3.x in
    // memory, so no double-initialization occurs.
    link_python();

    println!("cargo:rerun-if-changed=build.rs");
}

fn link_python() {
    let python = std::env::var("PYO3_PYTHON")
        .or_else(|_| std::env::var("PYTHON"))
        .unwrap_or_else(|_| "python3".to_string());

    let output = std::process::Command::new(&python)
        .args([
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ])
        .output();

    if let Ok(out) = output {
        if out.status.success() {
            let ver = String::from_utf8_lossy(&out.stdout);
            let ver = ver.trim();
            println!("cargo:rustc-link-lib=python{ver}");
        }
    }

    // Emit standard Python library search paths; cargo ignores duplicates.
    for path in &["/usr/lib/x86_64-linux-gnu", "/usr/lib", "/usr/local/lib"] {
        println!("cargo:rustc-link-search=native={path}");
    }
}
