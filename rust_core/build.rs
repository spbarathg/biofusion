fn main() {
    // Tell Cargo to build a dynamic library that can be used as a C library
    println!("cargo:rustc-link-lib=dylib=ant_bot_core");
    
    // Make sure to rebuild if these files change
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/worker_ant.rs");
    println!("cargo:rerun-if-changed=src/dex_client.rs");
    println!("cargo:rerun-if-changed=src/tx_executor.rs");
    println!("cargo:rerun-if-changed=src/pathfinder.rs");
    println!("cargo:rerun-if-changed=src/config.rs");
    
    // Any C dependencies would be linked here
    // println!("cargo:rustc-link-lib=solana_client");
} 