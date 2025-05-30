use std::path::Path;
use anyhow::Result;
use wasmtime::{Engine, Linker, Module, Store};
use wasmtime_wasi::{preview1::WasiP1Ctx, DirPerms, FilePerms, WasiCtxBuilder};
use wasmtime_wasi_nn::{backend::openvino::OpenvinoBackend, witx::WasiNnCtx, Backend, InMemoryRegistry};

fn main() -> Result<()> {
    // Test the tensor size issue directly
    println!("Testing tensor dimensions...");
    
    // Create a simple i32 tensor
    let tokens: Vec<i32> = vec![1, 2, 3, 4, 5];
    let mut padded = tokens.clone();
    padded.resize(64, 0);
    
    let bytes: Vec<u8> = padded.iter()
        .flat_map(|&i| i.to_le_bytes())
        .collect();
    
    println!("Tokens: {} i32s", padded.len());
    println!("Bytes: {} bytes", bytes.len());
    println!("Expected for [1,64] i32 tensor: {} bytes", 1 * 64 * 4);
    
    // The error suggests 256 vs 64 mismatch
    // 256 bytes = 64 i32s (what we have)
    // 64 bytes = 16 i32s OR 64 i8s
    
    println!("\nPossible interpretations:");
    println!("- If expecting 64 bytes for i32s: {} tokens", 64 / 4);
    println!("- If expecting 64 elements as bytes: tensor might be i8 type");
    
    Ok(())
}