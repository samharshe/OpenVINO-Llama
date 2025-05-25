use std::{path::Path, sync::Arc};

use anyhow::Result;
use wasmtime::{Engine, Instance, Linker, Memory, Module, Store};
use wasmtime_wasi::{preview1::WasiP1Ctx, DirPerms, FilePerms, WasiCtxBuilder};
use wasmtime_wasi_nn::{backend::openvino::OpenvinoBackend, witx::WasiNnCtx, Backend, InMemoryRegistry};
use serde_json::Value;

struct Context
{
    wasi: WasiP1Ctx,
    wasi_nn: WasiNnCtx,
}

impl Context
{
    fn new(preopen_dir: &Path, preload_model: bool, mut backend: Backend) -> Result<Self>
    {
        let mut builder = WasiCtxBuilder::new();
        builder.inherit_stdio().preopened_dir(preopen_dir, "fixture", DirPerms::READ, FilePerms::READ)?;
        let wasi = builder.build_p1();

        let mut registry = InMemoryRegistry::new();
        if preload_model {
            registry.load((backend).as_dir_loadable().unwrap(), preopen_dir)?;
        }
        let wasi_nn = WasiNnCtx::new([backend], registry.into());

        Ok(Self {
            wasi,
            wasi_nn,
        })
    }
}

pub struct WasmInstance
{
    instance: Instance,
    store: Store<Context>,
    memory: Memory,
}

impl WasmInstance
{
    pub fn new(engine: Arc<Engine>, module: Arc<Module>) -> anyhow::Result<WasmInstance>
    {
        let path = Path::new("./fixture");
        let mut store =
            Store::new(&engine, Context::new(path, true, Backend::from(OpenvinoBackend::default())).unwrap());

        let mut linker = Linker::new(&engine);
        wasmtime_wasi_nn::witx::add_to_linker(&mut linker, |s: &mut Context| &mut s.wasi_nn)?;
        wasmtime_wasi::preview1::add_to_linker_sync(&mut linker, |s: &mut Context| &mut s.wasi)?;

        let instance = linker.instantiate(&mut store, &module)?;
        let memory = instance.get_memory(&mut store, "memory").expect("failed to find 'memory' export");

        Ok(Self {
            instance,
            store,
            memory,
        })
    }

    pub fn infer(&mut self, tensor_bytes: Vec<u8>) -> anyhow::Result<Value>
    {
        let result_ptr = 1000; // Buffer for JSON response (length + data)
        let ptr = 1008;
        self.memory.write(&mut self.store, ptr, &tensor_bytes)?;

        let infer = self.instance.get_typed_func::<(i32, i32, i32), i32>(&mut self.store, "infer")?;

        infer.call(&mut self.store, (ptr as i32, tensor_bytes.len() as i32, result_ptr))?;
        
        // Read length first (4 bytes)
        let mut length_buf = [0u8; 4];
        self.memory.read(&mut self.store, result_ptr as usize, &mut length_buf)?;
        let json_length = u32::from_le_bytes(length_buf) as usize;
        
        // Read JSON string
        let mut json_buf = vec![0u8; json_length];
        self.memory.read(&mut self.store, (result_ptr + 4) as usize, &mut json_buf)?;
        let json_str = String::from_utf8(json_buf)?;
        
        // Parse JSON and return
        let json_value: Value = serde_json::from_str(&json_str)?;
        Ok(json_value)
    }
}
