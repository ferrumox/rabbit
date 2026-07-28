//! A safetensors WRITER — rabbit's own `safetensors.rs` only ever reads checkpoints (an
//! inference engine has no reason to write one), but a converter's whole job is producing a new
//! checkpoint directory, so this is genuinely new. Mirrors `safetensors.numpy.save_file`'s
//! on-disk layout exactly (same one `crate::safetensors::Shards::open` already reads): an 8-byte
//! little-endian header length, a JSON header mapping each tensor name to
//! `{dtype, shape, data_offsets:[start,end]}`, then every tensor's raw bytes concatenated in
//! header order.
//!
//! Confirmed against the real checkpoint rabbit already runs (`glm-5.2-colibri-int4`): a
//! quantized weight's `shape` is its FLAT byte-count (`[6291456]`, not `[rows,cols]`) and its
//! `.qs` scale sidecar's `shape` is likewise flat (`[n_scales]`) — `rows`/`cols` for a packed U8
//! tensor come from the model's own config at load time, not from the safetensors header, so
//! this writer doesn't need to (and shouldn't try to) reconstruct a 2D shape for those. Plain
//! F32 tensors (norms, kept-f32 biases, the router) keep whatever real shape they had after
//! dequant, matching `convert_shard`'s own `out_dict[name] = w.astype(np.float32)` (no flatten).

use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// One tensor's data + declared shape, keyed by name in the caller's `BTreeMap` (sorted purely
/// for deterministic, diffable output across runs — the safetensors format itself doesn't
/// require any particular physical tensor order, only that each `data_offsets` pair is
/// internally consistent, which this writer guarantees regardless of order).
pub enum OutTensor {
    F32 { shape: Vec<u64>, data: Vec<f32> },
    U8 { shape: Vec<u64>, data: Vec<u8> },
}

impl OutTensor {
    fn dtype(&self) -> &'static str {
        match self {
            OutTensor::F32 { .. } => "F32",
            OutTensor::U8 { .. } => "U8",
        }
    }

    fn raw_bytes(&self) -> Vec<u8> {
        match self {
            OutTensor::F32 { data, .. } => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            OutTensor::U8 { data, .. } => data.clone(),
        }
    }

    fn shape(&self) -> &[u64] {
        match self {
            OutTensor::F32 { shape, .. } => shape,
            OutTensor::U8 { shape, .. } => shape,
        }
    }
}

/// Writes `tensors` to `path` in safetensors format — the mirror of
/// `crate::safetensors::Shards::open` + `read_f32`/`read_raw`, one level up (a whole new file
/// instead of reading an existing one).
pub fn write_safetensors(path: &Path, tensors: &BTreeMap<String, OutTensor>) -> io::Result<()> {
    let mut header = Map::new();
    let mut data = Vec::new();
    for (name, t) in tensors {
        let bytes = t.raw_bytes();
        let start = data.len() as u64;
        data.extend_from_slice(&bytes);
        let end = data.len() as u64;
        header.insert(
            name.clone(),
            Value::Object({
                let mut m = Map::new();
                m.insert("dtype".to_string(), Value::String(t.dtype().to_string()));
                m.insert("shape".to_string(), Value::Array(t.shape().iter().map(|&d| Value::from(d)).collect()));
                m.insert("data_offsets".to_string(), Value::Array(vec![Value::from(start), Value::from(end)]));
                m
            }),
        );
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header))?;

    let mut f = File::create(path)?;
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    f.write_all(&header_bytes)?;
    f.write_all(&data)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::Shards;

    #[test]
    fn round_trips_through_rabbit_s_own_reader() {
        let dir = std::env::temp_dir().join("rabbit_test_convert_writer_roundtrip");
        std::fs::create_dir_all(&dir).unwrap();

        let mut tensors = BTreeMap::new();
        tensors.insert("model.norm.weight".to_string(), OutTensor::F32 { shape: vec![4], data: vec![1.0, 2.0, 3.0, 4.0] });
        tensors.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            OutTensor::U8 { shape: vec![6], data: vec![0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc] },
        );
        tensors.insert("model.layers.0.mlp.down_proj.weight.qs".to_string(), OutTensor::F32 { shape: vec![2], data: vec![0.5, 0.25] });

        let path = dir.join("out-00000.safetensors");
        write_safetensors(&path, &tensors).unwrap();

        let shards = Shards::open(&dir).unwrap();
        assert_eq!(shards.read_f32("model.norm.weight", false).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shards.read_raw("model.layers.0.mlp.down_proj.weight", false).unwrap(), vec![0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc]);
        assert_eq!(shards.read_f32("model.layers.0.mlp.down_proj.weight.qs", false).unwrap(), vec![0.5, 0.25]);

        std::fs::remove_dir_all(&dir).ok();
    }
}
