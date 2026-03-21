//! ESPectre detector — STUB (moving to deep learning)
//! Original algorithm removed. Keeping interface for backward compatibility.

use serde::{Serialize, Deserialize};

#[derive(Clone)]
pub struct EspectreNode {
    pub score: f32,
    pub is_motion: bool,
}

impl EspectreNode {
    pub fn new() -> Self {
        Self { score: 0.0, is_motion: false }
    }
    pub fn process(&mut self, _amplitudes: &[f32]) -> (f32, bool) {
        (0.0, false) // stub — DL model will replace this
    }
}

pub struct EspectreDetector {
    nodes: std::collections::HashMap<u8, EspectreNode>,
    data_path: Option<String>,
}

impl EspectreDetector {
    pub fn new(_n_nodes: usize) -> Self {
        Self { nodes: std::collections::HashMap::new(), data_path: None }
    }
    pub fn new_with_persistence(_n_nodes: usize, path: &str) -> Self {
        Self { nodes: std::collections::HashMap::new(), data_path: Some(path.to_string()) }
    }
    pub fn process_node(&mut self, node_id: u8, _amplitudes: &[f32]) -> (f32, bool) {
        let node = self.nodes.entry(node_id).or_insert_with(EspectreNode::new);
        (0.0, false) // stub
    }
    pub fn fused_result(&self) -> (f32, bool, Vec<f32>) {
        (0.0, false, vec![])
    }
}
