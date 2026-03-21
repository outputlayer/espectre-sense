//! Vital signs detector — STUB (moving to deep learning)

use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VitalSigns {
    pub heart_rate_bpm: Option<f64>,
    pub heartbeat_bpm: f64,
    pub heartbeat_confidence: f64,
    pub breathing_rate_bpm: Option<f64>,
    pub breathing_rpm: f64,
    pub breathing_confidence: f64,
    pub signal_quality: f64,
}

pub struct VitalSignDetector {
    sample_rate: f64,
}

impl VitalSignDetector {
    pub fn new(sample_rate: f64) -> Self {
        Self { sample_rate }
    }
    pub fn process_frame(&mut self, _amplitude: &[f64], _phase: &[f64]) -> VitalSigns {
        VitalSigns::default()
    }
    pub fn process(&mut self, _amps: &[f32]) -> VitalSigns {
        VitalSigns::default()
    }
    pub fn extract_breathing(&self) -> (Option<f64>, f64) { (None, 0.0) }
    pub fn extract_heartbeat(&self) -> (Option<f64>, f64) { (None, 0.0) }
    pub fn reset(&mut self) {}
    pub fn buffer_status(&self) -> (usize, usize, usize, usize) { (0, 0, 0, 0) }
}

pub fn bandpass_filter(_data: &[f64], _low_hz: f64, _high_hz: f64, _sample_rate: f64) -> Vec<f64> {
    vec![]
}

pub fn run_benchmark(_n_frames: usize) -> (std::time::Duration, std::time::Duration) {
    (std::time::Duration::ZERO, std::time::Duration::ZERO)
}
