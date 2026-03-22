//! Pure-Rust CNN inference for WiFi CSI room presence classification.
//!
//! v5: L2 normalization + baseline subtraction + global normalization +
//!     adaptive baseline + temporal voting + hysteresis (sitting↔walking).

use std::collections::HashMap;
use std::path::Path;

const NUM_NODES: usize = 3;
const NUM_SUBCARRIERS: usize = 56;
const FEATURES: usize = NUM_NODES * NUM_SUBCARRIERS; // 168
const WINDOW: usize = 40;
const SUBSAMPLE: usize = 5;
const INFER_EVERY: usize = 1;
const CLASSES: [&str; 4] = ["empty", "lying", "walking", "sitting"];
const NUM_CLASSES: usize = 4;

const BASELINE_ADAPT_RATE: f32 = 0.005;
const BASELINE_CONFIDENCE_THRESHOLD: f32 = 0.85;
const VOTE_WINDOW: usize = 7; // increased from 5 for smoother predictions

// Hysteresis: higher confidence needed for confusing transitions
const TRANSITION_CONFIDENCE: f32 = 0.55;

// ── NN ops ───────────────────────────────────────────────────────────────────

fn conv1d(input: &[f32], out: &mut [f32], weight: &[f32], bias: &[f32],
          in_ch: usize, out_ch: usize, kernel: usize, padding: usize, seq_len: usize) {
    let out_len = seq_len;
    for oc in 0..out_ch {
        for t in 0..out_len {
            let mut sum = bias[oc];
            for ic in 0..in_ch {
                for k in 0..kernel {
                    let pos = t as isize + k as isize - padding as isize;
                    if pos >= 0 && (pos as usize) < seq_len {
                        sum += weight[oc * in_ch * kernel + ic * kernel + k]
                             * input[ic * seq_len + pos as usize];
                    }
                }
            }
            out[oc * out_len + t] = sum;
        }
    }
}

fn batchnorm(data: &mut [f32], gamma: &[f32], beta: &[f32],
             mean: &[f32], var: &[f32], channels: usize, seq_len: usize) {
    let eps = 1e-5f32;
    for c in 0..channels {
        let inv_std = 1.0 / (var[c] + eps).sqrt();
        for t in 0..seq_len {
            let idx = c * seq_len + t;
            data[idx] = gamma[c] * (data[idx] - mean[c]) * inv_std + beta[c];
        }
    }
}

fn relu(data: &mut [f32]) {
    for v in data.iter_mut() { if *v < 0.0 { *v = 0.0; } }
}

fn maxpool1d(input: &[f32], out: &mut [f32], channels: usize, in_len: usize) -> usize {
    let out_len = in_len / 2;
    for c in 0..channels {
        for t in 0..out_len {
            out[c * out_len + t] = input[c * in_len + t * 2].max(input[c * in_len + t * 2 + 1]);
        }
    }
    out_len
}

fn adaptive_avg_pool1d(input: &[f32], out: &mut [f32], channels: usize, in_len: usize) {
    for c in 0..channels {
        out[c] = (0..in_len).map(|t| input[c * in_len + t]).sum::<f32>() / in_len as f32;
    }
}

fn linear(input: &[f32], out: &mut [f32], weight: &[f32], bias: &[f32],
          in_f: usize, out_f: usize) {
    for o in 0..out_f {
        out[o] = bias[o] + (0..in_f).map(|i| weight[o * in_f + i] * input[i]).sum::<f32>();
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ── Weights ──────────────────────────────────────────────────────────────────

struct ModelWeights {
    conv1_w: Vec<f32>, conv1_b: Vec<f32>,
    bn1_gamma: Vec<f32>, bn1_beta: Vec<f32>, bn1_mean: Vec<f32>, bn1_var: Vec<f32>,
    conv2_w: Vec<f32>, conv2_b: Vec<f32>,
    bn2_gamma: Vec<f32>, bn2_beta: Vec<f32>, bn2_mean: Vec<f32>, bn2_var: Vec<f32>,
    conv3_w: Vec<f32>, conv3_b: Vec<f32>,
    bn3_gamma: Vec<f32>, bn3_beta: Vec<f32>, bn3_mean: Vec<f32>, bn3_var: Vec<f32>,
    fc1_w: Vec<f32>, fc1_b: Vec<f32>,
    fc2_w: Vec<f32>, fc2_b: Vec<f32>,
}

impl ModelWeights {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&data)?;
        let w = &json["weights"];
        let get = |key: &str| -> Vec<f32> {
            w[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()
        };
        Ok(Self {
            conv1_w: get("features.0.weight"), conv1_b: get("features.0.bias"),
            bn1_gamma: get("features.1.weight"), bn1_beta: get("features.1.bias"),
            bn1_mean: get("features.1.running_mean"), bn1_var: get("features.1.running_var"),
            conv2_w: get("features.5.weight"), conv2_b: get("features.5.bias"),
            bn2_gamma: get("features.6.weight"), bn2_beta: get("features.6.bias"),
            bn2_mean: get("features.6.running_mean"), bn2_var: get("features.6.running_var"),
            conv3_w: get("features.10.weight"), conv3_b: get("features.10.bias"),
            bn3_gamma: get("features.11.weight"), bn3_beta: get("features.11.bias"),
            bn3_mean: get("features.11.running_mean"), bn3_var: get("features.11.running_var"),
            fc1_w: get("classifier.0.weight"), fc1_b: get("classifier.0.bias"),
            fc2_w: get("classifier.3.weight"), fc2_b: get("classifier.3.bias"),
        })
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let seq = WINDOW;
        let mut buf1 = vec![0.0f32; 128 * seq];
        conv1d(input, &mut buf1, &self.conv1_w, &self.conv1_b, 168, 128, 7, 3, seq);
        batchnorm(&mut buf1, &self.bn1_gamma, &self.bn1_beta, &self.bn1_mean, &self.bn1_var, 128, seq);
        relu(&mut buf1);
        let mut pool1 = vec![0.0f32; 128 * (seq / 2)];
        let seq1 = maxpool1d(&buf1, &mut pool1, 128, seq);

        let mut buf2 = vec![0.0f32; 256 * seq1];
        conv1d(&pool1, &mut buf2, &self.conv2_w, &self.conv2_b, 128, 256, 5, 2, seq1);
        batchnorm(&mut buf2, &self.bn2_gamma, &self.bn2_beta, &self.bn2_mean, &self.bn2_var, 256, seq1);
        relu(&mut buf2);
        let mut pool2 = vec![0.0f32; 256 * (seq1 / 2)];
        let seq2 = maxpool1d(&buf2, &mut pool2, 256, seq1);

        let mut buf3 = vec![0.0f32; 128 * seq2];
        conv1d(&pool2, &mut buf3, &self.conv3_w, &self.conv3_b, 256, 128, 3, 1, seq2);
        batchnorm(&mut buf3, &self.bn3_gamma, &self.bn3_beta, &self.bn3_mean, &self.bn3_var, 128, seq2);
        relu(&mut buf3);
        let mut pooled = vec![0.0f32; 128];
        adaptive_avg_pool1d(&buf3, &mut pooled, 128, seq2);

        let mut fc1_out = vec![0.0f32; 64];
        linear(&pooled, &mut fc1_out, &self.fc1_w, &self.fc1_b, 128, 64);
        relu(&mut fc1_out);

        let mut logits = vec![0.0f32; 4];
        linear(&fc1_out, &mut logits, &self.fc2_w, &self.fc2_b, 64, 4);
        logits
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

pub struct DlClassifier {
    model: ModelWeights,
    node_latest: HashMap<u8, Vec<f32>>,
    frame_buffer: Vec<[f32; FEATURES]>,
    frame_count: usize,
    baseline: Vec<f32>,
    feat_mean: Vec<f32>,
    feat_std: Vec<f32>,
    last_class: usize,
    last_confidence: f32,
    vote_history: Vec<[f32; NUM_CLASSES]>,
    prev_reported_class: usize,
}

impl DlClassifier {
    pub fn load(model_dir: &Path) -> anyhow::Result<Self> {
        let weights_path = model_dir.join("csi_light_weights.json");
        tracing::info!("Loading DL model from {}", weights_path.display());
        let model = ModelWeights::load(&weights_path)?;

        let baseline = load_npy_f32(&model_dir.join("baseline.npy"))?;
        let feat_mean = load_npy_f32(&model_dir.join("feat_mean.npy"))?;
        let feat_std = load_npy_f32(&model_dir.join("feat_std.npy"))?;

        assert_eq!(baseline.len(), FEATURES);
        assert_eq!(feat_mean.len(), FEATURES);
        assert_eq!(feat_std.len(), FEATURES);

        tracing::info!("DL classifier ready: {} classes, window={}, vote={}, hysteresis={}",
                       CLASSES.len(), WINDOW, VOTE_WINDOW, TRANSITION_CONFIDENCE);

        Ok(Self {
            model,
            node_latest: HashMap::new(),
            frame_buffer: Vec::with_capacity(WINDOW),
            frame_count: 0,
            baseline,
            feat_mean,
            feat_std,
            last_class: 0,
            last_confidence: 0.0,
            vote_history: Vec::with_capacity(VOTE_WINDOW),
            prev_reported_class: 0,
        })
    }

    pub fn feed_frame(&mut self, node_id: u8, amplitudes: &[f64]) -> Option<(&'static str, f32)> {
        let mut padded = vec![0.0f32; NUM_SUBCARRIERS];
        let n = amplitudes.len().min(NUM_SUBCARRIERS);
        for i in 0..n { padded[i] = amplitudes[i] as f32; }
        self.node_latest.insert(node_id, padded);

        self.frame_count += 1;
        if self.frame_count % SUBSAMPLE != 0 { return None; }

        // 1. Assemble frame
        let mut raw_frame = [0.0f32; FEATURES];
        for ni in 0..NUM_NODES {
            let nid = (ni + 1) as u8;
            let off = ni * NUM_SUBCARRIERS;
            if let Some(amps) = self.node_latest.get(&nid) {
                for s in 0..NUM_SUBCARRIERS { raw_frame[off + s] = amps[s]; }
            }
        }

        // 2. L2 normalize (removes AGC artifacts)
        let l2_norm: f32 = raw_frame.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut l2_frame = raw_frame;
        if l2_norm > 1e-6 {
            for v in l2_frame.iter_mut() { *v /= l2_norm; }
        }

        // 3. Subtract baseline
        let mut frame = [0.0f32; FEATURES];
        for i in 0..FEATURES {
            frame[i] = l2_frame[i] - self.baseline[i];
        }

        if self.frame_buffer.len() >= WINDOW { self.frame_buffer.remove(0); }
        self.frame_buffer.push(frame);
        if self.frame_buffer.len() < WINDOW { return None; }

        let sc = self.frame_count / SUBSAMPLE;
        if sc % INFER_EVERY != 0 {
            return Some((CLASSES[self.last_class], self.last_confidence));
        }

        // 4. Build input with global normalization (features, seq_len) = (168, 40)
        let mut input = vec![0.0f32; FEATURES * WINDOW];
        for f in 0..FEATURES {
            let std_val = self.feat_std[f];
            let m = self.feat_mean[f];
            for t in 0..WINDOW {
                let raw = self.frame_buffer[t][f];
                input[f * WINDOW + t] = if std_val > 1e-6 { (raw - m) / std_val } else { 0.0 };
            }
        }

        let logits = self.model.forward(&input);
        let probs = softmax(&logits);

        // 5. Temporal voting: average probabilities over last VOTE_WINDOW predictions
        let mut prob_arr = [0.0f32; NUM_CLASSES];
        for (i, &p) in probs.iter().enumerate().take(NUM_CLASSES) {
            prob_arr[i] = p;
        }
        if self.vote_history.len() >= VOTE_WINDOW {
            self.vote_history.remove(0);
        }
        self.vote_history.push(prob_arr);

        let mut avg_probs = [0.0f32; NUM_CLASSES];
        let n_votes = self.vote_history.len() as f32;
        for vote in &self.vote_history {
            for c in 0..NUM_CLASSES {
                avg_probs[c] += vote[c];
            }
        }
        for c in 0..NUM_CLASSES {
            avg_probs[c] /= n_votes;
        }

        // Pick class from averaged probabilities
        let mut best_class = 0;
        let mut best_conf = avg_probs[0];
        for c in 1..NUM_CLASSES {
            if avg_probs[c] > best_conf {
                best_conf = avg_probs[c];
                best_class = c;
            }
        }

        // 6. Hysteresis: require higher confidence for sitting↔walking transitions
        let (final_class, final_conf) = if self.prev_reported_class != best_class {
            let is_confusing_transition = matches!(
                (self.prev_reported_class, best_class),
                (2, 3) | (3, 2) // walking(2) ↔ sitting(3)
            );
            if is_confusing_transition && best_conf < TRANSITION_CONFIDENCE {
                // Not confident enough — stay in current class
                (self.prev_reported_class, avg_probs[self.prev_reported_class])
            } else {
                (best_class, best_conf)
            }
        } else {
            (best_class, best_conf)
        };

        self.prev_reported_class = final_class;
        self.last_class = final_class;
        self.last_confidence = final_conf;

        // 7. Adaptive baseline: slowly update when empty detected
        if final_class == 0 && final_conf > BASELINE_CONFIDENCE_THRESHOLD {
            for i in 0..FEATURES {
                let raw_l2 = l2_frame[i];
                self.baseline[i] += BASELINE_ADAPT_RATE * (raw_l2 - self.baseline[i]);
            }
        }

        Some((CLASSES[self.last_class], self.last_confidence))
    }

    pub fn last_result(&self) -> (&'static str, f32) {
        (CLASSES[self.last_class], self.last_confidence)
    }

    pub fn is_ready(&self) -> bool {
        self.frame_buffer.len() >= WINDOW
    }
}

fn load_npy_f32(path: &Path) -> anyhow::Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        anyhow::bail!("Not a .npy file: {}", path.display());
    }
    let major = data[6];
    let hl = if major == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    let hs = if major == 1 { 10 } else { 12 };
    let hdr = std::str::from_utf8(&data[hs..hs + hl])?;
    let raw = &data[hs + hl..];

    if hdr.contains("<f4") || hdr.contains("float32") {
        Ok((0..raw.len() / 4)
            .map(|i| f32::from_le_bytes([raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]]))
            .collect())
    } else if hdr.contains("<f8") || hdr.contains("float64") {
        Ok((0..raw.len() / 8)
            .map(|i| {
                let b: [u8; 8] = raw[i*8..i*8+8].try_into().unwrap();
                f64::from_le_bytes(b) as f32
            })
            .collect())
    } else {
        anyhow::bail!("Unsupported dtype: {}", hdr);
    }
}
