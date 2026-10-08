use std::collections::HashMap;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use candle_core::{DType, Device, Tensor};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use super::*;
use crate::Error;

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("kit-forecast-ckpt-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
enum Py {
    Int(i32),
    Float(f64),
    Str(String),
    Bool(bool),
    None,
    Dict(Vec<(String, Py)>),
    Raw(Vec<u8>),
}

fn s(text: &str) -> Py {
    Py::Str(text.to_string())
}

fn dict(entries: Vec<(&str, Py)>) -> Py {
    Py::Dict(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

fn put_str(out: &mut Vec<u8>, text: &str) {
    out.push(b'X');
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

fn put_int(out: &mut Vec<u8>, value: i32) {
    out.push(b'J');
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_global(out: &mut Vec<u8>, module: &str, class: &str) {
    out.push(b'c');
    out.extend_from_slice(format!("{module}\n{class}\n").as_bytes());
}

fn put_int_tuple(out: &mut Vec<u8>, values: &[usize]) {
    out.push(b'(');
    for &v in values {
        put_int(out, v as i32);
    }
    out.push(b't');
}

fn put_py(out: &mut Vec<u8>, value: &Py) {
    match value {
        Py::Int(v) => put_int(out, *v),
        Py::Float(v) => {
            out.push(b'G');
            out.extend_from_slice(&v.to_be_bytes());
        }
        Py::Str(v) => put_str(out, v),
        Py::Bool(true) => out.push(0x88),
        Py::Bool(false) => out.push(0x89),
        Py::None => out.push(b'N'),
        Py::Dict(entries) => {
            out.push(b'}');
            if !entries.is_empty() {
                out.push(b'(');
                for (key, v) in entries {
                    put_str(out, key);
                    put_py(out, v);
                }
                out.push(b'u');
            }
        }
        Py::Raw(bytes) => out.extend_from_slice(bytes),
    }
}

struct TensorDef {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
    stride: Option<Vec<usize>>,
    offset: usize,
}

fn tensor_pickle(def: &TensorDef, key: usize, class: &str, numel: usize) -> Vec<u8> {
    let mut out = Vec::new();
    put_global(&mut out, "torch._utils", "_rebuild_tensor_v2");
    out.push(b'(');
    out.push(b'(');
    put_str(&mut out, "storage");
    put_global(&mut out, "torch", class);
    put_str(&mut out, &key.to_string());
    put_str(&mut out, "cpu");
    put_int(&mut out, numel as i32);
    out.push(b't');
    out.push(b'Q');
    put_int(&mut out, def.offset as i32);
    put_int_tuple(&mut out, &def.shape);
    let stride = def.stride.clone().unwrap_or_else(|| {
        let mut stride = vec![1usize; def.shape.len()];
        for i in (0..def.shape.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * def.shape[i + 1];
        }
        stride
    });
    put_int_tuple(&mut out, &stride);
    out.push(0x89);
    put_global(&mut out, "collections", "OrderedDict");
    out.push(b')');
    out.push(b'R');
    out.push(b't');
    out.push(b'R');
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Layout {
    Ema,
    FlatEma,
    Model,
    Neither,
    EmptyEmaWithModel,
}

struct Fixture {
    model: ModelConfig,
    tensors: Vec<TensorDef>,
    layout: Layout,
    class: &'static str,
    omit_yday: bool,
    extra_model_key: bool,
    byteorder: &'static str,
    prefix: &'static str,
    junk_global: Option<Vec<u8>>,
}

fn tiny_config() -> ModelConfig {
    ModelConfig {
        d_model: 8,
        n_layers: 2,
        n_heads: 2,
        ffn_hidden: 12,
        in_features: 5,
        rope_theta: 10000.0,
        n_registers: 4,
        inst_vocab: 10,
        inst_dim: 4,
        market_slots: 3,
        sector_slots: 5,
        scale_slots: 8,
        t_emb_dim: 6,
        clock_dim: 16,
        sess_hidden: 6,
        yday_dim: 8,
        qk_norm: true,
        attn_backend: "auto".to_string(),
        use_activation_checkpointing: false,
    }
}

fn official_model_config() -> ModelConfig {
    ModelConfig {
        d_model: 1024,
        n_layers: 21,
        n_heads: 8,
        ffn_hidden: 2816,
        in_features: 5,
        rope_theta: 10000.0,
        n_registers: 4,
        inst_vocab: 8192,
        inst_dim: 512,
        market_slots: 16,
        sector_slots: 64,
        scale_slots: 16,
        t_emb_dim: 256,
        clock_dim: 16,
        sess_hidden: 256,
        yday_dim: 8,
        qk_norm: true,
        attn_backend: "auto".to_string(),
        use_activation_checkpointing: false,
    }
}

fn tensor_values(spec: &TensorSpec, salt: usize) -> Vec<f32> {
    let n = spec.numel().unwrap();
    (0..n)
        .map(|i| (((i * 31 + salt * 7) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

impl Fixture {
    fn new() -> Self {
        let model = tiny_config();
        let tensors = model
            .tensor_specs()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, spec)| TensorDef {
                values: tensor_values(&spec, i),
                name: spec.name,
                shape: spec.shape,
                stride: None,
                offset: 0,
            })
            .collect();
        Self {
            model,
            tensors,
            layout: Layout::Ema,
            class: "FloatStorage",
            omit_yday: false,
            extra_model_key: false,
            byteorder: "little",
            prefix: "ckpt/",
            junk_global: None,
        }
    }

    fn tensor(&mut self, name: &str) -> &mut TensorDef {
        self.tensors.iter_mut().find(|t| t.name == name).unwrap()
    }

    fn model_dict(&self) -> Py {
        let c = &self.model;
        let mut entries = vec![
            ("d_model", Py::Int(c.d_model as i32)),
            ("n_layers", Py::Int(c.n_layers as i32)),
            ("n_heads", Py::Int(c.n_heads as i32)),
            ("ffn_hidden", Py::Int(c.ffn_hidden as i32)),
            ("in_features", Py::Int(c.in_features as i32)),
            ("rope_theta", Py::Float(c.rope_theta)),
            ("n_registers", Py::Int(c.n_registers as i32)),
            ("inst_vocab", Py::Int(c.inst_vocab as i32)),
            ("inst_dim", Py::Int(c.inst_dim as i32)),
            ("market_slots", Py::Int(c.market_slots as i32)),
            ("sector_slots", Py::Int(c.sector_slots as i32)),
            ("scale_slots", Py::Int(c.scale_slots as i32)),
            ("t_emb_dim", Py::Int(c.t_emb_dim as i32)),
            ("clock_dim", Py::Int(c.clock_dim as i32)),
            ("sess_hidden", Py::Int(c.sess_hidden as i32)),
            ("yday_dim", Py::Int(c.yday_dim as i32)),
            ("qk_norm", Py::Bool(c.qk_norm)),
            ("attn_backend", s(&c.attn_backend)),
            ("use_activation_checkpointing", Py::Bool(false)),
        ];
        if self.omit_yday {
            entries.retain(|(k, _)| *k != "yday_dim");
        }
        if self.extra_model_key {
            entries.push(("future_option", Py::Int(7)));
        }
        dict(entries)
    }

    fn bytes_per_element(&self) -> usize {
        match self.class {
            "HalfStorage" => 2,
            _ => 4,
        }
    }

    fn build(&self) -> Vec<u8> {
        let mut state_entries: Vec<(String, Py)> = Vec::new();
        let mut storages: Vec<Vec<u8>> = Vec::new();
        for (key, def) in self.tensors.iter().enumerate() {
            let mut bytes = Vec::new();
            let width = self.bytes_per_element();
            for _ in 0..def.offset {
                bytes.extend(std::iter::repeat_n(0x55u8, width));
            }
            for v in &def.values {
                if width == 4 {
                    bytes.extend_from_slice(&v.to_le_bytes());
                } else {
                    bytes.extend_from_slice(&[0x00, 0x3c]);
                }
            }
            let numel = bytes.len() / width;
            storages.push(bytes);
            state_entries.push((
                def.name.clone(),
                Py::Raw(tensor_pickle(def, key, self.class, numel)),
            ));
        }
        if let Some(raw) = &self.junk_global {
            state_entries.push(("junk".to_string(), Py::Raw(raw.clone())));
        }
        let state = Py::Dict(state_entries);
        let mut top: Vec<(&str, Py)> = vec![
            ("step", Py::Int(321)),
            (
                "config",
                dict(vec![
                    ("model", self.model_dict()),
                    (
                        "data",
                        dict(vec![
                            ("Lc", Py::Int(64)),
                            ("Lh", Py::Int(16)),
                            ("trunc_min", Py::Int(32)),
                            ("scale_windows", s("1m:40/8/20, 5m:30/6")),
                            ("soft_clip_c", Py::Float(6.0)),
                            ("vol_ema_half_life_days", Py::Float(5.0)),
                            ("stride", Py::Int(0)),
                        ]),
                    ),
                    ("train", dict(vec![("ema_decay", Py::Float(0.5))])),
                    (
                        "infer",
                        dict(vec![("steps", Py::Int(12)), ("solver", s("heun"))]),
                    ),
                ]),
            ),
            ("world_size", Py::Int(1)),
        ];
        match self.layout {
            Layout::Ema => top.push((
                "ema",
                dict(vec![("decay", Py::Float(0.9999)), ("shadow", state)]),
            )),
            Layout::FlatEma => top.push(("ema", state)),
            Layout::Model => top.push(("model", state)),
            Layout::Neither => top.push(("ema", Py::None)),
            Layout::EmptyEmaWithModel => {
                top.push(("ema", Py::Dict(vec![])));
                top.push(("model", state));
            }
        }
        let mut pickle = vec![0x80, 2];
        put_py(
            &mut pickle,
            &Py::Dict(top.into_iter().map(|(k, v)| (k.to_string(), v)).collect()),
        );
        pickle.push(b'.');

        let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        let prefix = self.prefix;
        zip.start_file(format!("{prefix}data.pkl"), options)
            .unwrap();
        zip.write_all(&pickle).unwrap();
        zip.start_file(format!("{prefix}byteorder"), options)
            .unwrap();
        zip.write_all(self.byteorder.as_bytes()).unwrap();
        for (key, bytes) in storages.iter().enumerate() {
            zip.start_file(format!("{prefix}data/{key}"), options)
                .unwrap();
            zip.write_all(bytes).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    fn load(&self) -> Result<Checkpoint> {
        Checkpoint::load_pt_reader(Cursor::new(self.build()), &Device::Cpu)
    }
}

fn expect_checkpoint_err<T: std::fmt::Debug>(result: Result<T>, needle: &str) {
    match result {
        Err(Error::Checkpoint(msg)) => {
            assert!(msg.contains(needle), "message {msg:?} lacks {needle:?}")
        }
        other => panic!("expected Error::Checkpoint containing {needle:?}, got {other:?}"),
    }
}

fn expect_config_err<T: std::fmt::Debug>(result: Result<T>, needle: &str) {
    match result {
        Err(Error::InvalidConfig(msg)) => {
            assert!(msg.contains(needle), "message {msg:?} lacks {needle:?}")
        }
        other => panic!("expected Error::InvalidConfig containing {needle:?}, got {other:?}"),
    }
}

fn flat(weights: &ModelWeights, name: &str) -> Vec<f32> {
    weights
        .get(name)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn tiny_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_ckpt.pt")
}

#[test]
fn official_config_yields_201_tensors_and_285789189_parameters() {
    let cfg = official_model_config();
    let specs = cfg.tensor_specs().unwrap();
    assert_eq!(specs.len(), 201);
    assert_eq!(cfg.parameter_count().unwrap(), 285_789_189);
    let names: std::collections::HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names.len(), specs.len());
}

#[test]
fn official_config_spot_checks_shapes() {
    let specs: HashMap<String, Vec<usize>> = official_model_config()
        .tensor_specs()
        .unwrap()
        .into_iter()
        .map(|s| (s.name, s.shape))
        .collect();
    assert_eq!(specs["register"], vec![4, 1024]);
    assert_eq!(specs["cond_emb.inst.weight"], vec![8192, 512]);
    assert_eq!(specs["adaln.w2.weight"], vec![6144, 1024]);
    assert_eq!(specs["blocks.20.attn.q_norm.weight"], vec![128]);
    assert!(!specs.contains_key("blocks.21.bias_i"));
}

#[test]
fn qk_norm_disabled_drops_norm_tensors() {
    let mut cfg = tiny_config();
    let with = cfg.tensor_specs().unwrap().len();
    cfg.qk_norm = false;
    let without = cfg.tensor_specs().unwrap().len();
    assert_eq!(with - without, 2 * cfg.n_layers);
}

#[test]
fn model_config_validation_rejects_bad_dimensions() {
    let mut cfg = tiny_config();
    cfg.n_heads = 3;
    expect_config_err(cfg.validate(), "not divisible");
    let mut cfg = tiny_config();
    cfg.d_model = 0;
    expect_config_err(cfg.validate(), "d_model");
    let mut cfg = tiny_config();
    cfg.t_emb_dim = 5;
    expect_config_err(cfg.validate(), "t_emb_dim");
    let mut cfg = tiny_config();
    cfg.rope_theta = f64::NAN;
    expect_config_err(cfg.validate(), "rope_theta");
}

#[test]
fn model_config_json_tolerates_unknown_keys_but_requires_layout_fields() {
    let mut value = serde_json::to_value(tiny_config()).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("surprise".to_string(), serde_json::json!(1));
    let parsed: ModelConfig = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(parsed, tiny_config());
    value.as_object_mut().unwrap().remove("n_layers");
    assert!(serde_json::from_value::<ModelConfig>(value).is_err());
}

#[test]
fn windows_follow_upstream_semantics() {
    let w = Windows::parse("1m:1200/120/300, 5m:960/96 ,1d:250/20/64", 1024, 128, 256).unwrap();
    assert_eq!(
        w.get("1m"),
        WindowSpec {
            context_len: 1200,
            horizon_len: 120,
            trunc_min: 300
        }
    );
    assert_eq!(w.get("5m").trunc_min, 256);
    let narrow = Windows::parse("30m:100/10", 1024, 128, 256).unwrap();
    assert_eq!(narrow.get("30m").trunc_min, 100);
    assert_eq!(
        w.get("15m"),
        WindowSpec {
            context_len: 1024,
            horizon_len: 128,
            trunc_min: 256
        }
    );
    assert_eq!(w.get("1d").total_len(), 270);
    let none = Windows::parse("", 64, 16, 256).unwrap();
    assert!(none.by_timescale.is_empty());
    assert_eq!(none.get("1h").trunc_min, 64);
}

#[test]
fn windows_reject_malformed_specs() {
    let parse = |spec: &str| Windows::parse(spec, 64, 16, 32);
    expect_config_err(parse("1m"), "missing ':'");
    expect_config_err(parse("3m:10/2"), "unknown timescale");
    expect_config_err(parse("1m:10/2,1m:20/4"), "more than once");
    expect_config_err(parse("1m:"), "no value");
    expect_config_err(parse("1m:10"), "Lc/Lh");
    expect_config_err(parse("1m:10/2/3/4"), "Lc/Lh");
    expect_config_err(parse("1m:ten/2"), "not an integer");
    expect_config_err(parse("1m:0/2"), "positive");
    expect_config_err(parse("1m:-4/2"), "positive");
    expect_config_err(parse("1m:10/2/11"), "must not exceed");
}

#[test]
fn loads_synthetic_checkpoint_with_exact_values() {
    let fixture = Fixture::new();
    let ckpt = fixture.load().unwrap();
    let specs = fixture.model.tensor_specs().unwrap();
    assert_eq!(ckpt.config.model, fixture.model);
    assert_eq!(ckpt.weights.len(), specs.len());
    assert_eq!(
        ckpt.weights.parameter_count() as u64,
        fixture.model.parameter_count().unwrap()
    );
    assert_eq!(ckpt.info.step, Some(321));
    assert_eq!(ckpt.info.ema_decay, Some(0.9999));
    assert_eq!(ckpt.info.source_kind, SourceKind::Ema);
    assert_eq!(ckpt.info.source_kind.as_str(), "ema");
    assert_eq!(ckpt.config.window("1m").context_len, 40);
    assert_eq!(ckpt.config.window("1m").trunc_min, 20);
    assert_eq!(ckpt.config.window("5m").trunc_min, 30);
    assert_eq!(ckpt.config.window("1d").context_len, 64);
    assert_eq!(ckpt.config.soft_clip_c, 6.0);
    assert_eq!(ckpt.config.vol_ema_half_life_days, 5.0);
    assert_eq!(ckpt.config.infer.steps, 12);
    assert_eq!(ckpt.config.infer.solver, "heun");
    assert_eq!(ckpt.config.infer.n_paths, 64);
    assert_eq!(ckpt.config.infer.cfg_w, 1.0);
    for (i, spec) in specs.iter().enumerate() {
        let t = ckpt.weights.get(&spec.name).unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.dims(), spec.shape.as_slice());
        assert_eq!(flat(&ckpt.weights, &spec.name), tensor_values(spec, i));
    }
}

#[test]
fn archive_prefix_is_taken_from_the_pickle_entry() {
    for prefix in ["", "_clean/", "a/b/"] {
        let mut fixture = Fixture::new();
        fixture.prefix = prefix;
        assert_eq!(
            fixture.load().unwrap().weights.len(),
            49,
            "prefix {prefix:?}"
        );
    }
}

#[test]
fn tolerates_unknown_model_keys() {
    let mut fixture = Fixture::new();
    fixture.extra_model_key = true;
    assert_eq!(fixture.load().unwrap().config.model, tiny_config());
}

#[test]
fn honors_storage_offsets() {
    let mut fixture = Fixture::new();
    fixture.tensor("head_bias").offset = 3;
    let ckpt = fixture.load().unwrap();
    let expected = tensor_values(
        fixture
            .model
            .tensor_specs()
            .unwrap()
            .iter()
            .find(|s| s.name == "head_bias")
            .unwrap(),
        1,
    );
    assert_eq!(flat(&ckpt.weights, "head_bias"), expected);
}

#[test]
fn flat_ema_dictionary_is_accepted() {
    let mut fixture = Fixture::new();
    fixture.layout = Layout::FlatEma;
    let ckpt = fixture.load().unwrap();
    assert_eq!(ckpt.info.source_kind, SourceKind::Ema);
    assert_eq!(ckpt.info.ema_decay, None);
}

#[test]
fn falls_back_to_model_weights_without_ema() {
    for layout in [Layout::Model, Layout::EmptyEmaWithModel] {
        let mut fixture = Fixture::new();
        fixture.layout = layout;
        let ckpt = fixture.load().unwrap();
        assert_eq!(ckpt.info.source_kind, SourceKind::Model);
        assert_eq!(ckpt.info.source_kind.as_str(), "model");
        assert_eq!(ckpt.info.ema_decay, None);
        assert_eq!(ckpt.weights.len(), 49);
    }
}

#[test]
fn rejects_checkpoint_without_ema_or_model() {
    let mut fixture = Fixture::new();
    fixture.layout = Layout::Neither;
    expect_checkpoint_err(fixture.load(), "neither `ema` nor `model`");
}

#[test]
fn rejects_missing_yday_dim() {
    let mut fixture = Fixture::new();
    fixture.omit_yday = true;
    expect_checkpoint_err(fixture.load(), "yday_dim");
}

#[test]
fn rejects_missing_tensor() {
    let mut fixture = Fixture::new();
    fixture.tensors.retain(|t| t.name != "out_proj.bias");
    match fixture.load() {
        Err(Error::MissingTensor(name)) => assert_eq!(name, "out_proj.bias"),
        other => panic!("expected MissingTensor, got {other:?}"),
    }
}

#[test]
fn rejects_unexpected_tensors_and_lists_them() {
    let mut fixture = Fixture::new();
    for extra in ["model.extra", "another.one"] {
        fixture.tensors.push(TensorDef {
            name: extra.to_string(),
            shape: vec![2],
            values: vec![0.0, 1.0],
            stride: None,
            offset: 0,
        });
    }
    match fixture.load() {
        Err(Error::Checkpoint(msg)) => {
            assert!(
                msg.contains("model.extra") && msg.contains("another.one"),
                "{msg}"
            );
            assert!(msg.contains("2 unexpected"), "{msg}");
        }
        other => panic!("expected Checkpoint error, got {other:?}"),
    }
}

#[test]
fn rejects_wrong_shape() {
    let mut fixture = Fixture::new();
    let def = fixture.tensor("head_bias");
    def.shape = vec![4, 4];
    match fixture.load() {
        Err(Error::ShapeMismatch {
            name,
            actual,
            expected,
        }) => {
            assert_eq!(name, "head_bias");
            assert_eq!(actual, vec![4, 4]);
            assert_eq!(expected, vec![16]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn rejects_non_f32_dtype() {
    let mut fixture = Fixture::new();
    fixture.class = "HalfStorage";
    expect_checkpoint_err(fixture.load(), "F16");
}

#[test]
fn rejects_unsupported_storage_class() {
    let mut fixture = Fixture::new();
    fixture.class = "ComplexFloatStorage";
    expect_checkpoint_err(fixture.load(), "ComplexFloatStorage");
}

#[test]
fn rejects_non_finite_values() {
    let mut fixture = Fixture::new();
    fixture.tensor("register").values[3] = f32::NAN;
    expect_checkpoint_err(fixture.load(), "non-finite");
    let mut fixture = Fixture::new();
    fixture.tensor("register").values[0] = f32::INFINITY;
    expect_checkpoint_err(fixture.load(), "non-finite");
}

#[test]
fn rejects_non_contiguous_tensor() {
    let mut fixture = Fixture::new();
    fixture.tensor("in_proj.weight").stride = Some(vec![1, 8]);
    expect_checkpoint_err(fixture.load(), "not contiguous");
}

#[test]
fn rejects_storage_that_is_too_small() {
    let mut fixture = Fixture::new();
    let def = fixture.tensor("head_bias");
    def.offset = 5;
    def.values.truncate(10);
    def.shape = vec![16];
    expect_checkpoint_err(fixture.load(), "head_bias");
}
#[test]
fn rejects_big_endian_archives() {
    let mut fixture = Fixture::new();
    fixture.byteorder = "big";
    expect_checkpoint_err(fixture.load(), "byte order");
}

#[test]
fn rejects_disallowed_pickle_globals() {
    let mut call = Vec::new();
    put_global(&mut call, "os", "system");
    put_str(&mut call, "echo pwned");
    call.extend_from_slice(&[0x85, b'R']);
    let mut fixture = Fixture::new();
    fixture.junk_global = Some(call);
    expect_checkpoint_err(fixture.load(), "os.system");

    let mut bare = Vec::new();
    put_global(&mut bare, "builtins", "eval");
    let mut fixture = Fixture::new();
    fixture.junk_global = Some(bare);
    expect_checkpoint_err(fixture.load(), "builtins.eval");

    let mut spoofed = Vec::new();
    put_global(&mut spoofed, "torch", "FloatStorage");
    spoofed.extend_from_slice(b")R");
    let mut fixture = Fixture::new();
    fixture.junk_global = Some(spoofed);
    expect_checkpoint_err(fixture.load(), "torch.FloatStorage");
}

#[test]
fn rejects_truncated_and_foreign_files() {
    let bytes = Fixture::new().build();
    for cut in [bytes.len() / 2, bytes.len() - 10, 30, 0] {
        let result = Checkpoint::load_pt_reader(Cursor::new(bytes[..cut].to_vec()), &Device::Cpu);
        assert!(
            matches!(result, Err(Error::Checkpoint(_))),
            "cut at {cut}: {result:?}"
        );
    }
    expect_checkpoint_err(
        Checkpoint::load_pt_reader(
            Cursor::new(b"this is not a zip file".to_vec()),
            &Device::Cpu,
        ),
        "zip",
    );
}

#[test]
fn rejects_corrupt_pickle() {
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file("x/data.pkl", options).unwrap();
    zip.write_all(&[0x80, 2, 0xff, 0xff]).unwrap();
    let bytes = zip.finish().unwrap().into_inner();
    expect_checkpoint_err(
        Checkpoint::load_pt_reader(Cursor::new(bytes), &Device::Cpu),
        "malformed pickle",
    );
}

#[test]
fn rejects_archive_without_pickle() {
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file("x/other", options).unwrap();
    zip.write_all(b"1").unwrap();
    let bytes = zip.finish().unwrap().into_inner();
    expect_checkpoint_err(
        Checkpoint::load_pt_reader(Cursor::new(bytes), &Device::Cpu),
        "data.pkl",
    );
}

#[test]
fn missing_file_is_an_io_error() {
    let result = Checkpoint::load_pt(
        std::env::temp_dir().join("kit-forecast-definitely-missing.pt"),
        &Device::Cpu,
    );
    assert!(matches!(result, Err(Error::Io(_))), "{result:?}");
}

#[test]
fn var_builder_serves_nested_names() {
    let ckpt = Fixture::new().load().unwrap();
    let d = ckpt.config.model.d_model;
    let vb = ckpt.weights.var_builder();
    let w = vb
        .pp("blocks")
        .pp("1")
        .pp("attn")
        .pp("qkv")
        .get((3 * d, d), "weight")
        .unwrap();
    assert_eq!(w.dtype(), DType::F32);
    assert_eq!(w.dims(), &[3 * d, d]);
    assert_eq!(
        w.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        flat(&ckpt.weights, "blocks.1.attn.qkv.weight")
    );
    assert!(vb.pp("blocks").pp("2").get(d, "bias_i").is_err());
    let map = ckpt.weights.clone().into_tensors();
    assert_eq!(map.len(), 49);
}

#[test]
fn from_tensors_validates_strictly() {
    let cfg = tiny_config();
    let good = || -> HashMap<String, Tensor> {
        Fixture::new()
            .tensors
            .into_iter()
            .map(|t| {
                let tensor = Tensor::from_vec(t.values, t.shape, &Device::Cpu).unwrap();
                (t.name, tensor)
            })
            .collect()
    };
    assert_eq!(ModelWeights::from_tensors(good(), &cfg).unwrap().len(), 49);

    let mut map = good();
    map.remove("register");
    assert!(matches!(
        ModelWeights::from_tensors(map, &cfg),
        Err(Error::MissingTensor(n)) if n == "register"
    ));

    let mut map = good();
    map.insert(
        "zzz".to_string(),
        Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap(),
    );
    expect_checkpoint_err(ModelWeights::from_tensors(map, &cfg), "zzz");

    let mut map = good();
    map.insert(
        "head_bias".to_string(),
        Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap(),
    );
    assert!(matches!(
        ModelWeights::from_tensors(map, &cfg),
        Err(Error::ShapeMismatch { .. })
    ));

    let mut map = good();
    map.insert(
        "head_bias".to_string(),
        Tensor::zeros(16, DType::F64, &Device::Cpu).unwrap(),
    );
    expect_checkpoint_err(ModelWeights::from_tensors(map, &cfg), "F64");

    let mut map = good();
    let nan = Tensor::from_vec(vec![f32::NAN; 16], 16, &Device::Cpu).unwrap();
    map.insert("head_bias".to_string(), nan);
    expect_checkpoint_err(ModelWeights::from_tensors(map, &cfg), "non-finite");

    assert!(matches!(
        ModelWeights::from_tensors(HashMap::new(), &cfg),
        Err(Error::MissingTensor(_))
    ));
}

#[test]
fn kit_config_round_trips_through_json() {
    let ckpt = Fixture::new().load().unwrap();
    let text = ckpt.config.to_json().unwrap();
    let again = KitConfig::from_json(&text).unwrap();
    assert_eq!(again, ckpt.config);
    let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
    value["soft_clip_c"] = serde_json::json!(-1.0);
    expect_config_err(KitConfig::from_json(&value.to_string()), "soft_clip_c");
    assert!(matches!(KitConfig::from_json("{"), Err(Error::Json(_))));
}

#[test]
fn directory_format_round_trips_and_stays_strict() {
    let tmp = TempDir::new();
    let ckpt = Fixture::new().load().unwrap();
    let dir = tmp.path().join("exchange");
    ckpt.save_dir(&dir).unwrap();
    for file in [WEIGHTS_FILE, CONFIG_FILE, INFO_FILE] {
        assert!(dir.join(file).is_file(), "{file}");
    }
    let back = Checkpoint::load_dir(&dir, &Device::Cpu).unwrap();
    assert_eq!(back.config, ckpt.config);
    assert_eq!(back.info, ckpt.info);
    for name in ckpt.weights.tensors().keys() {
        assert_eq!(
            flat(&back.weights, name),
            flat(&ckpt.weights, name),
            "{name}"
        );
    }

    std::fs::remove_file(dir.join(INFO_FILE)).unwrap();
    let no_info = Checkpoint::load_dir(&dir, &Device::Cpu).unwrap();
    assert_eq!(no_info.info.source_kind, SourceKind::Model);
    assert_eq!(no_info.info.step, None);

    let mut partial = ckpt.weights.clone().into_tensors();
    partial.remove("out_proj.bias");
    candle_core::safetensors::save(&partial, dir.join(WEIGHTS_FILE)).unwrap();
    assert!(matches!(
        Checkpoint::load_dir(&dir, &Device::Cpu),
        Err(Error::MissingTensor(n)) if n == "out_proj.bias"
    ));

    std::fs::remove_file(dir.join(CONFIG_FILE)).unwrap();
    expect_checkpoint_err(Checkpoint::load_dir(&dir, &Device::Cpu), "file not found");
}

#[test]
fn tiny_pytorch_fixture_loads_with_torch_golden_values() {
    let ckpt = Checkpoint::load_pt(tiny_fixture_path(), &Device::Cpu).unwrap();
    assert_eq!(ckpt.weights.len(), 49);
    assert_eq!(ckpt.weights.parameter_count(), 9525);
    assert_eq!(ckpt.info.step, Some(1234));
    assert_eq!(ckpt.info.ema_decay, Some(0.999));
    assert_eq!(ckpt.info.source_kind, SourceKind::Ema);
    let m = &ckpt.config.model;
    assert_eq!(
        (
            m.d_model,
            m.n_layers,
            m.n_heads,
            m.ffn_hidden,
            m.inst_vocab,
            m.inst_dim
        ),
        (16, 2, 4, 24, 16, 8)
    );
    assert_eq!((m.sess_hidden, m.t_emb_dim, m.yday_dim), (16, 16, 8));
    assert_eq!(m.attn_backend, "math");
    assert_eq!(
        ckpt.config.window("1m"),
        WindowSpec {
            context_len: 32,
            horizon_len: 8,
            trunc_min: 16
        }
    );
    assert_eq!(
        ckpt.config.window("5m"),
        WindowSpec {
            context_len: 24,
            horizon_len: 6,
            trunc_min: 16
        }
    );
    assert_eq!(
        ckpt.config.window("1d"),
        WindowSpec {
            context_len: 20,
            horizon_len: 4,
            trunc_min: 10
        }
    );
    assert_eq!(
        ckpt.config.window("1h"),
        WindowSpec {
            context_len: 48,
            horizon_len: 8,
            trunc_min: 16
        }
    );
    assert_eq!(ckpt.config.infer.solver, "heun");
    assert_eq!(ckpt.config.infer.steps, 8);
    assert_eq!(ckpt.config.infer.n_paths, 4);
    assert_eq!(ckpt.config.infer.cfg_w, 1.5);
    assert_eq!(ckpt.config.infer.hist_cfg_w, 1.25);

    let head = |name: &str| flat(&ckpt.weights, name)[..4].to_vec();
    let sum = |name: &str| {
        flat(&ckpt.weights, name)
            .iter()
            .map(|&v| f64::from(v))
            .sum::<f64>()
    };
    assert_eq!(
        head("in_proj.weight"),
        [
            0.0012127164518460631f64,
            0.036588843911886215,
            -0.01614626869559288,
            -0.008265744894742966
        ]
        .map(|v| v as f32)
    );
    assert!((sum("in_proj.weight") - 0.1637566153658554).abs() < 1e-5);
    assert_eq!(
        head("cond_emb.inst.weight"),
        [
            -0.02041001431643963f64,
            0.02205798402428627,
            -0.04905413091182709,
            -0.039624329656362534
        ]
        .map(|v| v as f32)
    );
    assert!((sum("cond_emb.inst.weight") - -1.1132061046373565).abs() < 1e-5);
    assert_eq!(
        head("blocks.1.mlp.w2.weight"),
        [
            0.09774298220872879f64,
            -0.00015018774138297886,
            -0.09545138478279114,
            0.05691462382674217
        ]
        .map(|v| v as f32)
    );
    assert!((sum("blocks.1.mlp.w2.weight") - -0.7583121488314646).abs() < 1e-5);
    assert!((sum("out_proj.bias") - 0.19449548795819283).abs() < 1e-6);
}

#[test]
fn tiny_pytorch_fixture_survives_the_exchange_format() {
    let tmp = TempDir::new();
    let ckpt = Checkpoint::load_pt(tiny_fixture_path(), &Device::Cpu).unwrap();
    ckpt.save_dir(tmp.path()).unwrap();
    let back = Checkpoint::load_dir(tmp.path(), &Device::Cpu).unwrap();
    assert_eq!(back.config, ckpt.config);
    assert_eq!(
        flat(&back.weights, "register"),
        flat(&ckpt.weights, "register")
    );
}

#[test]
#[ignore = "needs references/ckpt/step_market_284M.pt (1.1 GB); run with --ignored"]
fn official_checkpoint_loads_with_torch_golden_values() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("references/ckpt/step_market_284M.pt");
    let ckpt = Checkpoint::load_pt(path, &Device::Cpu).unwrap();
    assert_eq!(ckpt.weights.len(), 201);
    assert_eq!(ckpt.weights.parameter_count(), 285_789_189);
    assert_eq!(ckpt.info.step, Some(50_000));
    assert_eq!(ckpt.info.ema_decay, Some(0.9999));
    assert_eq!(ckpt.info.source_kind, SourceKind::Ema);
    assert_eq!(ckpt.config.model, official_model_config());
    assert_eq!(ckpt.config.soft_clip_c, 6.0);
    assert_eq!(ckpt.config.vol_ema_half_life_days, 5.0);
    let expected = [
        ("1m", 1200, 120, 300),
        ("5m", 960, 96, 240),
        ("15m", 800, 64, 200),
        ("30m", 640, 40, 160),
        ("1h", 400, 20, 100),
        ("2h", 360, 20, 90),
        ("1d", 250, 20, 64),
    ];
    for (scale, context_len, horizon_len, trunc_min) in expected {
        assert_eq!(
            ckpt.config.window(scale),
            WindowSpec {
                context_len,
                horizon_len,
                trunc_min
            },
            "{scale}"
        );
    }
    assert_eq!(ckpt.config.infer.steps, 32);
    assert_eq!(ckpt.config.infer.solver, "euler");
    assert_eq!(ckpt.config.infer.n_paths, 64);

    let weights = &ckpt.weights;
    assert_eq!(
        flat(weights, "in_proj.weight")[..4],
        [
            -0.003224718850106001f64,
            0.006248371675610542,
            -0.006779349874705076,
            0.026672085747122765
        ]
        .map(|v| v as f32)
    );
    let inst = flat(weights, "cond_emb.inst.weight");
    assert_eq!(
        inst[..4],
        [
            0.00431020325049758f64,
            0.009009630419313908,
            -0.04042408615350723,
            -0.0021446468308568
        ]
        .map(|v| v as f32)
    );
    let inst_sum: f64 = inst.iter().map(|&v| f64::from(v)).sum();
    assert!((inst_sum - 29.76612091064453).abs() < 1e-2, "{inst_sum}");
    let w2 = flat(weights, "blocks.20.mlp.w2.weight");
    assert_eq!(
        w2[w2.len() - 4..],
        [
            0.01663339138031006f64,
            -0.011036165989935398,
            0.042404480278491974,
            0.001349194790236652
        ]
        .map(|v| v as f32)
    );
    let w2_sum: f64 = w2.iter().map(|&v| f64::from(v)).sum();
    assert!((w2_sum - 58.52176892594134).abs() < 1e-2, "{w2_sum}");
    assert_eq!(
        flat(weights, "register")[..4],
        [
            -0.003590756095945835f64,
            0.006117996294051409,
            0.008689154870808125,
            -0.013278232887387276
        ]
        .map(|v| v as f32)
    );
    assert_eq!(
        flat(weights, "out_proj.bias"),
        [
            -0.0007236311794258654f64,
            -0.0009423706214874983,
            -0.013600477017462254,
            -0.01346997544169426,
            0.004780667368322611
        ]
        .map(|v| v as f32)
    );
}
