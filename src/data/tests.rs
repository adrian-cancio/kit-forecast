use chrono::NaiveDate;

use super::*;
use crate::market::{Market, Timescale, Timestamp, TradingCalendar};

type RawBar = (u32, u32, u32, f64, f64, f64, f64, f64);

const RAW: [RawBar; 32] = [
    (5, 10, 0, 10.0, 10.05, 9.98, 10.04, 1000.0),
    (5, 10, 30, 10.0593, 10.1087, 10.0439, 10.0901, 1285.0),
    (5, 11, 0, 10.1005, 10.1267, 10.0946, 10.1074, 1392.15),
    (5, 11, 30, 10.0935, 10.105, 10.073, 10.0731, 1278.21),
    (5, 13, 30, 10.0553, 10.0577, 10.0139, 10.0174, 1067.24),
    (5, 14, 0, 10.0217, 10.0221, 9.9714, 9.9842, 956.74),
    (5, 14, 30, 10.0041, 10.0113, 9.9649, 9.9845, 1068.17),
    (5, 15, 0, 9.9909, 10.0149, 9.9733, 9.9983, 1355.04),
    (6, 10, 0, 9.9817, 10.0326, 9.9732, 10.0127, 1638.1),
    (6, 10, 30, 9.9974, 10.0516, 9.9966, 10.0374, 1740.97),
    (6, 11, 0, 10.0458, 10.0807, 10.0442, 10.0761, 1623.64),
    (6, 11, 30, 10.096, 10.1022, 10.0858, 10.1022, 1412.74),
    (6, 13, 30, 10.1044, 10.1091, 10.0648, 10.0834, 1305.72),
    (6, 14, 0, 10.0647, 10.079, 10.0074, 10.0265, 1421.4),
    (6, 14, 30, 10.0144, 10.0343, 9.9658, 9.9771, 1710.09),
    (6, 15, 0, 9.9892, 10.0057, 9.9678, 9.9702, 1991.14),
    (7, 10, 0, 9.9887, 10.004, 9.9883, 9.9968, 2089.7),
    (7, 10, 30, 9.9946, 10.0264, 9.9874, 10.0261, 1969.02),
    (7, 11, 0, 10.0063, 10.0488, 9.9897, 10.0463, 1758.27),
    (7, 11, 30, 10.0379, 10.0793, 10.0179, 10.0677, 1654.78),
    (7, 13, 30, 10.0831, 10.1079, 10.0689, 10.0886, 1774.7),
    (7, 14, 0, 10.1053, 10.1239, 10.0792, 10.0837, 2065.13),
    (7, 14, 30, 10.0772, 10.0872, 10.0388, 10.0388, 2344.1),
    (7, 15, 0, 10.0188, 10.0203, 9.9771, 9.9817, 2438.35),
    (8, 10, 0, 9.9774, 9.9784, 9.9449, 9.959, 2314.38),
    (8, 10, 30, 9.9766, 9.994, 9.9567, 9.9854, 2103.85),
    (8, 11, 0, 9.9991, 10.0487, 9.9827, 10.031, 2003.93),
    (8, 11, 30, 10.0206, 10.0804, 10.0135, 10.0607, 2128.07),
    (8, 13, 30, 10.0413, 10.0835, 10.0409, 10.0707, 2420.16),
    (8, 14, 0, 10.0707, 10.0789, 10.0682, 10.0756, 2697.0),
    (8, 14, 30, 10.095, 10.0951, 10.0613, 10.0729, 2786.91),
    (8, 15, 0, 10.0833, 10.0893, 10.0254, 10.0446, 2659.7),
];

const STATS_JSON: &str = r#"{
  "ashare|30m|r_body": 0.005281821123909282,
  "ashare|30m|r_dn": 0.002439172305073014,
  "ashare|30m|r_gap": 0.004806835110591831,
  "ashare|30m|r_up": 0.0028039091113660993,
  "ashare|30m|v": 0.685340201639828
}"#;

const HL: f64 = 40.0;

fn ts(d: u32, h: u32, m: u32) -> Timestamp {
    NaiveDate::from_ymd_opt(2026, 1, d)
        .unwrap()
        .and_hms_opt(h, m, 0)
        .unwrap()
}

fn bars() -> Vec<Bar> {
    RAW.iter()
        .map(|&(d, h, m, o, hi, l, c, v)| Bar::new(ts(d, h, m), o, hi, l, c, v))
        .collect()
}

fn stats() -> Stats {
    Stats::from_json_str(STATS_JSON).unwrap()
}

fn table() -> TradingCalendar {
    let days = (5..=9)
        .chain(12..=16)
        .map(|d| NaiveDate::from_ymd_opt(2026, 1, d).unwrap());
    TradingCalendar::from_days(days)
}

fn near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "got {a}, want {b} (tol {tol})");
}

fn near_all(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        near(*x, *y, tol);
    }
}

#[test]
fn ema_alpha_and_validation() {
    near(ema_alpha(HL).unwrap(), 0.017179401454748944, 1e-15);
    near(ema_alpha(1.0).unwrap(), 0.5, 1e-15);
    for bad in [0.0, -1.0, f64::NAN] {
        assert!(matches!(
            ema_alpha(bad),
            Err(crate::Error::InvalidConfig(_))
        ));
    }
}

#[test]
fn volume_ema_matches_pandas() {
    let v: Vec<f64> = bars().iter().map(|b| b.volume).collect();
    let ema = compute_volume_ema(&v, HL).unwrap();
    near_all(
        &ema[..5],
        &[
            1000.0,
            1000.0,
            1004.8961294146035,
            1011.5489191222954,
            1016.1299968830507,
        ],
        1e-9,
    );
    near(*ema.last().unwrap(), 1347.6847370753546, 1e-9);
    assert!(compute_volume_ema(&[], HL).unwrap().is_empty());
    assert_eq!(compute_volume_ema(&[7.0], HL).unwrap(), vec![7.0]);
}

#[test]
fn features_match_upstream() {
    let f = compute_features(&bars(), HL).unwrap();
    near_all(
        &f.r_gap[..6],
        &[
            0.0,
            0.0019204654820650499,
            0.001030182453198303,
            -0.0013761765261668798,
            -0.0017686457582347477,
            0.00042916099685084443,
        ],
        1e-12,
    );
    near_all(
        &f.r_body[..6],
        &[
            0.003992021269537457,
            0.0030571653735326624,
            0.0006829012681330039,
            -0.0020231478740408296,
            -0.0037762777343045527,
            -0.0037488984669010557,
        ],
        1e-12,
    );
    near_all(
        &f.r_up[..6],
        &[
            0.0009955202415018415,
            0.0018416940865147867,
            0.0019076712928236556,
            0.0011386985412389415,
            0.00023865161948888458,
            3.99125914302128e-05,
        ],
        1e-12,
    );
    near_all(
        &f.r_dn[..6],
        &[
            0.0020020026706729687,
            0.0015320946926245587,
            0.0005843001686407757,
            9.927479760354985e-06,
            0.00034945310944563415,
            0.0012828480983195513,
        ],
        1e-12,
    );
    near_all(
        &f.v[..6],
        &[
            0.0,
            0.25053712548234414,
            0.325688554996053,
            0.23377186587898766,
            0.04902750161878496,
            -0.061026748743158866,
        ],
        1e-12,
    );
    let last = f.len() - 1;
    near(f.r_gap[last], 0.0010319406359247405, 1e-12);
    near(f.r_body[last], -0.0038454133504954814, 1e-12);
    near(f.r_up[last], 0.0005948663213401274, 1e-12);
    near(f.r_dn[last], 0.0019133040216416175, 1e-12);
    near(f.v[last], 0.6794593973594754, 1e-12);
    near(f.r_gap.iter().sum(), 0.0029093303394009144, 1e-12);
    near(f.v.iter().sum(), 13.821172305919312, 1e-9);
    assert!(f.r_up.iter().chain(&f.r_dn).all(|&x| x >= 0.0));
}

#[test]
fn invert_features_round_trips() {
    let b = bars();
    let f = compute_features(&b, HL).unwrap();
    let inv = invert_features(&f, b[0].open, b[0].volume, HL).unwrap();
    for (i, bar) in b.iter().enumerate() {
        near(inv.open[i], bar.open, 1e-10);
        near(inv.high[i], bar.high, 1e-10);
        near(inv.low[i], bar.low, 1e-10);
        near(inv.close[i], bar.close, 1e-10);
        near(inv.volume[i], bar.volume, 1e-8);
    }
    let bars_back = inv
        .to_bars(&b.iter().map(|x| x.timestamp).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(bars_back.len(), b.len());
    assert!(inv.to_bars(&[]).is_err());
}

#[test]
fn invert_clamps_negative_volume_and_checks_lengths() {
    let f = Features {
        r_gap: vec![0.0],
        r_body: vec![0.0],
        r_up: vec![0.0],
        r_dn: vec![0.0],
        v: vec![-50.0],
    };
    let inv = invert_features(&f, 10.0, 100.0, 5.0).unwrap();
    assert_eq!(inv.volume, vec![0.0]);
    let bad = Features { v: vec![], ..f };
    assert!(matches!(
        invert_features(&bad, 10.0, 100.0, 5.0),
        Err(crate::Error::InvalidData(_))
    ));
}

#[test]
fn normalize_matches_upstream() {
    let cases = [
        (0.0, 0.0),
        (0.001, 0.2498554244729157),
        (-0.02, -4.093570741429018),
        (0.5, 5.999999994),
        (100.0, 5.999999994),
        (-100.0, -5.999999994),
    ];
    for (x, want) in cases {
        near(normalize(x, 0.004, DEFAULT_SOFT_CLIP_C), want, 1e-12);
    }
    near(denormalize(0.2498554244729157, 0.004, 6.0), 0.001, 1e-12);
    near(denormalize(-4.093570741429018, 0.004, 6.0), -0.02, 1e-12);
    near(denormalize(6.0, 0.004, 6.0), 0.17410388686194406, 1e-12);
    near(denormalize(-7.0, 0.004, 6.0), -0.17410388686194406, 1e-12);
    near(denormalize(3.0, 2.0, 6.0), 6.591673732008658, 1e-12);
    near(normalize(0.5, 0.004, 6.0), 6.0 * (1.0 - 1e-9), 1e-15);
    assert!(normalize(f64::NAN, 1.0, 6.0).is_nan());
    let v = [0.001, -0.02];
    near_all(
        &denormalize_slice(&normalize_slice(&v, 0.004, 6.0), 0.004, 6.0),
        &v,
        1e-12,
    );
}

#[test]
fn mad_scale_matches_upstream() {
    near(
        fit_mad_scale(&[1.0, 2.0, 3.0, 4.0, 100.0]).unwrap(),
        1.4826,
        1e-12,
    );
    near(
        fit_mad_scale(&[1.0, 1.0, 1.0, 1.0, 5.0]).unwrap(),
        1.6,
        1e-12,
    );
    near(fit_mad_scale(&[2.0, 2.0, 2.0]).unwrap(), 1e-8, 1e-20);
    near(fit_mad_scale(&[1.0, 2.0, 4.0, 8.0]).unwrap(), 2.2239, 1e-12);
    assert!(matches!(
        fit_mad_scale(&[]),
        Err(crate::Error::InvalidData(_))
    ));
}

#[test]
fn stats_json_parsing_and_lookup() {
    let s = stats();
    assert_eq!(s.len(), 5);
    assert_eq!(s.get("ashare|30m|v"), Some(0.685340201639828));
    let scales = s.feature_scales(Market::Ashare, Timescale::Min30).unwrap();
    assert_eq!(scales[0], 0.004806835110591831);
    assert_eq!(scales[4], 0.685340201639828);
    assert!(matches!(
        s.scale(Market::Us, Timescale::Min30, "v"),
        Err(crate::Error::InvalidData(_))
    ));
    assert!(matches!(
        Stats::from_json_str("{"),
        Err(crate::Error::Json(_))
    ));
    assert!(matches!(
        Stats::from_json_str("[1]"),
        Err(crate::Error::InvalidData(_))
    ));
    assert!(matches!(
        Stats::from_json_str(r#"{"a": null}"#),
        Err(crate::Error::InvalidData(_))
    ));
    assert_eq!(
        Stats::from_json_str(r#"{"a": "1.5"}"#).unwrap().get("a"),
        Some(1.5)
    );
    assert_eq!(
        stats_key(Market::Hk, Timescale::Day1, "r_gap"),
        "hk|1d|r_gap"
    );
}

#[test]
fn stats_json_path_round_trip() {
    let dir = std::env::temp_dir().join(format!("kit_forecast_stats_{}", std::process::id()));
    let path = dir.join("nested").join("stats.json");
    let s = stats();
    s.save_json_path(&path).unwrap();
    assert_eq!(Stats::from_json_path(&path).unwrap(), s);
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(matches!(
        Stats::from_json_path(dir.join("missing.json")),
        Err(crate::Error::Io(_))
    ));
}

fn sum32(rows: &[[f32; 5]]) -> f64 {
    rows.iter().flatten().map(|&x| f64::from(x)).sum()
}

#[test]
fn prepare_full_context_matches_upstream() {
    let w = PreparedWindow::prepare(
        &bars(),
        Market::Ashare,
        Timescale::Min30,
        20,
        5,
        &stats(),
        3,
        77,
        &table(),
        6.0,
        5.0,
    )
    .unwrap();
    assert_eq!(w.n_pad, 0);
    assert_eq!((w.lc(), w.lh(), w.len()), (20, 5, 25));
    near(w.prev_close, 10.0446, 1e-12);
    near(w.prev_ema, 1370.2243739918952, 1e-9);
    near(w.half_life_bars, 40.0, 0.0);
    let first = w.x_ctx[0];
    let want_first = [
        0.04529934749007225,
        -0.3933267593383789,
        0.16581042110919952,
        0.7529546618461609,
        0.3016558885574341,
    ];
    for j in 0..5 {
        near(f64::from(first[j]), want_first[j], 1e-6);
    }
    let last = w.x_ctx[19];
    let want_last = [
        0.21459035575389862,
        -0.7244946360588074,
        0.21206769347190857,
        0.7799685001373291,
        0.9824936985969543,
    ];
    for j in 0..5 {
        near(f64::from(last[j]), want_last[j], 1e-6);
    }
    near(sum32(&w.x_ctx), 30.29399688099511, 1e-4);
    assert_eq!(
        w.future_timestamps,
        vec![
            ts(9, 10, 0),
            ts(9, 10, 30),
            ts(9, 11, 0),
            ts(9, 11, 30),
            ts(9, 13, 30)
        ]
    );
    assert_eq!(w.ctx_timestamps.len(), 20);
    assert_eq!(w.ctx_timestamps[0], ts(6, 13, 30));
    assert_eq!(*w.ctx_timestamps.last().unwrap(), ts(8, 15, 0));
    assert_eq!(w.cond.event, vec![0; 25]);
    assert_eq!(
        w.cond.dow,
        vec![
            1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4
        ]
    );
    assert_eq!(w.cond.month, vec![0; 25]);
    assert_eq!(w.cond.pad_mask, vec![true; 25]);
    assert_eq!(
        (w.cond.market, w.cond.sector, w.cond.inst, w.cond.scale),
        (1, 3, 77, 4)
    );
    let sess: Vec<f32> = w.cond.sess.iter().flatten().copied().collect();
    let pattern = [0.0, 0.75, 0.25, 0.5, 0.5, 0.25, 0.75, 0.0];
    let want_sess: Vec<f32> = (0..25)
        .flat_map(|i| {
            let p = &pattern[(i % 4) * 2..(i % 4) * 2 + 2];
            [p[0], p[1]]
        })
        .collect();
    assert_eq!(sess.len(), 50);
    assert_eq!(sess, want_sess);
    let clock_sum: f64 = w.cond.clock.iter().flatten().map(|&x| f64::from(x)).sum();
    let yday_sum: f64 = w.cond.yday.iter().flatten().map(|&x| f64::from(x)).sum();
    near(clock_sum, -18.37429004907607, 1e-4);
    near(yday_sum, 122.77519112825394, 1e-4);
}

#[test]
fn prepare_pads_front_and_marks_first_bar() {
    let w = PreparedWindow::prepare(
        &bars(),
        Market::Ashare,
        Timescale::Min30,
        40,
        4,
        &stats(),
        3,
        77,
        &table(),
        6.0,
        5.0,
    )
    .unwrap();
    assert_eq!(w.n_pad, 8);
    assert_eq!(w.ctx_timestamps.len(), 32);
    assert_eq!(w.ctx_timestamps[0], ts(5, 10, 0));
    assert!(w.x_ctx[..8].iter().all(|r| *r == [0.0; 5]));
    let first = w.x_ctx[8];
    let want_first = [
        0.0,
        0.7518315315246582,
        0.3546334207057953,
        0.8156896829605103,
        0.0,
    ];
    for j in 0..5 {
        near(f64::from(first[j]), want_first[j], 1e-6);
    }
    near(sum32(&w.x_ctx), 45.55885400553234, 1e-4);
    let mut want_event = vec![0i64; 44];
    want_event[8] = 2;
    assert_eq!(w.cond.event, want_event);
    let mut want_pad = vec![true; 44];
    want_pad[..8].iter_mut().for_each(|m| *m = false);
    assert_eq!(w.cond.pad_mask, want_pad);
    assert!(w.cond.sess[..8].iter().all(|s| *s == [0.0; 2]));
    assert!(w.cond.clock[..8].iter().all(|s| *s == [0.0; 16]));
    assert!(w.cond.yday[..8].iter().all(|s| *s == [0.0; 8]));
    assert_eq!(&w.cond.dow[..10], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(&w.cond.dow[16..24], &[1; 8]);
    assert_eq!(w.cond.dow.len(), 44);
    let clock_sum: f64 = w.cond.clock.iter().flatten().map(|&x| f64::from(x)).sum();
    let yday_sum: f64 = w.cond.yday.iter().flatten().map(|&x| f64::from(x)).sum();
    near(clock_sum, -25.537879854440682, 1e-4);
    near(yday_sum, 173.5783839225769, 1e-4);
    near(w.prev_ema, 1370.2243739918952, 1e-9);
}

#[test]
fn prepare_truncated_context_event_uses_prior_bar() {
    let w = PreparedWindow::prepare(
        &bars(),
        Market::Ashare,
        Timescale::Min30,
        10,
        3,
        &stats(),
        0,
        0,
        &table(),
        6.0,
        5.0,
    )
    .unwrap();
    assert_eq!(w.n_pad, 0);
    assert_eq!(w.cond.event, vec![0; 13]);
    assert_eq!(w.ctx_timestamps[0], ts(7, 14, 30));
    assert_eq!(
        w.future_timestamps,
        vec![ts(9, 10, 0), ts(9, 10, 30), ts(9, 11, 0)]
    );
    let want_first = [
        -0.1341225653886795,
        -0.7193536758422852,
        0.35332798957824707,
        0.0,
        0.9965965747833252,
    ];
    for (got, want) in w.x_ctx[0].iter().zip(want_first) {
        near(f64::from(*got), want, 1e-6);
    }
    near(sum32(&w.x_ctx), 15.235810914309695, 1e-4);
}

#[test]
fn prepare_on_truncated_history_continues_after_cut() {
    let b = bars();
    let w = PreparedWindow::prepare(
        &b[..20],
        Market::Ashare,
        Timescale::Min30,
        8,
        3,
        &stats(),
        0,
        0,
        &table(),
        6.0,
        5.0,
    )
    .unwrap();
    assert_eq!(w.ctx_timestamps[0], ts(6, 13, 30));
    assert_eq!(
        w.future_timestamps,
        vec![ts(7, 13, 30), ts(7, 14, 0), ts(7, 14, 30)]
    );
    assert_eq!(w.cond.event, vec![0; 11]);
    near(w.prev_ema, 1149.678017393899, 1e-9);
    near(w.prev_close, 10.0677, 1e-12);
}

#[test]
fn prepare_rejects_bad_input() {
    let cal = table();
    let st = stats();
    let b = bars();
    let err = PreparedWindow::prepare(
        &b[..1],
        Market::Ashare,
        Timescale::Min30,
        4,
        2,
        &st,
        0,
        0,
        &cal,
        6.0,
        5.0,
    );
    assert!(matches!(err, Err(crate::Error::InvalidData(_))));
    let err = PreparedWindow::prepare(
        &b,
        Market::Us,
        Timescale::Min30,
        4,
        2,
        &st,
        0,
        0,
        &cal,
        6.0,
        5.0,
    );
    assert!(matches!(err, Err(crate::Error::InvalidData(_))));
    let err = PreparedWindow::prepare(
        &b,
        Market::Ashare,
        Timescale::Min30,
        4,
        2,
        &st,
        0,
        0,
        &cal,
        6.0,
        0.0,
    );
    assert!(matches!(err, Err(crate::Error::InvalidConfig(_))));
}

fn golden_paths() -> Vec<Vec<[f32; 5]>> {
    let raw: [[[f32; 5]; 5]; 4] = [
        [
            [3.061, -3.833, 0.627, -0.852, -0.679],
            [-0.323, -3.03, -0.348, -1.298, 4.984],
            [0.339, -0.529, -0.422, -1.002, -1.583],
            [-0.586, 0.723, -0.358, 1.437, -0.3],
            [0.036, 2.319, 0.818, -0.758, -0.274],
        ],
        [
            [0.811, 2.903, -0.404, -0.365, 1.503],
            [-1.33, -0.438, 1.324, 0.871, 0.137],
            [1.005, -4.242, 1.532, -1.439, -2.503],
            [0.415, 1.051, -0.667, -1.615, 0.039],
            [-0.079, 2.108, 1.121, 0.291, 1.667],
        ],
        [
            [-0.308, -1.389, 0.876, 0.874, -0.322],
            [-1.174, 0.344, -3.741, 1.035, 0.737],
            [-2.458, 0.092, -1.446, 1.136, -3.051],
            [-1.372, 1.064, 1.735, -3.237, -0.747],
            [0.492, -0.914, 2.386, -1.787, 0.532],
        ],
        [
            [-1.573, 2.109, -0.032, -0.558, -2.577],
            [2.523, 1.129, 1.13, 1.707, 0.524],
            [-0.959, -1.2, -1.2, 2.055, -2.191],
            [-0.895, -0.482, 0.337, 0.863, -1.874],
            [-2.595, -0.007, 1.82, 1.136, 0.323],
        ],
    ];
    raw.iter().map(|p| p.to_vec()).collect()
}

#[test]
fn denormalize_paths_matches_upstream() {
    let den = denormalize_paths(
        &golden_paths(),
        Market::Ashare,
        Timescale::Min30,
        &stats(),
        6.0,
    )
    .unwrap();
    near_all(
        &den[0][0],
        &[
            0.016236193332617603,
            -0.023964683907119477,
            0.0017644926520583796,
            -0.0020923143692613264,
            -0.46734793093209615,
        ],
        1e-6,
    );
    near_all(
        &den[3][4],
        &[
            -0.013352460856844326,
            -3.697276578327925e-05,
            0.005268881638400396,
            0.002804740303916109,
            0.22157910776390252,
        ],
        1e-6,
    );
    let d2 = denormalize_paths_default(&golden_paths(), Market::Ashare, Timescale::Min30, &stats())
        .unwrap();
    assert_eq!(den, d2);
    assert!(
        denormalize_paths(&golden_paths(), Market::Us, Timescale::Min30, &stats(), 6.0).is_err()
    );
}

fn ohlcv_paths() -> Vec<Ohlcv> {
    let den = denormalize_paths(
        &golden_paths(),
        Market::Ashare,
        Timescale::Min30,
        &stats(),
        6.0,
    )
    .unwrap();
    paths_to_ohlcv(&den, 10.0, 1200.0, 20.0).unwrap()
}

#[test]
fn paths_to_ohlcv_matches_upstream() {
    let oh = ohlcv_paths();
    assert_eq!(oh.len(), 4);
    let p = &oh[1];
    near_all(
        &p.open,
        &[
            10.039300503375445,
            10.142540683036097,
            10.168531364755998,
            9.908282642987835,
            9.960231508761698,
        ],
        1e-6,
    );
    near_all(
        &p.high,
        &[
            10.208692144108023,
            10.180895598717141,
            10.213297634327601,
            9.964014743896772,
            10.108839976618327,
        ],
        1e-6,
    );
    near_all(
        &p.low,
        &[
            10.039300503375445,
            10.097434006089745,
            9.888505392934512,
            9.908282642987835,
            9.953158706068272,
        ],
        1e-6,
    );
    near_all(
        &p.close,
        &[
            10.208692144108023,
            10.119061951170135,
            9.888505392934512,
            9.964014743896772,
            10.076739476447521,
        ],
        1e-6,
    );
    near_all(
        &p.volume,
        &[
            3439.4970345206552,
            1402.045233978128,
            205.2395566726649,
            1277.6638928978246,
            4027.157033000715,
        ],
        1e-3,
    );
    for p in &oh {
        for t in 0..p.len() {
            assert!(p.high[t] >= p.open[t].max(p.close[t]) - 1e-12);
            assert!(p.low[t] <= p.open[t].min(p.close[t]) + 1e-12);
            assert!(p.volume[t] >= 0.0);
        }
    }
}

#[test]
fn paths_to_ohlcv_clamps_negative_shadows() {
    let paths = vec![vec![[0.0, 0.0, -0.5, -0.5, 0.0]]];
    let oh = paths_to_ohlcv(&paths, 10.0, 100.0, 5.0).unwrap();
    near(oh[0].high[0], 10.0, 1e-12);
    near(oh[0].low[0], 10.0, 1e-12);
}

#[test]
fn limit_projection_matches_upstream() {
    let oh = ohlcv_paths();
    let lim = apply_limit_projection(&oh[1], 10.0, Some(0.01));
    near_all(
        &lim.open,
        &[
            10.039300503375445,
            10.142540683036097,
            10.168531364755998,
            9.91769261834185,
            9.960231508761698,
        ],
        1e-6,
    );
    near_all(
        &lim.high,
        &[
            10.1,
            10.180895598717141,
            10.213297634327601,
            9.964014743896772,
            10.06365489133574,
        ],
        1e-6,
    );
    near_all(
        &lim.low,
        &[
            10.039300503375445,
            10.097434006089745,
            10.017871331658434,
            9.91769261834185,
            9.953158706068272,
        ],
        1e-6,
    );
    near_all(
        &lim.close,
        &[
            10.1,
            10.119061951170135,
            10.017871331658434,
            9.964014743896772,
            10.06365489133574,
        ],
        1e-6,
    );
    assert_eq!(lim.volume, oh[1].volume);
    assert_eq!(apply_limit_projection(&oh[1], 10.0, None), oh[1]);
}

fn dist_close(d: &Dist, samples: &[f64], q: [f64; 5], tol: f64) {
    near_all(&d.samples, samples, tol);
    near_all(&[d.q05, d.q25, d.q50, d.q75, d.q95], &q, tol);
}

#[test]
fn summarize_ohlcv_paths_matches_upstream() {
    let s = summarize_ohlcv_paths(&ohlcv_paths(), Some(10.0)).unwrap();
    dist_close(
        &s.terminal_return,
        &[
            -0.013968350551098609,
            0.007644652685063865,
            -0.028446585851207032,
            -0.008491579426893155,
        ],
        [
            -0.02627485055619077,
            -0.017587909376125713,
            -0.011229964988995882,
            -0.0044575213989039,
            0.005224217868270306,
        ],
        1e-9,
    );
    dist_close(
        &s.max_drawdown,
        &[
            0.028072706753866007,
            0.031866517235489056,
            0.028446585851207032,
            0.03134963289834802,
        ],
        [
            0.02812878861846716,
            0.028353116076871777,
            0.029898109374777528,
            0.03147885398263328,
            0.0317889845849179,
        ],
        1e-9,
    );
    near_all(
        &s.realized_vol.samples,
        &[
            0.011855022502333506,
            0.017380415323758402,
            0.00464766440035742,
            0.013320322388360506,
        ],
        1e-12,
    );
    near_all(
        &[
            s.realized_vol.q05,
            s.realized_vol.q25,
            s.realized_vol.q50,
            s.realized_vol.q75,
            s.realized_vol.q95,
        ],
        &[
            0.005728768115653833,
            0.010053182976839485,
            0.012587672445347094,
            0.014335345622210113,
            0.016771401383448743,
        ],
        1e-8,
    );
    near_all(
        &s.close_return_bands.q05,
        &[
            -0.008769917830735066,
            -0.017620780630251454,
            -0.021400654601207082,
            -0.006007990048484691,
            -0.011755294087273294,
        ],
        1e-9,
    );
    near_all(
        &s.close_return_bands.q50,
        &[
            -0.0019174648416421408,
            -0.006358281699787582,
            -0.011570176387252662,
            -1.1089025942601083e-05,
            0.004377249255007376,
        ],
        1e-9,
    );
    let names: Vec<&str> = s.channel_bands.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["open", "high", "low", "close", "volume"]);
    near_all(
        &s.channel_bands[3].1.q95,
        &[
            10.183240048461748,
            10.214390348816512,
            10.084029275330533,
            10.036336092049643,
            10.052545113279926,
        ],
        1e-6,
    );
    near_all(
        &s.channel_bands[4].1.q25,
        &[
            608.889021231671,
            1603.5519081389707,
            184.08081922500327,
            605.2834782176378,
            1609.9010451993904,
        ],
        1e-3,
    );
}

#[test]
fn summarize_ohlcv_without_prev_close_uses_open() {
    let s = summarize_ohlcv_paths(&ohlcv_paths(), None).unwrap();
    near_all(
        &s.terminal_return.samples,
        &[
            -0.0302045438837164,
            0.003722304821203115,
            -0.02696477815332745,
            -0.0007496842605253344,
        ],
        1e-9,
    );
}

#[test]
fn summarize_feature_paths_matches_upstream() {
    let den = denormalize_paths(
        &golden_paths(),
        Market::Ashare,
        Timescale::Min30,
        &stats(),
        6.0,
    )
    .unwrap();
    let s = summarize_feature_paths(&den).unwrap();
    dist_close(
        &s.terminal_return,
        &[
            -0.013968350551098855,
            0.007644652685064508,
            -0.02844658585120725,
            -0.008491579426892862,
        ],
        [
            s.terminal_return.q05,
            s.terminal_return.q25,
            -0.01122996498899586,
            s.terminal_return.q75,
            s.terminal_return.q95,
        ],
        1e-6,
    );
    near_all(
        &s.max_drawdown.samples,
        &[
            0.0280727067538665,
            0.031866517235488605,
            0.02844658585120725,
            0.03134963289834766,
        ],
        1e-6,
    );
    near_all(
        &s.realized_vol.samples,
        &[
            0.011855022502333633,
            0.017380415323758346,
            0.004647664400357508,
            0.013320322388360594,
        ],
        1e-6,
    );
    near_all(
        &s.channel_bands[4].1.q75,
        &[
            0.09744652350876185,
            1.6043679156415807,
            -1.4584670203272392,
            -0.1476480846843237,
            0.5674956399922453,
        ],
        1e-6,
    );
    near_all(
        &s.close_return_bands.q95,
        &[
            0.018140304219953343,
            0.015535118393067813,
            -0.002655748108265267,
            0.006617744422244621,
            0.012816302377901122,
        ],
        1e-6,
    );
    let names: Vec<&str> = s.channel_bands.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, FEATURE_NAMES.to_vec());
}

#[test]
fn single_bar_horizon_has_zero_volatility() {
    let one: Vec<Ohlcv> = ohlcv_paths()
        .into_iter()
        .map(|p| Ohlcv {
            open: p.open[..1].to_vec(),
            high: p.high[..1].to_vec(),
            low: p.low[..1].to_vec(),
            close: p.close[..1].to_vec(),
            volume: p.volume[..1].to_vec(),
        })
        .collect();
    let s = summarize_ohlcv_paths(&one, Some(10.0)).unwrap();
    assert_eq!(s.realized_vol.samples, vec![0.0; 4]);
    near_all(
        &s.terminal_return.samples,
        &[
            -0.007728490574501681,
            0.02065443539561251,
            -0.00895369911124684,
            0.0038935608912173997,
        ],
        1e-9,
    );
}

#[test]
fn summarize_rejects_empty_and_ragged_input() {
    assert!(summarize_feature_paths(&[]).is_err());
    assert!(summarize_ohlcv_paths(&[], None).is_err());
    let ragged = vec![vec![[0.0; 5]; 2], vec![[0.0; 5]; 3]];
    assert!(matches!(
        summarize_feature_paths(&ragged),
        Err(crate::Error::InvalidData(_))
    ));
}

#[test]
fn quantile_matches_numpy_linear_interpolation() {
    let v = [3.0, 1.0, 4.0, 1.5];
    near(quantile(&v, 0.0).unwrap(), 1.0, 1e-15);
    near(quantile(&v, 0.5).unwrap(), 2.25, 1e-15);
    near(quantile(&v, 1.0).unwrap(), 4.0, 1e-15);
    near(quantile(&v, 0.05).unwrap(), 1.075, 1e-12);
    near(quantile(&v, 0.95).unwrap(), 3.85, 1e-12);
    assert!(quantile(&[1.0, f64::NAN], 0.5).unwrap().is_nan());
    assert!(quantile(&[], 0.5).is_err());
}
