use chrono::NaiveDate;

use super::*;

fn ts(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> Timestamp {
    NaiveDate::from_ymd_opt(y, mo, d)
        .unwrap()
        .and_hms_opt(h, mi, 0)
        .unwrap()
}

fn date(y: i32, mo: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, mo, d).unwrap()
}

fn close(a: f32, b: f64, tol: f64) -> bool {
    (f64::from(a) - b).abs() <= tol
}

fn assert_vec_close(got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w, tol), "index {i}: got {g}, want {w}");
    }
}

const HOLIDAY_CSV: &str = "\u{feff}trade_date\n20250925\n20250926\n20250929\n20250930\n\
20251009\n20251010\n20251013\n20251014\n20251015\n20251016\n20251017\n20251020\n20251021\n\
20251022\n20251023\n20251024\n20260105\n20260106\n20260107\n20260108\n20260109\n20260112\n\
20260113\n20260401\n20260402\n20260403\n20260407\n20260408\n20260409\n20260410\n";

fn table() -> TradingCalendar {
    TradingCalendar::from_csv_str(HOLIDAY_CSV)
}

#[test]
fn csv_parsing_skips_header_bom_and_bad_rows() {
    let cal = TradingCalendar::from_csv_str(
        "\u{feff}trade_date\r\n20240103\r\n\r\nbad\r\n20240102,x\r\n20241301\r\n",
    );
    assert_eq!(cal.days().unwrap(), &[date(2024, 1, 2), date(2024, 1, 3)]);
    assert!(!cal.is_weekends_only());
    assert!(TradingCalendar::from_csv_str("trade_date\n").is_weekends_only());
    assert!(TradingCalendar::from_csv_str("").is_weekends_only());
}

#[test]
fn csv_path_missing_file_is_weekends_only() {
    let dir = std::env::temp_dir().join("kit_forecast_market_missing_dir");
    let cal = TradingCalendar::from_csv_path(dir.join("nope.csv")).unwrap();
    assert!(cal.is_weekends_only());
}

#[test]
fn csv_path_reads_market_then_root_table() {
    let root = std::env::temp_dir().join(format!("kit_forecast_cal_{}", std::process::id()));
    let market_dir = root.join("ashare");
    std::fs::create_dir_all(&market_dir).unwrap();
    std::fs::write(root.join("trade_dates.csv"), "trade_date\n20240102\n").unwrap();

    let cal = TradingCalendar::load_market(&root, Market::Ashare);
    assert_eq!(cal.days().unwrap(), &[date(2024, 1, 2)]);

    std::fs::write(
        market_dir.join("trade_dates.csv"),
        "trade_date\n20240103\n20240104\n",
    )
    .unwrap();
    let cal = TradingCalendar::load_market(&root, Market::Ashare);
    assert_eq!(cal.days().unwrap(), &[date(2024, 1, 3), date(2024, 1, 4)]);

    let cal = TradingCalendar::load_market(&root, Market::Us);
    assert_eq!(cal.days().unwrap(), &[date(2024, 1, 2)]);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn trading_day_rules_follow_table_then_weekdays() {
    let cal = table();
    assert!(!cal.is_trading_day(date(2025, 10, 1)));
    assert!(cal.is_trading_day(date(2025, 10, 9)));
    assert!(!cal.is_trading_day(date(2025, 10, 11)));
    assert!(cal.is_trading_day(date(2026, 4, 13)));
    assert!(!cal.is_trading_day(date(2026, 4, 11)));
    assert!(!cal.is_trading_day(date(2026, 4, 12)));
    let wk = TradingCalendar::weekends_only();
    assert!(wk.is_trading_day(date(2025, 10, 1)));
    assert!(!wk.is_trading_day(date(2025, 10, 4)));
}

#[test]
fn future_timestamps_ashare_30m_across_weekend() {
    let want = vec![
        ts(2026, 4, 10, 15, 0),
        ts(2026, 4, 13, 10, 0),
        ts(2026, 4, 13, 10, 30),
        ts(2026, 4, 13, 11, 0),
        ts(2026, 4, 13, 11, 30),
        ts(2026, 4, 13, 13, 30),
    ];
    for cal in [table(), TradingCalendar::weekends_only()] {
        let got = gen_future_timestamps(
            ts(2026, 4, 10, 14, 30),
            6,
            Market::Ashare,
            Timescale::Min30,
            &cal,
        );
        assert_eq!(got, want);
    }
}

#[test]
fn future_timestamps_skip_holidays_only_with_table() {
    let last = ts(2025, 9, 30, 15, 0);
    let got = gen_future_timestamps(last, 3, Market::Ashare, Timescale::Day1, &table());
    assert_eq!(
        got,
        vec![
            ts(2025, 10, 9, 15, 0),
            ts(2025, 10, 10, 15, 0),
            ts(2025, 10, 13, 15, 0)
        ]
    );
    let got = gen_future_timestamps(
        last,
        3,
        Market::Ashare,
        Timescale::Day1,
        &TradingCalendar::weekends_only(),
    );
    assert_eq!(
        got,
        vec![
            ts(2025, 10, 1, 15, 0),
            ts(2025, 10, 2, 15, 0),
            ts(2025, 10, 3, 15, 0)
        ]
    );
}

#[test]
fn future_timestamps_crypto_skips_weekend_days_but_keeps_midnight_close() {
    let cal = TradingCalendar::weekends_only();
    let got = gen_future_timestamps(
        ts(2026, 1, 2, 22, 0),
        6,
        Market::Crypto,
        Timescale::Hour1,
        &cal,
    );
    assert_eq!(
        got,
        vec![
            ts(2026, 1, 2, 23, 0),
            ts(2026, 1, 3, 0, 0),
            ts(2026, 1, 5, 1, 0),
            ts(2026, 1, 5, 2, 0),
            ts(2026, 1, 5, 3, 0),
            ts(2026, 1, 5, 4, 0),
        ]
    );
    let got = gen_future_timestamps(
        ts(2026, 1, 3, 5, 0),
        3,
        Market::Crypto,
        Timescale::Hour1,
        &cal,
    );
    assert_eq!(
        got,
        vec![
            ts(2026, 1, 5, 1, 0),
            ts(2026, 1, 5, 2, 0),
            ts(2026, 1, 5, 3, 0)
        ]
    );
    let got = gen_future_timestamps(
        ts(2026, 1, 2, 23, 0),
        3,
        Market::Crypto,
        Timescale::Day1,
        &cal,
    );
    assert_eq!(
        got,
        vec![
            ts(2026, 1, 3, 0, 0),
            ts(2026, 1, 6, 0, 0),
            ts(2026, 1, 7, 0, 0)
        ]
    );
}

#[test]
fn future_timestamps_us_2h_and_hk_1h_use_short_stubs() {
    let cal = TradingCalendar::weekends_only();
    let got = gen_future_timestamps(ts(2026, 1, 2, 16, 0), 5, Market::Us, Timescale::Hour2, &cal);
    assert_eq!(
        got,
        vec![
            ts(2026, 1, 5, 11, 30),
            ts(2026, 1, 5, 13, 30),
            ts(2026, 1, 5, 15, 30),
            ts(2026, 1, 5, 16, 0),
            ts(2026, 1, 6, 11, 30),
        ]
    );
    let got = gen_future_timestamps(
        ts(2026, 1, 2, 11, 30),
        5,
        Market::Hk,
        Timescale::Hour1,
        &cal,
    );
    assert_eq!(
        got,
        vec![
            ts(2026, 1, 2, 12, 0),
            ts(2026, 1, 2, 14, 0),
            ts(2026, 1, 2, 15, 0),
            ts(2026, 1, 2, 16, 0),
            ts(2026, 1, 5, 10, 30),
        ]
    );
}

#[test]
fn future_timestamps_zero_count_is_empty() {
    let cal = TradingCalendar::weekends_only();
    assert!(
        gen_future_timestamps(
            ts(2026, 1, 2, 15, 0),
            0,
            Market::Ashare,
            Timescale::Min30,
            &cal
        )
        .is_empty()
    );
}

#[test]
fn calendar_features_ashare_30m() {
    let stamps = [
        ts(2026, 1, 5, 10, 0),
        ts(2026, 1, 5, 11, 30),
        ts(2026, 1, 5, 13, 30),
        ts(2026, 1, 5, 15, 0),
        ts(2026, 1, 5, 9, 45),
        ts(2026, 1, 6, 12, 10),
        ts(2024, 12, 31, 15, 0),
    ];
    let f = calendar_features(&stamps, Market::Ashare, Timescale::Min30);
    assert_eq!(f.len(), 7);
    assert_eq!(f.dow, vec![0, 0, 0, 0, 0, 1, 1]);
    assert_eq!(f.month, vec![0, 0, 0, 0, 0, 0, 11]);
    let sess: Vec<f32> = f.sess.iter().flatten().copied().collect();
    assert_eq!(
        sess,
        vec![
            0.0, 0.75, 0.75, 0.0, 0.0, 0.75, 0.75, 0.0, 0.0, 0.75, 0.75, 0.0, 0.75, 0.0
        ]
    );
    assert_vec_close(
        &f.clock[0],
        &[
            0.5,
            -0.8660253882408142,
            -0.8660253882408142,
            0.5,
            1.0,
            3.0616169991140216e-16,
            -0.8660253882408142,
            -0.5,
            0.5,
            0.8660253882408142,
            6.123233998228043e-16,
            -1.0,
            -0.5,
            0.8660253882408142,
            0.8660253882408142,
            -0.5,
        ],
        1e-6,
    );
    assert_vec_close(
        &f.yday[0],
        &[
            0.06880242377519608,
            0.9976302981376648,
            0.13727876543998718,
            0.9905324578285217,
            0.2051045000553131,
            0.9787400960922241,
            0.2719581723213196,
            0.9623090624809265,
        ],
        1e-6,
    );
    assert_vec_close(
        &f.yday[6],
        &[
            -0.01716632954776287,
            0.9998526573181152,
            -0.03432760015130043,
            0.9994106292724609,
            -0.051478754729032516,
            0.9986740946769714,
            -0.06861473619937897,
            0.997643232345581,
        ],
        1e-6,
    );
    let clock_sum: f64 = f.clock.iter().flatten().map(|&x| f64::from(x)).sum();
    let yday_sum: f64 = f.yday.iter().flatten().map(|&x| f64::from(x)).sum();
    assert!((clock_sum - -1.4541066214442244).abs() < 1e-5);
    assert!((yday_sum - 31.62558969296515).abs() < 1e-5);
}

#[test]
fn calendar_features_crypto_midnight_belongs_to_previous_day() {
    let stamps = [
        ts(2026, 3, 1, 0, 0),
        ts(2026, 3, 1, 1, 0),
        ts(2026, 3, 1, 23, 0),
        ts(2024, 3, 1, 0, 0),
        ts(2024, 1, 1, 0, 0),
    ];
    let f = calendar_features(&stamps, Market::Crypto, Timescale::Hour1);
    assert_eq!(f.dow, vec![5, 6, 6, 3, 6]);
    assert_eq!(f.month, vec![1, 2, 2, 1, 11]);
    let sess: Vec<f32> = f.sess.iter().flatten().copied().collect();
    let want = [
        0.9583333134651184,
        0.0,
        0.0,
        0.9583333134651184,
        0.9166666865348816,
        0.0416666679084301,
        0.9583333134651184,
        0.0,
        0.9583333134651184,
        0.0,
    ];
    assert_vec_close(&sess, &want, 1e-7);
    for k in 0..8 {
        assert!(close(f.clock[0][2 * k], 0.0, 1e-6));
        assert!(close(f.clock[0][2 * k + 1], 1.0, 1e-6));
    }
    assert_vec_close(
        &f.yday[0],
        &[
            0.8406184315681458,
            0.541627824306488,
            0.9106046557426453,
            -0.41327860951423645,
            0.1457991898059845,
            -0.9893141984939575,
            -0.7526668310165405,
            -0.658401608467102,
        ],
        1e-6,
    );
    assert_vec_close(
        &f.yday[3],
        &[
            0.8483511805534363,
            0.5294339060783386,
            0.8982917666435242,
            -0.43939951062202454,
            0.1028209999203682,
            -0.9946998953819275,
            -0.7894179224967957,
            -0.6138561367988586,
        ],
        1e-6,
    );
    assert_vec_close(
        &f.yday[4],
        &[
            -0.017213355749845505,
            0.9998518228530884,
            -0.034421611577272415,
            0.9994074106216431,
            -0.05161966755986214,
            0.9986668229103088,
            -0.06880242377519608,
            0.9976302981376648,
        ],
        1e-6,
    );
}

#[test]
fn calendar_features_daily_scale() {
    let f = calendar_features(
        &[ts(2026, 7, 4, 16, 0), ts(2026, 12, 31, 16, 0)],
        Market::Us,
        Timescale::Day1,
    );
    assert_eq!(f.dow, vec![5, 3]);
    assert_eq!(f.month, vec![6, 11]);
    assert_eq!(f.sess, vec![[0.5, 0.5], [0.5, 0.5]]);
    assert_vec_close(
        &f.clock[0],
        &[
            -0.8660253882408142,
            -0.5,
            0.8660253882408142,
            -0.5,
            -4.898587410340671e-16,
            1.0,
            -0.8660253882408142,
            -0.5,
            0.8660253882408142,
            -0.5,
            -9.797174820681343e-16,
            1.0,
            -0.8660253882408142,
            -0.5,
            0.8660253882408142,
            -0.5,
        ],
        1e-6,
    );
    assert_vec_close(
        &f.yday[1],
        &[
            -0.017213355749845505,
            0.9998518228530884,
            -0.034421611577272415,
            0.9994074106216431,
            -0.05161966755986214,
            0.9986668229103088,
            -0.06880242377519608,
            0.9976302981376648,
        ],
        1e-6,
    );

    let f = calendar_features(&[ts(2026, 7, 4, 0, 0)], Market::Crypto, Timescale::Day1);
    assert_eq!(f.dow, vec![4]);
    assert_eq!(f.month, vec![6]);
    assert_eq!(f.sess, vec![[0.5, 0.5]]);
    assert!(close(f.clock[0][0], 0.0, 1e-7) && close(f.clock[0][1], 1.0, 1e-7));
}

#[test]
fn calendar_features_empty_input() {
    let f = calendar_features(&[], Market::Ashare, Timescale::Min1);
    assert!(f.is_empty());
}

fn event_stamps() -> Vec<Timestamp> {
    vec![
        ts(2026, 1, 5, 10, 0),
        ts(2026, 1, 5, 10, 30),
        ts(2026, 1, 5, 11, 30),
        ts(2026, 1, 5, 13, 30),
        ts(2026, 1, 5, 14, 0),
        ts(2026, 1, 5, 15, 0),
        ts(2026, 1, 6, 10, 0),
        ts(2026, 1, 9, 15, 0),
        ts(2026, 1, 12, 10, 0),
        ts(2026, 1, 12, 12, 0),
        ts(2026, 1, 12, 13, 30),
        ts(2026, 1, 13, 10, 0),
    ]
}

fn events(
    stamps: &[Timestamp],
    m: Market,
    s: Timescale,
    prev: Option<Timestamp>,
    cal: &TradingCalendar,
) -> Vec<i64> {
    detect_events(stamps, m, s, prev, DEFAULT_GAP_MULTIPLIER, cal)
}

#[test]
fn detect_events_intraday_matches_upstream() {
    let stamps = event_stamps();
    let wk = TradingCalendar::weekends_only();
    let want = vec![2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1];
    assert_eq!(
        events(&stamps, Market::Ashare, Timescale::Min30, None, &wk),
        want
    );
    assert_eq!(
        events(&stamps, Market::Ashare, Timescale::Min30, None, &table()),
        want
    );
    let with_prev = events(
        &stamps,
        Market::Ashare,
        Timescale::Min30,
        Some(ts(2026, 1, 5, 9, 30)),
        &wk,
    );
    assert_eq!(with_prev, vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1]);
}

#[test]
fn detect_events_holidays_only_count_with_table() {
    let stamps = [
        ts(2025, 9, 30, 15, 0),
        ts(2025, 10, 8, 15, 0),
        ts(2025, 10, 9, 15, 0),
        ts(2025, 10, 10, 15, 0),
        ts(2025, 10, 14, 15, 0),
        ts(2025, 10, 20, 15, 0),
    ];
    assert_eq!(
        events(&stamps, Market::Ashare, Timescale::Day1, None, &table()),
        vec![2, 0, 0, 0, 0, 1]
    );
    assert_eq!(
        events(
            &stamps,
            Market::Ashare,
            Timescale::Day1,
            None,
            &TradingCalendar::weekends_only()
        ),
        vec![2, 1, 0, 0, 0, 1]
    );
}

#[test]
fn detect_events_beyond_table_end_uses_weekday_rule() {
    let stamps = [
        ts(2026, 4, 9, 15, 0),
        ts(2026, 4, 10, 15, 0),
        ts(2026, 4, 13, 15, 0),
        ts(2026, 4, 14, 15, 0),
        ts(2026, 4, 16, 15, 0),
        ts(2026, 4, 20, 15, 0),
    ];
    let want = vec![2, 0, 0, 0, 0, 1];
    assert_eq!(
        events(&stamps, Market::Ashare, Timescale::Day1, None, &table()),
        want
    );
    assert_eq!(
        events(
            &stamps,
            Market::Ashare,
            Timescale::Day1,
            None,
            &TradingCalendar::weekends_only()
        ),
        want
    );
}

#[test]
fn detect_events_crypto_weekend_is_never_a_gap() {
    let wk = TradingCalendar::weekends_only();
    let stamps = [
        ts(2026, 1, 2, 22, 0),
        ts(2026, 1, 2, 23, 0),
        ts(2026, 1, 3, 0, 0),
        ts(2026, 1, 3, 1, 0),
        ts(2026, 1, 5, 0, 0),
        ts(2026, 1, 5, 3, 0),
    ];
    assert_eq!(
        events(&stamps, Market::Crypto, Timescale::Hour1, None, &wk),
        vec![2, 0, 0, 0, 0, 0]
    );
    let wkend = [
        ts(2026, 1, 2, 15, 0),
        ts(2026, 1, 3, 10, 0),
        ts(2026, 1, 4, 10, 0),
        ts(2026, 1, 5, 10, 0),
    ];
    assert_eq!(
        events(&wkend, Market::Ashare, Timescale::Min30, None, &wk),
        vec![2, 0, 0, 0]
    );
}

#[test]
fn detect_events_empty_input() {
    let wk = TradingCalendar::weekends_only();
    assert!(events(&[], Market::Ashare, Timescale::Min30, None, &wk).is_empty());
}

#[test]
fn attach_window_calendar_zeroes_target_events() {
    let stamps = event_stamps();
    let wk = TradingCalendar::weekends_only();
    let w = attach_window_calendar(
        &stamps,
        Market::Ashare,
        Timescale::Min30,
        8,
        None,
        None,
        &wk,
    )
    .unwrap();
    assert_eq!(w.event, vec![2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
    assert_eq!(w.features.len(), stamps.len());

    let given = [2, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1];
    let w = attach_window_calendar(
        &stamps,
        Market::Ashare,
        Timescale::Min30,
        8,
        Some(&given),
        None,
        &wk,
    )
    .unwrap();
    assert_eq!(w.event, vec![2, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]);

    let err = attach_window_calendar(
        &stamps,
        Market::Ashare,
        Timescale::Min30,
        8,
        Some(&given[..3]),
        None,
        &wk,
    );
    assert!(matches!(err, Err(crate::Error::InvalidData(_))));
}
