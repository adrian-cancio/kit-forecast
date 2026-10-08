use std::f64::consts::PI;
use std::fs;
use std::io;
use std::path::Path;

use chrono::{Datelike, Days, Duration, NaiveDate, NaiveTime, Timelike};

use super::kind::{Market, Timescale, Timestamp, daily_grid_minutes, session_grids};
use crate::{Error, Result};

/// Number of clock Fourier features per bar (`sin`/`cos` for harmonics 1..=8).
pub const CLOCK_DIM: usize = 16;
/// Number of in-session position features per bar.
pub const SESS_DIM: usize = 2;
/// Number of day-of-year Fourier features per bar (`sin`/`cos` for harmonics 1..=4).
pub const YDAY_DIM: usize = 8;
/// Default multiple of the expected bar gap above which a bar is tagged as following a gap.
pub const DEFAULT_GAP_MULTIPLIER: f64 = 3.0;
/// Event code of the first bar of a series.
pub const EVENT_FIRST_BAR: i64 = 2;
/// Event code of a bar that follows an abnormal gap.
pub const EVENT_GAP: i64 = 1;

/// Table of exchange trading days. Without a table only Saturdays and Sundays are closed,
/// and the same rule applies to dates after the end of the table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TradingCalendar {
    days: Option<Vec<NaiveDate>>,
}

impl TradingCalendar {
    /// Calendar that only skips Saturdays and Sundays.
    pub fn weekends_only() -> Self {
        Self { days: None }
    }

    /// Calendar from an explicit list of trading days. The list is sorted; an empty list
    /// yields a weekends-only calendar.
    pub fn from_days<I: IntoIterator<Item = NaiveDate>>(days: I) -> Self {
        let mut days: Vec<NaiveDate> = days.into_iter().collect();
        days.sort();
        if days.is_empty() {
            Self { days: None }
        } else {
            Self { days: Some(days) }
        }
    }

    /// Parse the text of a trading-day CSV.
    ///
    /// The first line is always treated as a header, the first column holds `YYYYMMDD`
    /// dates, and cells that are not valid dates are dropped. A UTF-8 BOM and blank lines
    /// are ignored. No valid dates yields a weekends-only calendar.
    pub fn from_csv_str(text: &str) -> Self {
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let days = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .skip(1)
            .filter_map(|line| parse_yyyymmdd(first_csv_field(line)));
        Self::from_days(days)
    }

    /// Read a trading-day CSV. A path that is not an existing file yields a weekends-only
    /// calendar; any other I/O failure is returned as an error.
    pub fn from_csv_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.is_file() {
            return Ok(Self::weekends_only());
        }
        match fs::read_to_string(path) {
            Ok(text) => Ok(Self::from_csv_str(&text)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self::weekends_only()),
            Err(e) => Err(Error::Io(e)),
        }
    }

    /// Like [`TradingCalendar::from_csv_path`] but falls back to a weekends-only calendar on
    /// every failure.
    pub fn load_csv_or_weekends<P: AsRef<Path>>(path: P) -> Self {
        Self::from_csv_path(path).unwrap_or_default()
    }

    /// Load the trading-day table of a market from a data tree.
    ///
    /// Tries `<data_root>/<market>/trade_dates.csv`, then `<data_root>/trade_dates.csv`,
    /// else returns a weekends-only calendar.
    pub fn load_market<P: AsRef<Path>>(data_root: P, market: Market) -> Self {
        let root = data_root.as_ref();
        let per_market =
            Self::load_csv_or_weekends(root.join(market.as_str()).join("trade_dates.csv"));
        if per_market.days.is_some() {
            return per_market;
        }
        Self::load_csv_or_weekends(root.join("trade_dates.csv"))
    }

    /// Ascending trading days, or `None` for a weekends-only calendar.
    pub fn days(&self) -> Option<&[NaiveDate]> {
        self.days.as_deref()
    }

    /// Whether the calendar has no table and only skips weekends.
    pub fn is_weekends_only(&self) -> bool {
        self.days.is_none()
    }

    /// Whether `day` is a trading day. Dates after the end of the table use the weekday rule.
    pub fn is_trading_day(&self, day: NaiveDate) -> bool {
        let weekday = day.weekday().num_days_from_monday() < 5;
        match &self.days {
            None => weekday,
            Some(days) => match days.last() {
                Some(&last) if day > last => weekday,
                _ => days.binary_search(&day).is_ok(),
            },
        }
    }
}

fn first_csv_field(line: &str) -> &str {
    if let Some(rest) = line.strip_prefix('"') {
        return rest.split('"').next().unwrap_or("");
    }
    line.split(',').next().unwrap_or("")
}

fn parse_yyyymmdd(s: &str) -> Option<NaiveDate> {
    if s.len() != 8 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let year: i32 = s[0..4].parse().ok()?;
    let month: u32 = s[4..6].parse().ok()?;
    let day: u32 = s[6..8].parse().ok()?;
    NaiveDate::from_ymd_opt(year, month, day)
}

/// Close timestamps of the next `n` bars after `last_ts`.
///
/// Days that are not trading days in `calendar` are skipped; with a weekends-only calendar
/// that includes the weekends of 24/7 markets. For 24/7 markets a trading day's final bar
/// closes at 00:00 of the following calendar day, even when that day is a weekend.
pub fn gen_future_timestamps(
    last_ts: Timestamp,
    n: usize,
    market: Market,
    timescale: Timescale,
    calendar: &TradingCalendar,
) -> Vec<Timestamp> {
    let grid = daily_grid_minutes(market, timescale);
    let mut out = Vec::with_capacity(n);
    let mut day = last_ts.date();
    while out.len() < n {
        if calendar.is_trading_day(day) {
            let midnight = day.and_time(NaiveTime::MIN);
            for &m in &grid {
                let Some(cand) = midnight.checked_add_signed(Duration::minutes(m)) else {
                    continue;
                };
                if cand > last_ts {
                    out.push(cand);
                    if out.len() == n {
                        break;
                    }
                }
            }
        }
        match day.succ_opt() {
            Some(next) => day = next,
            None => break,
        }
    }
    out
}

/// Per-bar calendar features, one entry per timestamp.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CalendarFeatures {
    /// `phi = 2*pi*minute_of_day/1440`; `[sin(k*phi), cos(k*phi)]` for `k = 1..=8`.
    pub clock: Vec<[f32; CLOCK_DIM]>,
    /// `[bars_since_open, bars_to_close] / bars_per_session`, in `[0, 1]`.
    pub sess: Vec<[f32; SESS_DIM]>,
    /// Day of week, Monday = 0.
    pub dow: Vec<i64>,
    /// Month, January = 0.
    pub month: Vec<i64>,
    /// `[sin(k*phi_y), cos(k*phi_y)]` for `k = 1..=4`, `phi_y = 2*pi*(day_of_year-1)/year_length`.
    pub yday: Vec<[f32; YDAY_DIM]>,
}

impl CalendarFeatures {
    /// Number of bars.
    pub fn len(&self) -> usize {
        self.dow.len()
    }

    /// Whether there are no bars.
    pub fn is_empty(&self) -> bool {
        self.dow.is_empty()
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn session_fractions(j: usize, b: usize) -> [f32; SESS_DIM] {
    let b = b as f64;
    [(j as f64 / b) as f32, ((b - 1.0 - j as f64) / b) as f32]
}

fn session_position(m: i64, grids: &[Vec<i64>], sessions: &[(i64, i64)]) -> [f32; SESS_DIM] {
    for grid in grids {
        if let Ok(j) = grid.binary_search(&m) {
            return session_fractions(j, grid.len());
        }
    }
    let mut best = 0usize;
    let mut best_dist = i64::MAX;
    for (i, &(s, e)) in sessions.iter().enumerate() {
        let d = (m - s).abs().min((m - e).abs());
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    let (s, e) = sessions[best];
    let b = grids[best].len();
    let j = if (m - s).abs() <= (m - e).abs() {
        0
    } else {
        b - 1
    };
    session_fractions(j, b)
}

/// Calendar features for each timestamp.
///
/// Timestamps are bar close times. On 24/7 markets a bar closing at 00:00 belongs to the
/// previous day, so `dow`, `month` and `yday` use the timestamp minus 1 ns, while `clock`
/// uses the real close time. For `1d` the clock uses the market close minute and `sess` is
/// fixed at `[0.5, 0.5]`. Off-grid intraday timestamps snap to the nearest session edge.
pub fn calendar_features(
    timestamps: &[Timestamp],
    market: Market,
    timescale: Timescale,
) -> CalendarFeatures {
    let n = timestamps.len();
    let mut out = CalendarFeatures {
        clock: Vec::with_capacity(n),
        sess: Vec::with_capacity(n),
        dow: Vec::with_capacity(n),
        month: Vec::with_capacity(n),
        yday: Vec::with_capacity(n),
    };
    let daily = timescale.is_daily_single_bar();
    let shift_midnight = market.is_24_7();
    let grids = session_grids(market, timescale);
    let sessions = market.session_templates();

    for &ts in timestamps {
        let own = if shift_midnight {
            ts.checked_sub_signed(Duration::nanoseconds(1))
                .unwrap_or(ts)
        } else {
            ts
        };
        out.dow
            .push(i64::from(own.weekday().num_days_from_monday()));
        out.month.push(i64::from(own.month0()));

        let year_len = if is_leap_year(own.year()) {
            366.0
        } else {
            365.0
        };
        let phi_y = 2.0 * PI * (f64::from(own.ordinal()) - 1.0) / year_len;
        let mut yday = [0f32; YDAY_DIM];
        for k in 1..=4usize {
            let ang = phi_y * k as f64;
            yday[2 * (k - 1)] = ang.sin() as f32;
            yday[2 * (k - 1) + 1] = ang.cos() as f32;
        }
        out.yday.push(yday);

        let (minutes, sess) = if daily {
            (market.close_minute(), [0.5f32, 0.5f32])
        } else {
            let minutes = i64::from(ts.hour()) * 60 + i64::from(ts.minute());
            let m = if shift_midnight && minutes == 0 {
                1440
            } else {
                minutes
            };
            (minutes, session_position(m, &grids, sessions))
        };
        out.sess.push(sess);

        let phi = 2.0 * PI * ((minutes as f64) % 1440.0) / 1440.0;
        let mut clock = [0f32; CLOCK_DIM];
        for k in 1..=8usize {
            let ang = phi * k as f64;
            clock[2 * (k - 1)] = ang.sin() as f32;
            clock[2 * (k - 1) + 1] = ang.cos() as f32;
        }
        out.clock.push(clock);
    }
    out
}

fn epoch_ns(ts: Timestamp) -> i128 {
    let utc = ts.and_utc();
    i128::from(utc.timestamp()) * 1_000_000_000 + i128::from(utc.timestamp_subsec_nanos())
}

fn expected_next_close(ts: Timestamp, grid_sec: &[i64], calendar: &TradingCalendar) -> Timestamp {
    let day = ts.date();
    let sec = i64::from(ts.time().num_seconds_from_midnight());
    let dow = i64::from(day.weekday().num_days_from_monday());
    let pos = grid_sec.partition_point(|&g| g <= sec);
    let has_more = pos < grid_sec.len();
    let shift = if dow >= 5 {
        7 - dow
    } else if dow == 4 {
        3
    } else {
        1
    };
    let fallback_day = day.checked_add_days(Days::new(shift as u64)).unwrap_or(day);

    let (same_day, exp_day) = match calendar.days() {
        None => {
            let same_day = dow < 5 && has_more;
            (same_day, if same_day { day } else { fallback_day })
        }
        Some(td) => {
            let beyond = td.last().is_some_and(|&last| day > last);
            let is_td = td.binary_search(&day).is_ok();
            let same_day = (if beyond { dow < 5 } else { is_td }) && has_more;
            let ri = td.partition_point(|&d| d <= day);
            let exhausted = ri >= td.len();
            let next_td = td[ri.min(td.len() - 1)];
            let next_day = if beyond || exhausted {
                fallback_day
            } else {
                next_td
            };
            (same_day, if same_day { day } else { next_day })
        }
    };
    let exp_sec = if same_day {
        grid_sec[pos.min(grid_sec.len() - 1)]
    } else {
        grid_sec[0]
    };
    exp_day
        .and_time(NaiveTime::MIN)
        .checked_add_signed(Duration::seconds(exp_sec))
        .unwrap_or(ts)
}

/// Tag each bar with an event code.
///
/// - `2` ([`EVENT_FIRST_BAR`]): first bar of the series (only when `prev_ts` is `None`;
///   otherwise the first bar is compared against `prev_ts` like any other bar).
/// - `1` ([`EVENT_GAP`]): the gap to the previous bar exceeds `gap_multiplier` times the gap
///   expected from the calendar, so overnight, lunch, weekends and holidays are not gaps.
/// - `0`: otherwise.
///
/// The expected next close is the vectorized upstream rule, which differs from
/// [`gen_future_timestamps`] in places: it looks at the weekday of the previous timestamp
/// itself and, without a table, skips weekends with a fixed Friday +3 / Saturday +2 shift.
pub fn detect_events(
    timestamps: &[Timestamp],
    market: Market,
    timescale: Timescale,
    prev_ts: Option<Timestamp>,
    gap_multiplier: f64,
    calendar: &TradingCalendar,
) -> Vec<i64> {
    let n = timestamps.len();
    let mut ev = vec![0i64; n];
    if n == 0 {
        return ev;
    }
    let grid_sec: Vec<i64> = daily_grid_minutes(market, timescale)
        .into_iter()
        .map(|m| m * 60)
        .collect();
    let start = match prev_ts {
        None => {
            ev[0] = EVENT_FIRST_BAR;
            1
        }
        Some(_) => 0,
    };
    for i in start..n {
        let prev = if i == 0 {
            prev_ts.unwrap_or(timestamps[0])
        } else {
            timestamps[i - 1]
        };
        let expected = expected_next_close(prev, &grid_sec, calendar);
        let expected_gap = (epoch_ns(expected) - epoch_ns(prev)) as f64 / 1e9;
        let actual_gap = (epoch_ns(timestamps[i]) - epoch_ns(prev)) as f64 / 1e9;
        if actual_gap > gap_multiplier * expected_gap {
            ev[i] = EVENT_GAP;
        }
    }
    ev
}

/// Calendar features plus event codes of a context + target window.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WindowCalendar {
    /// Per-bar calendar features.
    pub features: CalendarFeatures,
    /// Event code per bar; always 0 from index `n_ctx` onward.
    pub event: Vec<i64>,
}

/// Calendar features and events for a (context + target) window.
///
/// `n_ctx` is the length of the context part; events are forced to 0 after it. `events`
/// overrides event detection; otherwise events are detected with
/// [`DEFAULT_GAP_MULTIPLIER`], `prev_ts` (the bar before the window, if any) and `calendar`.
pub fn attach_window_calendar(
    timestamps: &[Timestamp],
    market: Market,
    timescale: Timescale,
    n_ctx: usize,
    events: Option<&[i64]>,
    prev_ts: Option<Timestamp>,
    calendar: &TradingCalendar,
) -> Result<WindowCalendar> {
    let features = calendar_features(timestamps, market, timescale);
    let mut event = match events {
        None => detect_events(
            timestamps,
            market,
            timescale,
            prev_ts,
            DEFAULT_GAP_MULTIPLIER,
            calendar,
        ),
        Some(ev) => {
            if ev.len() != timestamps.len() {
                return Err(Error::InvalidData(format!(
                    "events length {} must match timestamps length {}",
                    ev.len(),
                    timestamps.len()
                )));
            }
            ev.to_vec()
        }
    };
    for e in event.iter_mut().skip(n_ctx) {
        *e = 0;
    }
    Ok(WindowCalendar { features, event })
}
