//! Market definitions, trading sessions, the trading-day calendar and per-bar calendar features.
//!
//! - [`Market`] / [`Timescale`] / [`Timestamp`]: ids, session templates and bar grids
//! - [`TradingCalendar`]: optional trading-day table loaded from CSV
//! - [`gen_future_timestamps`]: close times of the next bars on the exchange calendar
//! - [`calendar_features`], [`detect_events`], [`attach_window_calendar`]: model conditions

mod calendar;
mod kind;
#[cfg(test)]
mod tests;

pub use calendar::{
    CLOCK_DIM, CalendarFeatures, DEFAULT_GAP_MULTIPLIER, EVENT_FIRST_BAR, EVENT_GAP, SESS_DIM,
    TradingCalendar, WindowCalendar, YDAY_DIM, attach_window_calendar, calendar_features,
    detect_events, gen_future_timestamps,
};
pub use kind::{
    Market, Timescale, Timestamp, bars_per_day, bars_per_session, daily_grid_minutes, session_grids,
};
