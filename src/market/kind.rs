use std::fmt;
use std::str::FromStr;

/// Naive (timezone-less) exchange-local timestamp. A bar's timestamp is its close time.
pub type Timestamp = chrono::NaiveDateTime;

const ASHARE_SESSIONS: &[(i64, i64)] = &[(570, 690), (780, 900)];
const HK_SESSIONS: &[(i64, i64)] = &[(570, 720), (780, 960)];
const US_SESSIONS: &[(i64, i64)] = &[(570, 960)];
const FUTURES_SESSIONS: &[(i64, i64)] = &[(540, 900)];
const CRYPTO_SESSIONS: &[(i64, i64)] = &[(0, 1440)];

/// Trading venue. The numeric ids match the model's market embedding table (slot 0 is reserved).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Market {
    /// Mainland China A-shares.
    Ashare,
    /// Hong Kong equities.
    Hk,
    /// US equities.
    Us,
    /// Futures.
    Futures,
    /// 24/7 crypto markets.
    Crypto,
}

impl Market {
    /// Every market, in id order.
    pub const ALL: [Market; 5] = [
        Market::Ashare,
        Market::Hk,
        Market::Us,
        Market::Futures,
        Market::Crypto,
    ];

    /// Embedding id (`ashare=1, hk=2, us=3, futures=4, crypto=5`).
    pub fn id(self) -> i64 {
        match self {
            Market::Ashare => 1,
            Market::Hk => 2,
            Market::Us => 3,
            Market::Futures => 4,
            Market::Crypto => 5,
        }
    }

    /// Lowercase name used in stats keys and data directories.
    pub fn as_str(self) -> &'static str {
        match self {
            Market::Ashare => "ashare",
            Market::Hk => "hk",
            Market::Us => "us",
            Market::Futures => "futures",
            Market::Crypto => "crypto",
        }
    }

    /// Trading sessions as `(open_minute, close_minute)` of the local day.
    pub fn session_templates(self) -> &'static [(i64, i64)] {
        match self {
            Market::Ashare => ASHARE_SESSIONS,
            Market::Hk => HK_SESSIONS,
            Market::Us => US_SESSIONS,
            Market::Futures => FUTURES_SESSIONS,
            Market::Crypto => CRYPTO_SESSIONS,
        }
    }

    /// Market close minute of the day, with 1440 mapped to 0.
    pub fn close_minute(self) -> i64 {
        self.last_session_end() % 1440
    }

    /// Whether the final session ends at midnight, so 00:00 closes belong to the previous day.
    pub fn is_24_7(self) -> bool {
        self.last_session_end() == 1440
    }

    fn last_session_end(self) -> i64 {
        self.session_templates()
            .last()
            .map(|&(_, end)| end)
            .unwrap_or(0)
    }
}

impl fmt::Display for Market {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Market {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Market::ALL
            .into_iter()
            .find(|m| m.as_str() == s)
            .ok_or_else(|| crate::Error::InvalidConfig(format!("unknown market {s:?}")))
    }
}

/// Bar duration. The numeric ids match the model's timescale embedding table (slot 0 is reserved).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Timescale {
    /// 1 minute.
    Min1,
    /// 5 minutes.
    Min5,
    /// 15 minutes.
    Min15,
    /// 30 minutes.
    Min30,
    /// 1 hour.
    Hour1,
    /// 2 hours.
    Hour2,
    /// One bar per trading day, stamped at the market close.
    Day1,
}

impl Timescale {
    /// Every timescale, in id order.
    pub const ALL: [Timescale; 7] = [
        Timescale::Min1,
        Timescale::Min5,
        Timescale::Min15,
        Timescale::Min30,
        Timescale::Hour1,
        Timescale::Hour2,
        Timescale::Day1,
    ];

    /// Embedding id (`1m=1 .. 1d=7`).
    pub fn id(self) -> i64 {
        match self {
            Timescale::Min1 => 1,
            Timescale::Min5 => 2,
            Timescale::Min15 => 3,
            Timescale::Min30 => 4,
            Timescale::Hour1 => 5,
            Timescale::Hour2 => 6,
            Timescale::Day1 => 7,
        }
    }

    /// Nominal bar length in minutes (`1d` is 1440).
    pub fn minutes(self) -> i64 {
        match self {
            Timescale::Min1 => 1,
            Timescale::Min5 => 5,
            Timescale::Min15 => 15,
            Timescale::Min30 => 30,
            Timescale::Hour1 => 60,
            Timescale::Hour2 => 120,
            Timescale::Day1 => 1440,
        }
    }

    /// Short name used in stats keys and data directories (`"1m"`, ..., `"1d"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Timescale::Min1 => "1m",
            Timescale::Min5 => "5m",
            Timescale::Min15 => "15m",
            Timescale::Min30 => "30m",
            Timescale::Hour1 => "1h",
            Timescale::Hour2 => "2h",
            Timescale::Day1 => "1d",
        }
    }

    /// Whether the scale has exactly one bar per trading day.
    pub fn is_daily_single_bar(self) -> bool {
        matches!(self, Timescale::Day1)
    }
}

impl fmt::Display for Timescale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Timescale {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Timescale::ALL
            .into_iter()
            .find(|t| t.as_str() == s)
            .ok_or_else(|| crate::Error::InvalidConfig(format!("unknown timescale {s:?}")))
    }
}

fn session_close_minutes(start: i64, end: i64, step: i64) -> Vec<i64> {
    let mut mins = Vec::new();
    let mut m = start + step;
    while m < end {
        mins.push(m);
        m += step;
    }
    mins.push(end);
    mins
}

/// Close minutes of the bars in each session. The last bar of a session may be a short stub.
/// `1d` yields a single grid holding the market close minute (1440 for 24/7 markets).
pub fn session_grids(market: Market, timescale: Timescale) -> Vec<Vec<i64>> {
    if timescale.is_daily_single_bar() {
        return vec![vec![market.last_session_end()]];
    }
    let step = timescale.minutes();
    market
        .session_templates()
        .iter()
        .map(|&(s, e)| session_close_minutes(s, e, step))
        .collect()
}

/// Close minutes of all bars in a trading day (1440 means the next day's 00:00).
pub fn daily_grid_minutes(market: Market, timescale: Timescale) -> Vec<i64> {
    session_grids(market, timescale)
        .into_iter()
        .flatten()
        .collect()
}

/// Number of bars in each session (US 1h gives `[7]`, US 2h gives `[4]`, any 1d gives `[1]`).
pub fn bars_per_session(market: Market, timescale: Timescale) -> Vec<usize> {
    if timescale.is_daily_single_bar() {
        return vec![1];
    }
    session_grids(market, timescale)
        .iter()
        .map(Vec::len)
        .collect()
}

/// Total bars per trading day (A-share 1h gives 4, US 1h gives 7, any 1d gives 1).
pub fn bars_per_day(market: Market, timescale: Timescale) -> usize {
    bars_per_session(market, timescale).iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_match_upstream_tables() {
        let ids: Vec<i64> = Market::ALL.iter().map(|m| m.id()).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
        let ids: Vec<i64> = Timescale::ALL.iter().map(|t| t.id()).collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5, 6, 7]);
        let mins: Vec<i64> = Timescale::ALL.iter().map(|t| t.minutes()).collect();
        assert_eq!(mins, vec![1, 5, 15, 30, 60, 120, 1440]);
    }

    #[test]
    fn parse_round_trips() {
        for m in Market::ALL {
            assert_eq!(m.as_str().parse::<Market>().unwrap(), m);
        }
        for t in Timescale::ALL {
            assert_eq!(t.to_string().parse::<Timescale>().unwrap(), t);
        }
        assert!(matches!(
            "nyse".parse::<Market>(),
            Err(crate::Error::InvalidConfig(_))
        ));
        assert!(matches!(
            "3m".parse::<Timescale>(),
            Err(crate::Error::InvalidConfig(_))
        ));
    }

    #[test]
    fn bars_per_day_matches_upstream() {
        assert_eq!(bars_per_day(Market::Ashare, Timescale::Hour1), 4);
        assert_eq!(bars_per_day(Market::Us, Timescale::Hour1), 7);
        assert_eq!(bars_per_session(Market::Us, Timescale::Hour2), vec![4]);
        assert_eq!(bars_per_day(Market::Ashare, Timescale::Min1), 240);
        assert_eq!(bars_per_day(Market::Hk, Timescale::Min30), 11);
        assert_eq!(bars_per_day(Market::Crypto, Timescale::Hour1), 24);
        assert_eq!(bars_per_day(Market::Futures, Timescale::Hour2), 3);
        for m in Market::ALL {
            assert_eq!(bars_per_day(m, Timescale::Day1), 1);
        }
    }

    #[test]
    fn grids_end_with_short_stub() {
        let g = session_grids(Market::Ashare, Timescale::Hour2);
        assert_eq!(g, vec![vec![690], vec![900]]);
        let g = session_grids(Market::Us, Timescale::Hour2);
        assert_eq!(g, vec![vec![690, 810, 930, 960]]);
        assert_eq!(
            daily_grid_minutes(Market::Crypto, Timescale::Day1),
            vec![1440]
        );
        assert_eq!(
            daily_grid_minutes(Market::Ashare, Timescale::Day1),
            vec![900]
        );
    }

    #[test]
    fn close_minute_wraps_for_crypto() {
        assert_eq!(Market::Crypto.close_minute(), 0);
        assert!(Market::Crypto.is_24_7());
        assert_eq!(Market::Us.close_minute(), 960);
        assert!(!Market::Us.is_24_7());
    }
}
