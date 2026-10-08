//! High-level forecasting API: load a [`Kit`] model, describe a [`ForecastRequest`], get a
//! [`Forecast`].

mod device;
#[cfg(feature = "hf-hub")]
mod hub;
mod instruments;
mod kit;
mod options;
mod request;
mod result;

pub use device::Device;
pub use instruments::{Instrument, InstrumentTable};
pub use kit::Kit;
pub use options::ForecastOptions;
pub use request::ForecastRequest;
pub use result::Forecast;
