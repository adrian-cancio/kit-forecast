//! Conditioning inputs for [`KlineDiT`](super::KlineDiT).

use candle_core::Tensor;

/// Per-forward conditioning tensors (the `cond` dict of the reference model).
///
/// Shapes, with `B` = batch and `L` = context + target length:
///
/// | field      | shape     | dtype                                  |
/// |------------|-----------|----------------------------------------|
/// | `clock`    | `[B,L,16]`| float (`clock_dim`)                    |
/// | `sess`     | `[B,L,2]` | float                                  |
/// | `yday`     | `[B,L,8]` | float (`yday_dim`)                     |
/// | `dow`      | `[B,L]`   | `u32` / `i64` index (0..7)             |
/// | `event`    | `[B,L]`   | `u32` / `i64` index (0..3)             |
/// | `month`    | `[B,L]`   | `u32` / `i64` index (0..12)            |
/// | `market`   | `[B]`     | `u32` / `i64` index (`market_slots`)   |
/// | `sector`   | `[B]`     | `u32` / `i64` index (`sector_slots`)   |
/// | `inst`     | `[B]`     | `u32` / `i64` index (`inst_vocab`)     |
/// | `scale`    | `[B]`     | `u32` / `i64` index (`scale_slots`)    |
/// | `pad_mask` | `[B,L]`   | any integer dtype, non-zero = valid    |
///
/// `pad_mask = None` means every bar is valid. Index tensors must be `u32`, `i64`
/// or `u8` (the dtypes supported by candle's `index_select`).
#[derive(Debug, Clone)]
pub struct Condition {
    pub clock: Tensor,
    pub sess: Tensor,
    pub dow: Tensor,
    pub event: Tensor,
    pub month: Tensor,
    pub yday: Tensor,
    pub market: Tensor,
    pub sector: Tensor,
    pub inst: Tensor,
    pub scale: Tensor,
    pub pad_mask: Option<Tensor>,
}
