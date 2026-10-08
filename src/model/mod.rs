//! KlineDiT: single-stream DiT over `[registers | context | target]` tokens.
//!
//! Faithful port of `references/KiT/kit_v1/model`.

mod adaln;
mod attention;
mod block;
mod condition;
mod dit;
mod embeddings;
mod norms;
mod rope;
mod swiglu;

pub use condition::Condition;
pub use dit::{KlineDiT, Modulations};
