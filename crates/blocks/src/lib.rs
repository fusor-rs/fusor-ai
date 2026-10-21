//! Architecture blocks: shape-parameterized pieces of model architectures, written once on the
//! fusor-ai kernels and thread team. A block never names a checkpoint, a tensor of a particular
//! model, a sample rate or a language; the models in `crates/models` configure and wire them.
//!
//! Every block loads itself from a bundle with `bundle::Load`, under the tensor names relative to
//! a prefix that its documentation lists.

pub mod adaln_mlp;
mod attention;
pub mod conformer;
pub mod dit;
pub mod dpm;
pub mod flow;
pub mod istft;
mod layer;
pub mod norm;
mod resnet;
pub mod rotary;
pub mod seanet;
pub mod transformer;
pub mod unet;
pub mod vae;
pub mod vit;

/// The base of the sinusoids that encode a position or a time (`10000^(−2i/d)`).
const SINUSOID_BASE: f32 = 10_000.0;
