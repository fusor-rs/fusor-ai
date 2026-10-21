use kernels::{BLOCK, DType};

use crate::Error;

/// How a model's compiler quantizes: one profile per choice the command line offers.
pub trait Profile: Copy + 'static {
    /// Every profile, in the order the command line lists them.
    const ALL: &'static [Self];
    /// Name prefixes of the tensors kept f32 in every profile: small or precision-critical.
    const ALWAYS_F32: &'static [&'static str];

    /// The name in `manifest.json` and on the command line.
    fn name(self) -> &'static str;

    /// The format of `name`, a matrix of whole blocks that is not kept f32.
    fn matrix_dtype(self, name: &str) -> DType;

    /// The format of tensor `name` of `shape`: f32 unless it is a matrix of whole blocks that is
    /// not kept f32.
    fn dtype(self, name: &str, shape: &[usize]) -> DType {
        let matrix = shape.len() == 2 && shape[1].is_multiple_of(BLOCK);
        let kept = Self::ALWAYS_F32
            .iter()
            .any(|prefix| name.starts_with(prefix));
        if matrix && !kept {
            self.matrix_dtype(name)
        } else {
            DType::F32
        }
    }
}

pub fn profile<P: Profile>(name: &str) -> Result<P, Error> {
    P::ALL
        .iter()
        .copied()
        .find(|profile| profile.name() == name)
        .ok_or_else(|| Error::Format(format!("no profile {name:?} ({})", choices::<P>())))
}

/// The profiles' names as a sentence lists them: "f32, q7 or mix".
fn choices<P: Profile>() -> String {
    let names: Vec<&str> = P::ALL.iter().map(|profile| profile.name()).collect();
    match names.split_last() {
        Some((last, [])) => (*last).to_string(),
        Some((last, rest)) => format!("{} or {last}", rest.join(", ")),
        None => String::new(),
    }
}
