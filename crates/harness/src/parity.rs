//! A stage's output against the reference's goldens.

/// The comparison of two tensors of different lengths.
#[derive(Debug, thiserror::Error)]
#[error("{name}: {ours} values, reference {reference}")]
pub struct LengthMismatch {
    pub name: String,
    pub ours: usize,
    pub reference: usize,
}

/// `‖ours − reference‖ / ‖reference‖`, in f64.
pub fn relative_l2(ours: &[f32], reference: &[f32]) -> f64 {
    let diff: f64 = ours
        .iter()
        .zip(reference)
        .map(|(value, want)| f64::from(value - want).powi(2))
        .sum();
    let norm: f64 = reference.iter().map(|want| f64::from(*want).powi(2)).sum();
    (diff / norm).sqrt()
}

pub fn max_abs(ours: &[f32], reference: &[f32]) -> f32 {
    ours.iter()
        .zip(reference)
        .map(|(value, want)| (value - want).abs())
        .fold(0.0, f32::max)
}

/// `name: … values, rel …, max abs …` after checking that both have as many values.
pub fn compare(name: &str, ours: &[f32], reference: &[f32]) -> Result<String, LengthMismatch> {
    if ours.len() != reference.len() {
        return Err(LengthMismatch {
            name: name.into(),
            ours: ours.len(),
            reference: reference.len(),
        });
    }
    Ok(format!(
        "{name}: {} values, rel {:.2e}, max abs {:.2e}",
        ours.len(),
        relative_l2(ours, reference),
        max_abs(ours, reference)
    ))
}
