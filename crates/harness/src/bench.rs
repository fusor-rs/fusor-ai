//! Stage timings for a model's `bench`.

use std::time::Instant;

use bundle::Stages;

/// Milliseconds per call of `call`, over `calls` calls after one warm-up call.
pub fn time<T, E>(calls: usize, mut call: impl FnMut() -> Result<T, E>) -> Result<f64, E> {
    call()?;
    let start = Instant::now();
    for _ in 0..calls {
        call()?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / calls as f64)
}

/// Prints a stage's row of the bench table: its name and its milliseconds per `unit`.
pub fn row<S: Stages>(stage: S, milliseconds: f64, unit: &str) {
    println!("  {:12} {milliseconds:9.2} ms{unit}", stage.name());
}
