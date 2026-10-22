//! The native host every model's CLI shares: the thread pool, the bundle loader, the compile
//! command, the parity comparisons against reference goldens and the stage timings of `bench`.

pub mod bench;
pub mod parity;

use std::path::{Path, PathBuf};
use std::time::Instant;

use bundle::{Engine, Manifest, Store};

/// Bytes pushed into the store per write, as the browser loader streams a part.
const CHUNK: usize = 8 << 20;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Bundle(#[from] bundle::Error),
    #[error("the thread pool: {0}")]
    Threads(#[from] rayon::ThreadPoolBuildError),
}

/// Starts the global pool the team computes on: `requested` threads, else min(8, cores − 2).
pub fn threads(requested: Option<usize>) -> Result<(), Error> {
    let cores = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    let threads = requested.unwrap_or(8.min(cores.saturating_sub(2)).max(1));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    Ok(())
}

/// Loads a bundle directory the way the browser does: every part pushed in order, then hashed.
pub fn load<E: Engine>(dir: &Path) -> Result<E, Error> {
    let start = Instant::now();
    let manifest_path = dir.join("manifest.json");
    let manifest_json =
        std::fs::read_to_string(&manifest_path).map_err(unreadable(manifest_path))?;
    let manifest = Manifest::parse(&manifest_json)?;
    let mut store = Store::new(manifest.clone())?;
    for (index, part) in manifest.parts.iter().enumerate() {
        let path = dir.join(&part.file);
        let bytes = std::fs::read(&path).map_err(unreadable(path))?;
        for chunk in bytes.chunks(CHUNK) {
            store.write(index, chunk)?;
        }
        store.end_part(index)?;
    }
    let engine = E::new(&mut store)?;
    eprintln!(
        "loaded {} ({:.0} MB) in {:.2}s, {} threads",
        dir.display(),
        manifest.tensor_bytes() as f64 / 1e6,
        start.elapsed().as_secs_f64(),
        engine.threads()
    );
    Ok(engine)
}

fn unreadable(path: PathBuf) -> impl FnOnce(std::io::Error) -> Error {
    move |source| Error::Read { path, source }
}

/// Runs a compiler and prints what it wrote: tensors, parts, megabytes and seconds.
pub fn compile(
    run: impl FnOnce() -> Result<Manifest, compile::Error>,
) -> Result<(), compile::Error> {
    let start = Instant::now();
    let manifest = run()?;
    let mb = manifest.parts.iter().map(|part| part.bytes).sum::<u64>() as f64 / 1e6;
    println!(
        "{} tensors, {} parts, {mb:.1} MB in {:.1}s",
        manifest.tensors.len(),
        manifest.parts.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
