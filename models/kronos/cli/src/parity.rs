//! `kronos parity`: each stage, teacher-forced, against the reference's goldens (`reference/`).
//!
//! Each stage gets the reference's own input (its normalized rows, its tokens and stamps), so
//! errors do not compound from one stage to the next; the forecast whole is compared last.
//! Tensors are compared by relative L2 (`‖ours − ref‖ / ‖ref‖`); the stamps and the context's
//! tokens must be the reference's, one for one.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use bundle::Stages as _;
use compile::Checkpoint;
use harness::parity::compare;
use kronos::model::{FEATURES, MAX_CONTEXT, Stamp};
use kronos::pipeline::Stage;
use kronos::predictor::Window;
use kronos::series::{Normalized, Series};
use kronos::session::{Engine, Request};
use kronos::tokenizer::{self, Token};

/// The reference's tensors for one case (`reference/dump.py` documents them).
struct Goldens {
    normalized: Vec<f32>,
    mean: Vec<f32>,
    deviation: Vec<f32>,
    stamps: Vec<Stamp>,
    codes: Vec<f32>,
    tokens: Vec<Token>,
    s1_logits: Vec<f32>,
    s2_logits: Vec<f32>,
    drawn: Vec<Token>,
    decoded: Vec<f32>,
    forecast: Vec<f32>,
}

impl Goldens {
    fn read(path: &Path) -> Result<Self> {
        let file = Checkpoint::read(path)?;
        let fields = file.i64("stamps")?.into_iter().map(u8::try_from);
        let fields = fields.collect::<Result<Vec<u8>, _>>()?;
        let stamps = fields.as_chunks::<5>().0.to_vec();
        Ok(Self {
            normalized: file.f32("normalized")?,
            mean: file.f32("mean")?,
            deviation: file.f32("deviation")?,
            stamps,
            codes: file.f32("codes")?,
            tokens: tokens(&file, "s1", "s2")?,
            s1_logits: file.f32("s1_logits")?,
            s2_logits: file.f32("s2_logits")?,
            drawn: tokens(&file, "drawn_s1", "drawn_s2")?,
            decoded: file.f32("decoded")?,
            forecast: file.f32("forecast")?,
        })
    }
}

fn tokens(file: &Checkpoint, coarse: &str, fine: &str) -> Result<Vec<Token>> {
    let half = |name| -> Result<Vec<u16>> {
        let values = file.i64(name)?.into_iter().map(u16::try_from);
        Ok(values.collect::<Result<_, _>>()?)
    };
    let (coarse, fine) = (half(coarse)?, half(fine)?);
    Ok(coarse
        .into_iter()
        .zip(fine)
        .map(|(s1, s2)| Token { s1, s2 })
        .collect())
}

pub fn run(engine: &mut Engine, cases: &Path, series_dir: &Path, path: &Path) -> Result<()> {
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str()?.strip_prefix("golden_f32_"))
        .with_context(|| format!("{}: not a golden_f32_<case> file", path.display()))?;
    let request = crate::request(cases, name, series_dir)?;
    let goldens = Goldens::read(path)?;
    println!("# {name}");
    for &stage in Stage::ALL {
        let report = match stage {
            Stage::Normalize => normalize(&request, &goldens)?,
            Stage::Encode => encode(engine, &goldens)?,
            Stage::Read => read(engine, &goldens)?,
            Stage::Sample => {
                let forecast = crate::forecast(engine, &request)?;
                compare(
                    "sample, the forecast",
                    &forecast.values.concat(),
                    &goldens.forecast,
                )?
            }
            Stage::Decode => decode(engine, &goldens)?,
        };
        println!("{report}");
    }
    Ok(())
}

/// The context normalized, and every row's stamp, which must be the reference's.
fn normalize(request: &Request, goldens: &Goldens) -> Result<String> {
    let series = Series::parse(&request.csv)?;
    let context = request.context(&series)?;
    let end = context.end;
    let times = series.times[context.clone()].iter().copied();
    let stamps = times
        .chain(series.times_after(end, request.horizon))
        .map(|time| time.stamp());
    ensure!(
        stamps.eq(goldens.stamps.iter().copied()),
        "normalize: the stamps are not the reference's"
    );
    let normalized = Normalized::new(&series.rows[context]);
    Ok([
        compare("normalize", &normalized.values, &goldens.normalized)?,
        compare("normalize mean", &normalized.mean, &goldens.mean)?,
        compare(
            "normalize deviation",
            &normalized.deviation,
            &goldens.deviation,
        )?,
        format!("normalize: {} stamps identical", goldens.stamps.len()),
    ]
    .join("\n"))
}

fn encode(engine: &mut Engine, goldens: &Goldens) -> Result<String> {
    let codes = engine.codes(&goldens.normalized);
    let tokens = tokenizer::tokens(&codes);
    let same = tokens
        .iter()
        .zip(&goldens.tokens)
        .filter(|(ours, reference)| ours == reference);
    Ok(format!(
        "{}\nencode: {} of {} tokens identical",
        compare("encode", &codes, &goldens.codes)?,
        same.count(),
        goldens.tokens.len()
    ))
}

/// The first step's logits on the reference's context tokens and drawn `s1`.
fn read(engine: &mut Engine, goldens: &Goldens) -> Result<String> {
    let window = Window {
        tokens: &goldens.tokens,
        stamps: &goldens.stamps[..goldens.tokens.len()],
    };
    let [coarse, fine] = engine.logits(window, goldens.drawn[0].s1);
    Ok(format!(
        "{}\n{}",
        compare("read, s1 logits", &coarse, &goldens.s1_logits)?,
        compare("sample, s2 logits", &fine, &goldens.s2_logits)?
    ))
}

/// The reference's last window of tokens decoded.
fn decode(engine: &mut Engine, goldens: &Goldens) -> Result<String> {
    let tokens: Vec<Token> = goldens
        .tokens
        .iter()
        .chain(&goldens.drawn)
        .copied()
        .collect();
    let window = &tokens[tokens.len().saturating_sub(MAX_CONTEXT)..];
    let decoded = engine.decode(window);
    ensure!(
        decoded.len() == goldens.decoded.len(),
        "decode: {} rows, the reference {}",
        decoded.len() / FEATURES,
        goldens.decoded.len() / FEATURES
    );
    Ok(compare("decode", &decoded, &goldens.decoded)?)
}
