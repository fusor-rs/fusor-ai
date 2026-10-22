//! Dumps golden intermediate tensors from the reference `ptts` runtime (xn-ptts) for one
//! sentence, so a from-scratch port can be checked stage by stage (teacher-forced).
//!
//! Usage: phonon-golden-harness <model_dir> <out.safetensors> <out.wav> [f32|q8] [text]
//!
//! Recorded tensors (all f32 unless noted):
//!   tokens        [T]        i64  token ids of the prepared chunk (single chunk expected)
//!   text_emb      [1,T,512]       text-conditioner rows for those tokens
//!   voice_emb     [1,125,512]     default voice: speaker_wavs latents projected by speaker_proj
//!   conditions    [1,1,512]       summed conditioner vector (num_speakers=1, duration_delta=0)
//!   noise         [F,32]          the Gaussian noise fed to the flow net at each frame
//!   latent        [F,32]          the sampled (normalized) latent per frame
//!   eos_logit     [F]             raw EOS logit per frame
//!   pcm           [F*1920]        Mimi decoder output per frame, streaming, concatenated
//!   meta: frames_after_eos, frame_budget, seq_budget, temperature, seed in the JSON header.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use ptts::flow_lm::{NormalRng, Rng, StepInput};
use ptts::plan::EosPolicy;
use ptts::preprocess::{Lang, Normalize};
use ptts::tts_model::{MAX_TOKENS_PER_CHUNK, TTSModel};
use xn::{BackendQ, CpuDevice, Tensor, TypedTensor};

const TEMPERATURE: f32 = 0.3;
const SEED: u64 = 4242424242424242;
const DEFAULT_TEXT: &str =
    "The evening wind drifted down from the mountain pass, and the distant clouds slowly turned gold.";

/// Records every value the sampler draws so the port can replay them.
struct RecordingRng {
    inner: NormalRng,
    drawn: Vec<f32>,
}

impl Rng for RecordingRng {
    fn sample(&mut self) -> f32 {
        let v = self.inner.sample();
        self.drawn.push(v);
        v
    }
}

fn f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Result<TypedTensor<CpuDevice>> {
    Ok(TypedTensor::F32(Tensor::from_vec(data, shape, &xn::CPU)?))
}

fn run<Q: BackendQ<T = f32, B = CpuDevice>>(
    model_dir: &std::path::Path,
    weights: &str,
    text: &str,
) -> Result<(HashMap<String, TypedTensor<CpuDevice>>, HashMap<String, String>, Vec<f32>, usize)> {
    let cfg = ptts::checkpoint::read_config(model_dir.join("config.json"))?;
    let vb = ptts::loader::load_weights::<Q>(&model_dir.join(weights), &xn::CPU)?;
    let tokenizer = ptts::tok::Tok::open(&model_dir.join("tokenizer.json"))?;
    let conditions_map: HashMap<String, String> = HashMap::new();
    let model: TTSModel<Q> = TTSModel::load(&vb, Box::new(tokenizer), &cfg, &conditions_map)?;

    let voice_emb = ptts::loader::load_voice_emb(
        &model_dir.join("default-voice.safetensors"),
        cfg.embeddings_ext.as_deref(),
        model.speaker_proj(),
        &xn::CPU,
    )?;
    let conditions = ptts::loader::load_conditions::<f32, CpuDevice>(&vb, &cfg, &conditions_map)?
        .ok_or_else(|| anyhow!("checkpoint has no conditioners"))?;

    let tokenizer = model
        .flow_lm
        .conditioner
        .tokenizer
        .as_deref()
        .ok_or_else(|| anyhow!("no tokenizer"))?;
    let chunks = ptts::plan::chunks(
        tokenizer,
        text,
        Normalize::for_lang(Lang::En),
        MAX_TOKENS_PER_CHUNK,
        cfg.mimi.frame_rate,
    )?;
    if chunks.len() != 1 {
        return Err(anyhow!("expected one chunk, got {}", chunks.len()));
    }
    let chunk = &chunks[0];
    eprintln!("prepared chunk text: {:?}", chunk.text);
    eprintln!("tokens ({}): {:?}", chunk.tokens.len(), chunk.tokens);
    eprintln!("frames_after_eos={} frame_budget={}", chunk.frames_after_eos, chunk.frame_budget);

    let voice_len = model.voice_prompt_len(voice_emb.dim(1usize)?);
    let seq_budget = voice_len + chunk.tokens.len() + chunk.frame_budget;
    let mut state = model.init_flow_lm_state(1, seq_budget)?;
    model.prompt_audio(&mut state, &voice_emb)?;
    model.prompt_text(&mut state, &chunk.tokens)?;
    let text_emb = model.flow_lm.conditioner.embed_tokens(&chunk.tokens)?;

    let mut mimi_state = model.init_mimi_state(1)?;
    let mut rng = RecordingRng { inner: NormalRng::new(TEMPERATURE, SEED)?, drawn: vec![] };
    let mut eos = EosPolicy::new(chunk.frames_after_eos);
    let mut prev: Option<Tensor<f32, CpuDevice>> = None;
    let mut latents = Vec::new();
    let mut eos_logits = Vec::new();
    let mut pcm = Vec::new();
    let mut frames = 0usize;
    for _ in 0..chunk.frame_budget {
        let input = match &prev {
            None => StepInput::Bos { batch: 1 },
            Some(t) => StepInput::Latent(t),
        };
        let (latent, eos_logit) = model.generate_step_parts(&mut state, input, &mut rng)?;
        let eos_val = eos_logit.to_vec()?;
        let is_eos = model.eos_from_logit(&eos_val);
        let audio = model.decode_latent(&latent, &mut mimi_state)?;
        pcm.extend(audio.narrow(0, ..1)?.contiguous()?.to_vec()?);
        latents.extend(latent.to_vec()?);
        eos_logits.push(eos_val[0]);
        frames += 1;
        if eos.should_stop(is_eos) {
            break;
        }
        prev = Some(latent);
    }
    let ldim = model.flow_lm.ldim;
    eprintln!("frames={frames} pcm_len={} ({:.2}s)", pcm.len(), pcm.len() as f32 / 24000.0);

    let mut out = HashMap::new();
    out.insert(
        "tokens".to_string(),
        TypedTensor::I64(Tensor::from_vec(
            chunk.tokens.iter().map(|&t| t as i64).collect(),
            chunk.tokens.len(),
            &xn::CPU,
        )?),
    );
    out.insert("text_emb".to_string(), TypedTensor::F32(text_emb));
    out.insert("voice_emb".to_string(), TypedTensor::F32(voice_emb));
    out.insert("conditions".to_string(), TypedTensor::F32(conditions));
    out.insert("noise".to_string(), f32_tensor(rng.drawn.clone(), vec![frames, ldim])?);
    out.insert("latent".to_string(), f32_tensor(latents, vec![frames, ldim])?);
    out.insert("eos_logit".to_string(), f32_tensor(eos_logits, vec![frames])?);
    out.insert("pcm".to_string(), f32_tensor(pcm.clone(), vec![pcm.len()])?);

    let meta = HashMap::from([
        ("text".to_string(), text.to_string()),
        ("prepared_text".to_string(), chunk.text.clone()),
        ("frames_after_eos".to_string(), chunk.frames_after_eos.to_string()),
        ("frame_budget".to_string(), chunk.frame_budget.to_string()),
        ("seq_budget".to_string(), seq_budget.to_string()),
        ("voice_len".to_string(), voice_len.to_string()),
        ("temperature".to_string(), TEMPERATURE.to_string()),
        ("seed".to_string(), SEED.to_string()),
        ("weights".to_string(), weights.to_string()),
        ("frames".to_string(), frames.to_string()),
    ]);
    Ok((out, meta, pcm, model.sample_rate()))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = std::path::Path::new(args.get(1).context("model dir")?);
    let out_path = args.get(2).context("output safetensors")?;
    let wav_path = args.get(3).context("output wav")?;
    let quant = args.get(4).map(String::as_str).unwrap_or("f32");
    let text = args.get(5).map(String::as_str).unwrap_or(DEFAULT_TEXT);
    let (tensors, meta, pcm, sample_rate) = match quant {
        "f32" => run::<xn::Unquantized<f32, CpuDevice>>(model_dir, "model.safetensors", text)?,
        "q8" => run::<xn::quantized::Q80F32>(model_dir, "model.q8.gguf", text)?,
        other => return Err(anyhow!("quant must be f32 or q8, got {other}")),
    };
    xn::safetensors::save_with_data_info(&tensors, Some(meta), out_path)?;
    ptts::wav::write_wav_file(wav_path, &pcm, sample_rate as u32)?;
    eprintln!("wrote {out_path} and {wav_path}");
    Ok(())
}
