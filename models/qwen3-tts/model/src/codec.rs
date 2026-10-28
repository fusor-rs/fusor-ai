//! The speech tokenizer: its encoder turns a voice clip into frames of 16 codes (`encoder`), its
//! decoder frames into 24 kHz samples (`decoder`).

pub mod decoder;
pub mod encoder;
