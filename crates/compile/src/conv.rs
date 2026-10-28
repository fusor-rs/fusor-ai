//! A checkpoint's 1-D convolutions as the causal-conv matrices `blocks::seanet`,
//! `blocks::code2wav` and `blocks::ecapa` read: `[phases · out, window · in]`, columns oldest
//! input first.

use crate::{Checkpoint, Error, Profile, Tensors};

struct Conv1d {
    /// `[out, in, kernel]`, or `[in, out, kernel]` for a transposed conv.
    shape: [usize; 3],
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Conv1d {
    fn read(checkpoint: &Checkpoint, source: &str) -> Result<Self, Error> {
        let weight_name = format!("{source}.weight");
        let [first, second, kernel] = checkpoint.shape(&weight_name)?[..] else {
            return Err(Error::Format(format!("{weight_name} is not a conv weight")));
        };
        let bias_name = format!("{source}.bias");
        let bias = if checkpoint.contains(&bias_name) {
            Some(checkpoint.f32(&bias_name)?)
        } else {
            None
        };
        Ok(Self {
            shape: [first, second, kernel],
            weight: checkpoint.f32(&weight_name)?,
            bias,
        })
    }

    /// `[out, in, k]` as `[out, k · in]`, oldest input first.
    fn tap_major(&self) -> Vec<f32> {
        let [outputs, inputs, kernel] = self.shape;
        let mut matrix = Vec::with_capacity(self.weight.len());
        for output in 0..outputs {
            for tap in 0..kernel {
                matrix.extend(
                    (0..inputs).map(|input| self.weight[(output * inputs + input) * kernel + tap]),
                );
            }
        }
        matrix
    }

    /// A transposed conv `[in, out, w · s]` of stride `s` as a window-`w` conv of `s` phases,
    /// `[s · out, w · in]`: output step `s · p + j` is the sum over blocks `b` (0 the oldest
    /// input) of `W[:, :, (w − 1 − b) · s + j]` times input `p − (w − 1 − b)`.
    fn transposed(&self, stride: usize, source: &str) -> Result<Vec<f32>, Error> {
        let [inputs, outputs, kernel] = self.shape;
        if !kernel.is_multiple_of(stride) {
            return Err(Error::Format(format!(
                "{source}: kernel {kernel} for stride {stride}"
            )));
        }
        let window = kernel / stride;
        let at = |input: usize, output: usize, tap: usize| {
            self.weight[(input * outputs + output) * kernel + tap]
        };
        let mut matrix = Vec::with_capacity(self.weight.len());
        for phase in 0..stride {
            for output in 0..outputs {
                for block in 0..window {
                    let tap = (window - 1 - block) * stride + phase;
                    matrix.extend((0..inputs).map(|input| at(input, output, tap)));
                }
            }
        }
        Ok(matrix)
    }
}

impl<P: Profile> Tensors<'_, P> {
    /// The checkpoint's conv `source` as `name`: `[out, k · in]` and its bias when it has one.
    pub fn conv1d(
        &mut self,
        checkpoint: &Checkpoint,
        name: &str,
        source: &str,
    ) -> Result<(), Error> {
        let conv = Conv1d::read(checkpoint, source)?;
        self.linear(name, conv.shape[0], &conv.tap_major(), conv.bias.as_deref())
    }

    /// The checkpoint's transposed conv `source` of `stride` as `name`: `[stride · out, w · in]`
    /// and its bias, once per phase, when it has one.
    pub fn transposed_conv1d(
        &mut self,
        checkpoint: &Checkpoint,
        name: &str,
        source: &str,
        stride: usize,
    ) -> Result<(), Error> {
        let conv = Conv1d::read(checkpoint, source)?;
        let matrix = conv.transposed(stride, source)?;
        let bias = conv.bias.map(|bias| bias.repeat(stride));
        self.linear(name, stride * conv.shape[1], &matrix, bias.as_deref())
    }
}
