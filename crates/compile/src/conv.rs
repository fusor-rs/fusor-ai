//! A checkpoint's convolutions as the matrices `blocks::seanet`, `blocks::code2wav`,
//! `blocks::ecapa`, `blocks::wan_vae`, `blocks::pyramid_jit`'s patch embedding and
//! `kernels::conv::CausalConv3d` read: `[phases · out, taps · in]`, columns oldest input first,
//! then tap by tap.

use crate::{Checkpoint, Error, Profile, Tensors};

struct Conv {
    /// `weight`'s first two dimensions: `[out, in]`, or `[in, out]` for a transposed conv.
    channels: [usize; 2],
    /// The kernel's dimensions after them.
    kernel: Vec<usize>,
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Conv {
    fn read(checkpoint: &Checkpoint, source: &str) -> Result<Self, Error> {
        let weight_name = format!("{source}.weight");
        let (channels, kernel) = match checkpoint.shape(&weight_name)?.split_first_chunk() {
            Some((channels, kernel)) if !kernel.is_empty() => (*channels, kernel.to_vec()),
            _ => return Err(Error::Format(format!("{weight_name} is not a conv weight"))),
        };
        let bias_name = format!("{source}.bias");
        let bias = if checkpoint.contains(&bias_name) {
            Some(checkpoint.f32(&bias_name)?)
        } else {
            None
        };
        Ok(Self {
            channels,
            kernel,
            weight: checkpoint.f32(&weight_name)?,
            bias,
        })
    }

    /// `[out, in, k₁, …]` as `[out, taps · in]`, the taps row-major over the kernel's dimensions.
    fn tap_major(&self) -> Vec<f32> {
        let [outputs, inputs] = self.channels;
        let taps = self.kernel.iter().product::<usize>();
        let mut matrix = Vec::with_capacity(self.weight.len());
        for output in 0..outputs {
            for tap in 0..taps {
                matrix.extend(
                    (0..inputs).map(|input| self.weight[(output * inputs + input) * taps + tap]),
                );
            }
        }
        matrix
    }

    /// A transposed conv `[in, out, w · s]` of stride `s` as a window-`w` conv of `s` phases,
    /// `[s · out, w · in]`: output step `s · p + j` is the sum over blocks `b` (0 the oldest
    /// input) of `W[:, :, (w − 1 − b) · s + j]` times input `p − (w − 1 − b)`.
    fn transposed(&self, stride: usize, source: &str) -> Result<Vec<f32>, Error> {
        let ([inputs, outputs], &[kernel]) = (self.channels, self.kernel.as_slice()) else {
            return Err(Error::Format(format!("{source}: not a 1-D conv weight")));
        };
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
    /// The checkpoint's convolution `source` as `name`: `[out, taps · in]`, the taps row-major over
    /// the kernel's dimensions (the oldest frame or input first), and its bias when it has one.
    pub fn conv(&mut self, checkpoint: &Checkpoint, name: &str, source: &str) -> Result<(), Error> {
        let conv = Conv::read(checkpoint, source)?;
        self.linear(
            name,
            conv.channels[0],
            &conv.tap_major(),
            conv.bias.as_deref(),
        )
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
        let conv = Conv::read(checkpoint, source)?;
        let matrix = conv.transposed(stride, source)?;
        let bias = conv.bias.map(|bias| bias.repeat(stride));
        self.linear(name, stride * conv.channels[1], &matrix, bias.as_deref())
    }
}
