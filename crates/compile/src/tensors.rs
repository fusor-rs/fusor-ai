use kernels::Tensor;

use crate::{BundleWriter, Checkpoint, Error, Profile};

/// A model's tensors on their way into its bundle, each encoded in the format its profile picks.
pub struct Tensors<'bundle, P> {
    writer: &'bundle mut BundleWriter,
    profile: P,
}

impl<'bundle, P: Profile> Tensors<'bundle, P> {
    pub fn new(writer: &'bundle mut BundleWriter, profile: P) -> Self {
        Self { writer, profile }
    }

    /// Row-major `values` of `shape` as tensor `name`.
    pub fn add(&mut self, name: &str, shape: Vec<usize>, values: &[f32]) -> Result<(), Error> {
        let dtype = self.profile.dtype(name, &shape);
        let tensor = Tensor::encode(dtype, shape, values)
            .map_err(|error| Error::Format(format!("{name}: {error}")))?;
        self.writer.add_tensor(name, &tensor)
    }

    pub fn vector(&mut self, name: &str, values: &[f32]) -> Result<(), Error> {
        self.add(name, vec![values.len()], values)
    }

    /// A linear layer of `rows` outputs: `name.weight` and, when given, `name.bias`.
    pub fn linear(
        &mut self,
        name: &str,
        rows: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> Result<(), Error> {
        let shape = vec![rows, weight.len() / rows];
        self.add(&format!("{name}.weight"), shape, weight)?;
        match bias {
            Some(bias) => self.vector(&format!("{name}.bias"), bias),
            None => Ok(()),
        }
    }

    /// The checkpoint's linear layer (or 1-wide convolution) `source` as `name`: its weight and,
    /// when it has one, its bias.
    pub fn copy_linear(
        &mut self,
        checkpoint: &Checkpoint,
        name: &str,
        source: &str,
    ) -> Result<(), Error> {
        let weight_name = format!("{source}.weight");
        let rows = checkpoint.shape(&weight_name)?[0];
        let bias_name = format!("{source}.bias");
        let bias = if checkpoint.contains(&bias_name) {
            Some(checkpoint.f32(&bias_name)?)
        } else {
            None
        };
        let weight = checkpoint.f32(&weight_name)?;
        self.linear(name, rows, &weight, bias.as_deref())
    }

    /// The checkpoint's LayerNorm `source`, its `weight` and `bias`, as `name`.
    pub fn copy_norm(
        &mut self,
        checkpoint: &Checkpoint,
        name: &str,
        source: &str,
    ) -> Result<(), Error> {
        for part in ["weight", "bias"] {
            let values = checkpoint.f32(&format!("{source}.{part}"))?;
            self.vector(&format!("{name}.{part}"), &values)?;
        }
        Ok(())
    }

    /// A zero-centred RMSNorm's multipliers `1 + w`, from `source.weight`, as the vector `name`.
    pub fn shifted_norm(
        &mut self,
        checkpoint: &Checkpoint,
        name: &str,
        source: &str,
    ) -> Result<(), Error> {
        let weight = checkpoint.f32(&format!("{source}.weight"))?;
        let shifted: Vec<f32> = weight.iter().map(|value| 1.0 + value).collect();
        self.vector(name, &shifted)
    }

    /// The linear `name` reading `scale ⊙ x + shift` instead of `x`: `weight`'s columns times
    /// `scale`, the bias plus `weight · shift` (summed in f64).
    pub fn linear_of_affine(
        &mut self,
        name: &str,
        (weight, mut bias): (&[f32], Vec<f32>),
        (scale, shift): (&[f32], &[f32]),
    ) -> Result<(), Error> {
        let mut folded = Vec::with_capacity(weight.len());
        for (row, bias) in weight.chunks_exact(scale.len()).zip(&mut bias) {
            let offset: f64 = row
                .iter()
                .zip(shift)
                .map(|(&weight, &shift)| f64::from(weight) * f64::from(shift))
                .sum();
            *bias += offset as f32;
            folded.extend(row.iter().zip(scale).map(|(weight, scale)| weight * scale));
        }
        self.linear(name, bias.len(), &folded, Some(&bias))
    }
}
