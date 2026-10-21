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
}
