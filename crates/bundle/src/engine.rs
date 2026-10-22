use crate::{Error, Store};

/// A model's session, as its hosts drive it: the browser runtime and the native CLI build it
/// from a store once every part is in, and drive its own `start` and `step`, whose inputs differ
/// from model to model.
pub trait Engine: Sized {
    /// Builds the session from a store holding every part (`Store::complete`).
    fn new(store: &mut Store) -> Result<Self, Error>;

    /// The threads of the team it computes on.
    fn threads(&self) -> usize;

    /// Abandons the current job, if any.
    fn cancel(&mut self);
}
