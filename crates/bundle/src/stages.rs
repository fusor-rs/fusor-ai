/// The stages a model's work goes through, in order: what its hosts report progress in and time
/// one by one.
pub trait Stages: Copy + 'static {
    const ALL: &'static [Self];

    /// The stage's name in reports, in lower case.
    fn name(self) -> &'static str;
}
