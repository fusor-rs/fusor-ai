//! A session: the model built from a complete store, and the team it computes on.

use bundle::{Error, Store};
use team::Team;

pub struct Engine {
    team: Team,
}

impl bundle::Engine for Engine {
    fn new(_store: &mut Store) -> Result<Self, Error> {
        Ok(Self {
            team: Team::global(),
        })
    }

    fn threads(&self) -> usize {
        self.team.threads()
    }

    fn cancel(&mut self) {}
}
