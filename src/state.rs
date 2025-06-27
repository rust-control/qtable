#[derive(Clone, Copy)]
pub struct State(usize);

impl std::ops::Deref for State {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl State {
    pub fn new_on(qtable: &crate::QTable, index: usize) -> Option<Self> {
        (index < qtable.state_size()).then_some(Self(index))
    }
}
