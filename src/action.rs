#[derive(Clone, Copy)]
pub struct Action(usize);

impl std::ops::Deref for Action {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Action {
    pub(crate) fn new_on(qtable: &crate::QTable, index: usize) -> Option<Self> {
        (index < qtable.action_size()).then_some(Self(index))
    }
}
