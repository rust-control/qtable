#[derive(Clone, Copy)]
pub struct State(usize);

impl std::ops::Deref for State {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub enum StateError {
    InvalidStateIndex { given: usize, max: usize },
}
impl std::error::Error for StateError {}
impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateError::InvalidStateIndex { given, max } => {
                write!(f, "Invalid state index: {} (max: {})", given, max)
            }
        }
    }
}

impl State {
    pub fn new_on(qtable: &crate::QTable, index: usize) -> Result<Self, StateError> {
        (index < qtable.state_size())
            .then_some(Self(index))
            .ok_or(StateError::InvalidStateIndex {
                given: index,
                max: qtable.state_size(),
            })
    }

    pub fn index(&self) -> usize {
        self.0
    }
}
