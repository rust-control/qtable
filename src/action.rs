#[derive(Clone, Copy)]
pub struct Action(usize);

impl std::ops::Deref for Action {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub enum ActionError {
    InvalidActionIndex { given: usize, max: usize },
}
impl std::error::Error for ActionError {}
impl std::fmt::Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionError::InvalidActionIndex { given, max } => {
                write!(f, "Invalid action index: {} (max: {})", given, max)
            }
        }
    }
}

impl Action {
    pub fn new_on(qtable: &crate::QTable, index: usize) -> Result<Self, ActionError> {
        (index < qtable.action_size())
            .then_some(Self(index))
            .ok_or(ActionError::InvalidActionIndex {
                given: index,
                max: qtable.action_size(),
            })
    }

    pub fn index(&self) -> usize {
        self.0
    }
}
