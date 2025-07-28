use crate::{action::Action, state::State, value::QValue};

#[derive(Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct QTable {
    /// 2D vector for Q-values, indexed by state and action
    qvalues: Vec<Vec<QValue>>,
    config: QConfig,
}

#[derive(Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct QConfig {
    /// size of the state space
    pub state_size: usize,
    /// size of the action space
    pub action_size: usize,

    /// discount factor
    pub gamma: f64,
    /// learning rate
    pub alpha: f64,
    /// exploration rate
    pub epsilon: f64,
}

impl Default for QConfig {
    fn default() -> Self {
        Self {
            state_size: 14,
            action_size: 14,
            gamma: 0.99,
            alpha: 0.5,
            epsilon: 0.5,
        }
    }
}

impl QTable {
    pub fn new() -> Self {
        Self::new_with(Default::default())
    }

    pub fn new_with(config: QConfig) -> Self {
        let qvalues = (0..config.state_size)
            .map(|_| QValue::random_collect(config.action_size))
            .collect();
        
        Self { qvalues, config }
    }

    pub fn load(file_path: impl AsRef<std::path::Path>) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(file_path)?;
        let reader = std::io::BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }

    pub fn save(&self, file_path: impl AsRef<std::path::Path>) -> Result<(), std::io::Error> {
        let file = std::fs::File::create(file_path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, &self)?;
        Ok(())
    }
}

impl QTable {
    pub fn state_size(&self) -> usize {
        self.config.state_size
    }
    pub fn action_size(&self) -> usize {
        self.config.action_size
    }
    
    pub fn pick_state(&self, index: usize) -> Option<State> {
        State::new_on(self, index).ok()
    }
    pub fn pick_action(&self, index: usize) -> Option<Action> {
        Action::new_on(self, index).ok()
    }

    pub fn gamma(&self) -> f64 {
        self.config.gamma
    }
    pub fn alpha(&self) -> f64 {
        self.config.alpha
    }
    pub fn epsilon(&self) -> f64 {
        self.config.epsilon
    }

    /// Sets alpha to `alpha * rate`.
    /// This is useful for decay strategies where you want to reduce the learning rate over time.
    pub fn decay_alpha_with_rate(&mut self, rate: f64) {
        self.config.alpha = self.config.alpha * rate;
    }
    /// Sets epsilon to `epsilon * rate`.
    /// This is useful for decay strategies where you want to reduce the exploration rate over time.
    pub fn decay_epsilon_with_rate(&mut self, rate: f64) {
        self.config.epsilon = self.config.epsilon * rate;
    }
}

impl QTable {
    pub fn states(&self) -> impl Iterator<Item = State> {
        (0..self.state_size()).map(|i| State::new_on(self, i).unwrap())
    }

    pub fn actions(&self) -> impl Iterator<Item = Action> {
        (0..self.action_size()).map(|i| Action::new_on(self, i).unwrap())
    }
}

pub struct ActionQValuesForAState([QValue]);

impl std::ops::Deref for ActionQValuesForAState {
    type Target = [QValue];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::Index<Action> for ActionQValuesForAState {
    type Output = QValue;

    fn index(&self, action: Action) -> &Self::Output {
        &self.0[*action]
    }
}

impl std::ops::Index<State> for QTable {
    type Output = ActionQValuesForAState;

    fn index(&self, state: State) -> &Self::Output {
        // SAFETY: `ActionQValuesForAState` is just a newtype wrapper around a slice of `QValue`.
        unsafe { std::mem::transmute(&self.qvalues[*state][..]) }
    }
}

// Doesn't implement `IndexMut` to prevent direct modification of Q-values.
// This is to ensure that Q-values are **ONLY** updated through the `update` method,
// which applies the learning algorithm correctly.
impl std::ops::Index<(State, Action)> for QTable {
    type Output = QValue;

    fn index(&self, (state, action): (State, Action)) -> &Self::Output {
        &self.qvalues[*state][*action]
    }
}

pub trait Strategy {
    fn determine(qtable: &QTable, state: State) -> Action;
}

pub struct QUpdate {
    pub state: State,
    pub action: Action,
    pub reward: f64,
    pub next_state: State,
}

impl QTable {
    pub fn next_action<S: Strategy>(&self, state: State) -> Action {
        S::determine(self, state)
    }

    pub fn update(&mut self, QUpdate {
        state,
        action,
        reward,
        next_state,
    }: QUpdate) {
        let current_qvalue = self.qvalues[*state][*action];
        let next_max_qvalue = self.qvalues[*next_state].iter().max().unwrap().clone();

        self.qvalues[*state][*action] = QValue::new(
            (1. - self.alpha()) * (*current_qvalue) + self.alpha() * (reward + self.gamma() * (*next_max_qvalue)),
        );
    }
}

pub mod strategy {
    use super::{Strategy, Action, QTable, State};
    use rand::{Rng, distr::{weighted::WeightedIndex}};

    pub struct MostQValue;
    impl Strategy for MostQValue {
        fn determine(qtable: &QTable, state: State) -> Action {
            let max_action_qvalue = qtable[state].iter().max().unwrap();
            let candidates = qtable[state]
                .iter()
                .enumerate()
                .filter(|(_, qvalue)| *qvalue == max_action_qvalue)
                .map(|(index, _)| Action::new_on(qtable, index).unwrap())
                .collect::<Vec<_>>();
            candidates[rand::rng().random_range(0..candidates.len())]
        }
    }

    pub struct SoftMax;
    impl Strategy for SoftMax {
        fn determine(qtable: &QTable, state: State) -> Action {
            let qvalues = &qtable[state];
            let max_action_qvalue = qvalues.iter().max().unwrap();
            let softmax_probabilities = {
                let exp = qvalues.iter().map(|q| (**q - **max_action_qvalue).exp()).collect::<Vec<_>>();
                let exp_sum = exp.iter().sum::<f64>();
                exp.iter().map(|&e| e / exp_sum).collect::<Vec<_>>()
            };
            let selected_index = rand::rng().sample(WeightedIndex::new(&softmax_probabilities).unwrap());
            Action::new_on(qtable, selected_index).unwrap()
        }
    }

    pub struct EpsilonGreedy;
    impl Strategy for EpsilonGreedy {
        fn determine(qtable: &QTable, state: State) -> Action {
            if rand::rng().random_range(0.0..1.0) < qtable.epsilon() {
                Random::determine(qtable, state)
            } else {
                MostQValue::determine(qtable, state)
            }
        }
    }

    pub struct Random;
    impl Strategy for Random {
        fn determine(qtable: &QTable, _state: State) -> Action {
            let selected_index = rand::rng().random_range(0..qtable.action_size());
            Action::new_on(qtable, selected_index).unwrap()
        }
    }
}
