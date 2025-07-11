#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct QValue(f64);

impl std::cmp::Eq for QValue {}

impl std::cmp::Ord for QValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // NaN never apears in QValue because of the range constraint
        self.0.partial_cmp(&other.0).expect("Unexpected NaN in QValue comparison")
    }
}

impl std::ops::Deref for QValue {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl QValue {
    pub(super) fn new(value: f64) -> Self {
        Self(value)
    }

    pub(super) fn random_collect(size: usize) -> Vec<Self> {
        (0..size).map(|_| Self(rand::Rng::random(&mut rand::rng()))).collect()
    }
}
