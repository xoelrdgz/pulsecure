use std::sync::Arc;

use crate::ports::{DifferentialPrivacy, PrivateStatistics, Storage};
use crate::PulsecureError;

pub struct AnalyticsService<D, S>
where
    D: DifferentialPrivacy,
    S: Storage,
{
    privacy: Arc<std::sync::Mutex<D>>,
    storage: Arc<S>,
}

impl<D, S> AnalyticsService<D, S>
where
    D: DifferentialPrivacy,
    S: Storage,
    S::Error: Into<crate::adapters::StorageError>,
{
    pub fn new(privacy: D, storage: Arc<S>) -> Self {
        Self {
            privacy: Arc::new(std::sync::Mutex::new(privacy)),
            storage,
        }
    }

    pub fn get_statistics(&self, epsilon: f64) -> Result<PrivateStatistics, PulsecureError> {
        let privacy = self
            .privacy
            .lock()
            .map_err(|_| PulsecureError::Privacy("DP state lock poisoned".to_string()))?;

        if !privacy.can_query(epsilon) {
            return Err(PulsecureError::Validation(format!(
                "Insufficient privacy budget: need {}, have {}",
                epsilon,
                privacy.budget_remaining()
            )));
        }

        let diagnoses = self
            .storage
            .load_diagnoses()
            .map_err(|e| PulsecureError::Storage(e.into()))?;

        let stats = privacy
            .aggregate(&diagnoses, epsilon)
            .map_err(|e| PulsecureError::Privacy(e.to_string()))?;

        tracing::info!(
            "Generated private statistics (ε={}): count~{:.0}, positive_rate~{:.1}%",
            epsilon,
            stats.total_count,
            stats.positive_rate * 100.0
        );

        Ok(stats)
    }

    #[must_use]
    pub fn budget_remaining(&self) -> f64 {
        match self.privacy.lock() {
            Ok(privacy) => privacy.budget_remaining(),
            Err(_) => 0.0,
        }
    }

    #[must_use]
    pub fn epsilon_spent(&self) -> f64 {
        match self.privacy.lock() {
            Ok(privacy) => privacy.total_epsilon_spent(),
            Err(_) => 0.0,
        }
    }

    #[must_use]
    pub fn can_query(&self, epsilon: f64) -> bool {
        match self.privacy.lock() {
            Ok(privacy) => privacy.can_query(epsilon),
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::opendp::OpenDpAdapter;
    use crate::adapters::sqlite::SqliteStorage;

    fn create_test_service() -> AnalyticsService<OpenDpAdapter, SqliteStorage> {
        let privacy = OpenDpAdapter::new();
        let storage = Arc::new(SqliteStorage::in_memory().expect("Should create db"));
        AnalyticsService::new(privacy, storage)
    }

    #[test]
    fn test_empty_statistics() {
        let service = create_test_service();
        let stats = service.get_statistics(0.1).expect("Should get stats");

        assert_eq!(stats.total_count, 0.0);
    }

    #[test]
    fn test_budget_tracking() {
        let service = create_test_service();

        let initial = service.budget_remaining();
        assert!(initial > 0.0);

        service.get_statistics(0.1).expect("Should get stats");

        assert!(service.epsilon_spent() > 0.0);
    }
}
