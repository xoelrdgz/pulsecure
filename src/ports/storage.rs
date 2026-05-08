use crate::domain::{Diagnosis, KeyPair};

#[derive(Debug, Clone)]
pub struct DiagnosisPage {
    pub items: Vec<Diagnosis>,

    pub total_count: usize,

    pub offset: usize,

    pub limit: usize,

    pub has_more: bool,
}

impl DiagnosisPage {
    #[must_use]
    pub fn new(items: Vec<Diagnosis>, total_count: usize, offset: usize, limit: usize) -> Self {
        let has_more = offset + items.len() < total_count;
        Self {
            items,
            total_count,
            offset,
            limit,
            has_more,
        }
    }

    #[must_use]
    pub fn next_offset(&self) -> Option<usize> {
        if self.has_more {
            Some(self.offset + self.limit)
        } else {
            None
        }
    }

    #[must_use]
    pub fn prev_offset(&self) -> Option<usize> {
        if self.offset > 0 {
            Some(self.offset.saturating_sub(self.limit))
        } else {
            None
        }
    }
}

pub trait Storage: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn save_keys(&self, keys: &KeyPair) -> Result<(), Self::Error>;

    fn load_keys(&self) -> Result<Option<KeyPair>, Self::Error>;

    fn has_keys(&self) -> Result<bool, Self::Error>;

    fn delete_keys(&self) -> Result<(), Self::Error>;

    fn save_diagnosis(&self, diagnosis: &Diagnosis) -> Result<(), Self::Error>;

    fn load_diagnoses(&self) -> Result<Vec<Diagnosis>, Self::Error>;

    fn load_recent_diagnoses(&self, limit: usize) -> Result<Vec<Diagnosis>, Self::Error>;

    fn load_diagnoses_paginated(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<DiagnosisPage, Self::Error>;

    fn count_diagnoses(&self) -> Result<usize, Self::Error>;

    fn delete_diagnosis(&self, id: &str) -> Result<(), Self::Error>;

    fn clear_all(&self) -> Result<(), Self::Error>;
}
