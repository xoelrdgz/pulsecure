//! Storage port: Trait for persistent storage operations.
//!
//! This trait abstracts the storage backend (SQLite) from the application logic.

use crate::domain::{Diagnosis, KeyPair};

/// A page of diagnoses with pagination metadata.
#[derive(Debug, Clone)]
pub struct DiagnosisPage {
    /// Diagnoses in this page
    pub items: Vec<Diagnosis>,
    /// Total count of all diagnoses (for UI pagination)
    pub total_count: usize,
    /// Current page offset
    pub offset: usize,
    /// Page size limit
    pub limit: usize,
    /// Whether there are more pages
    pub has_more: bool,
}

impl DiagnosisPage {
    /// Create a new diagnosis page.
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

    /// Get the next page offset.
    #[must_use]
    pub fn next_offset(&self) -> Option<usize> {
        if self.has_more {
            Some(self.offset + self.limit)
        } else {
            None
        }
    }

    /// Get the previous page offset.
    #[must_use]
    pub fn prev_offset(&self) -> Option<usize> {
        if self.offset > 0 {
            Some(self.offset.saturating_sub(self.limit))
        } else {
            None
        }
    }
}

/// Trait for local storage operations.
///
/// All data is stored locally and never transmitted.
pub trait Storage: Send + Sync {
    /// Error type for storage operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Save a key pair to storage.
    ///
    /// The keys are stored encrypted at rest.
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    fn save_keys(&self, keys: &KeyPair) -> Result<(), Self::Error>;

    /// Load a key pair from storage.
    ///
    /// # Returns
    /// `None` if no keys are stored.
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    fn load_keys(&self) -> Result<Option<KeyPair>, Self::Error>;

    /// Check if keys exist in storage.
    fn has_keys(&self) -> Result<bool, Self::Error>;

    /// Delete stored keys.
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    fn delete_keys(&self) -> Result<(), Self::Error>;

    /// Save a diagnosis to storage.
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    fn save_diagnosis(&self, diagnosis: &Diagnosis) -> Result<(), Self::Error>;

    /// Load all diagnoses from storage.
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    fn load_diagnoses(&self) -> Result<Vec<Diagnosis>, Self::Error>;

    /// Load recent diagnoses (up to `limit`).
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    fn load_recent_diagnoses(&self, limit: usize) -> Result<Vec<Diagnosis>, Self::Error>;

    /// Load diagnoses with pagination (cursor-based).
    ///
    /// # Arguments
    /// * `offset` - Starting position (0-indexed)
    /// * `limit` - Maximum number of items to return
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    fn load_diagnoses_paginated(&self, offset: usize, limit: usize) -> Result<DiagnosisPage, Self::Error>;

    /// Get the total count of diagnoses.
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    fn count_diagnoses(&self) -> Result<usize, Self::Error>;

    /// Delete a diagnosis by ID.
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    fn delete_diagnosis(&self, id: &str) -> Result<(), Self::Error>;

    /// Clear all data (keys and diagnoses).
    ///
    /// # Errors
    /// Returns error if storage operation fails.
    fn clear_all(&self) -> Result<(), Self::Error>;
}

