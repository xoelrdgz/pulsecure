//! Async inference worker for non-blocking FHE computation.
//!
//! This module provides a background worker that runs FHE inference
//! without blocking the TUI main loop, enabling responsive UI during
//! long-running computations.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::application::InferenceService;
use crate::domain::{Diagnosis, PatientData};
use crate::ports::{FheEngine, Storage};

/// Progress updates from the inference worker.
#[derive(Debug, Clone)]
pub enum InferenceProgress {
    /// Starting encryption phase
    Encrypting,
    /// Encryption complete, starting FHE computation
    Computing,
    /// Computation complete, decrypting result
    Decrypting,
    /// Inference complete with diagnosis result
    Complete(Diagnosis),
    /// Error occurred during inference
    Error(String),
}

/// Handle to a running inference worker.
pub struct InferenceWorkerHandle {
    /// Receiver for progress updates
    pub progress_rx: Receiver<InferenceProgress>,
    /// Thread handle (for joining)
    _handle: JoinHandle<()>,
}

impl InferenceWorkerHandle {
    /// Try to receive the next progress update (non-blocking).
    #[must_use]
    pub fn try_recv(&self) -> Option<InferenceProgress> {
        self.progress_rx.try_recv().ok()
    }
}

/// Inference worker that runs FHE computation in background.
pub struct InferenceWorker;

impl InferenceWorker {
    /// Spawn a background inference task.
    ///
    /// Returns a handle to receive progress updates.
    pub fn spawn<F, S>(
        service: Arc<Mutex<InferenceService<F, S>>>,
        patient: PatientData,
    ) -> InferenceWorkerHandle
    where
        F: FheEngine + Send + Sync + 'static,
        S: Storage + Send + Sync + 'static,
        S::Error: Into<crate::adapters::StorageError> + Send,
    {
        let (tx, rx) = mpsc::channel();

        let handle = thread::spawn(move || {
            Self::run_inference_with_progress(service, patient, tx);
        });

        InferenceWorkerHandle {
            progress_rx: rx,
            _handle: handle,
        }
    }

    /// Run inference pipeline with progress updates.
    fn run_inference_with_progress<F, S>(
        service: Arc<Mutex<InferenceService<F, S>>>,
        patient: PatientData,
        tx: Sender<InferenceProgress>,
    ) where
        F: FheEngine + Send + Sync + 'static,
        S: Storage + Send + Sync + 'static,
        S::Error: Into<crate::adapters::StorageError> + Send,
    {
        // Report encryption start
        let _ = tx.send(InferenceProgress::Encrypting);

        // Small delay to allow UI to update
        thread::sleep(std::time::Duration::from_millis(100));

        // Report computation start
        let _ = tx.send(InferenceProgress::Computing);

        // Run the actual inference (this is the blocking part)
        let result = match service.lock() {
            Ok(svc) => svc.run_inference(patient),
            Err(_) => Err(crate::PulsecureError::Validation(
                "Inference service lock poisoned".to_string(),
            )),
        };

        match result {
            Ok(diagnosis) => {
                // Report decryption (already done inside run_inference)
                let _ = tx.send(InferenceProgress::Decrypting);
                thread::sleep(std::time::Duration::from_millis(50));

                // Report completion
                let _ = tx.send(InferenceProgress::Complete(diagnosis));
            }
            Err(e) => {
                let _ = tx.send(InferenceProgress::Error(e.to_string()));
            }
        }
    }
}
