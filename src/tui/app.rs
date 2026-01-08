//! Main TUI application state machine.
//!
//! Handles:
//! - Screen navigation
//! - Input event handling
//! - Service integration
//! - Async inference via background worker

use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    Terminal,
};

use crate::adapters::opendp::OpenDpAdapter;
use crate::adapters::sqlite::SqliteStorage;
use crate::adapters::tfhe::TfheAdapter;
use crate::application::{AnalyticsService, InferenceService};
use crate::domain::PatientData;

use super::ui::{
    analytics::{render_analytics, AnalyticsState},
    dashboard::{render_dashboard, DashboardState, RecentSummary},
    inference::{render_inference, InferenceState},
    patient::{render_patient_form, PatientFormState},
    render_disclaimer,
};
use super::worker::{InferenceProgress, InferenceWorker, InferenceWorkerHandle};

/// Current screen/view in the application
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Screen {
    Dashboard,
    PatientForm,
    Inference,
    Analytics,
}

/// Main application state
pub struct App {
    /// Current screen
    screen: Screen,

    /// Whether the app should quit
    should_quit: bool,

    /// Inference service (wrapped in Arc<Mutex> for interior mutability)
    inference_service: Arc<Mutex<InferenceService<TfheAdapter, SqliteStorage>>>,

    /// Analytics service
    analytics_service: AnalyticsService<OpenDpAdapter, SqliteStorage>,

    /// Dashboard state
    dashboard_state: DashboardState,

    /// Patient form state
    patient_form_state: PatientFormState,

    /// Inference state
    inference_state: InferenceState,

    /// Analytics state
    analytics_state: AnalyticsState,

    /// Pending inference worker (if running)
    pending_worker: Option<InferenceWorkerHandle>,

    /// Current inference phase (for UI animation)
    inference_phase: Option<InferencePhase>,

    /// When the current inference phase started (for UI animation)
    inference_phase_started_at: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferencePhase {
    Encrypting,
    Computing,
    Decrypting,
}

impl App {
    /// Create a new application instance using default adapters.
    ///
    /// This is a convenience method that constructs all adapters internally.
    /// For more control, use `with_dependencies()`.
    ///
    /// # Errors
    /// Returns error if services cannot be initialized.
    pub fn new() -> Result<Self> {
        // Initialize storage
        let db_path = std::env::var("PULSECURE_DB_PATH")
            .or_else(|_| std::env::var("Pulsecure_DB_PATH"))
            .unwrap_or_else(|_| "Pulsecure.db".to_string());
        let storage = Arc::new(SqliteStorage::new(&db_path)?);

        // Initialize FHE adapter
        let mut fhe = TfheAdapter::new();

        // Load model from configured path (supports deployment flexibility)
        let model_path = std::env::var("PULSECURE_MODEL_PATH")
            .or_else(|_| std::env::var("Pulsecure_MODEL_PATH"))
            .unwrap_or_else(|_| "models".to_string());
        let model_dir = std::path::Path::new(&model_path);

        if !model_dir.exists() {
            return Err(anyhow!(
                "Model path not found at {:?}. Set PULSECURE_MODEL_PATH to a directory containing model.json or calibrated_model.json.",
                model_dir
            ));
        }

        // Hospital-grade behavior: refuse to start if model cannot be loaded/verified.
        fhe.load_model(model_dir)
            .map_err(|e| anyhow!("Failed to load model from {:?}: {}", model_dir, e))?;
        let fhe = Arc::new(fhe);

        // Create services
        let inference_service = Arc::new(Mutex::new(InferenceService::new(
            fhe.clone(),
            storage.clone(),
        )));
        let analytics_service = AnalyticsService::new(OpenDpAdapter::new(), storage.clone());

        Self::with_dependencies(inference_service, analytics_service)
    }

    /// Create application with injected dependencies (Composition Root pattern).
    ///
    /// This allows `main.rs` or tests to construct all adapters externally,
    /// providing proper dependency injection for flexibility and testability.
    ///
    /// # Arguments
    /// * `inference_service` - Pre-configured inference service
    /// * `analytics_service` - Pre-configured analytics service
    /// * `storage` - Shared storage adapter
    ///
    /// # Errors
    /// Returns error if initialization fails.
    pub fn with_dependencies(
        inference_service: Arc<Mutex<InferenceService<TfheAdapter, SqliteStorage>>>,
        analytics_service: AnalyticsService<OpenDpAdapter, SqliteStorage>,
    ) -> Result<Self> {
        Ok(Self {
            screen: Screen::Dashboard,
            should_quit: false,
            inference_service,
            analytics_service,
            dashboard_state: DashboardState::default(),
            patient_form_state: PatientFormState::default(),
            inference_state: InferenceState::default(),
            analytics_state: AnalyticsState::default(),
            pending_worker: None,
            inference_phase: None,
            inference_phase_started_at: None,
        })
    }

    /// Run the main application loop.
    ///
    /// # Errors
    /// Returns error if terminal operations fail.
    pub fn run(&mut self) -> Result<()> {
        // Initialize inference service (load/generate keys)
        {
            let mut svc = self
                .inference_service
                .lock()
                .map_err(|_| anyhow!("Inference service lock poisoned"))?;
            svc.initialize()?;
        }

        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Initial state update
        self.update_dashboard_state();

        // Main loop
        let result = self.main_loop(&mut terminal);

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        result
    }

    fn main_loop(&mut self, terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
        loop {
            // Poll pending worker for progress updates
            self.poll_worker();

            // Animate inference progress (fake loading bar)
            self.tick_inference_progress();

            // Draw current screen
            terminal.draw(|f| {
                let area = f.area();
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Min(0),
                        Constraint::Length(3),
                    ])
                    .split(area);

                let content_area = chunks[0];
                let disclaimer_area = chunks[1];

                match self.screen {
                    Screen::Dashboard => {
                        // Defense-in-depth: avoid persisting recent diagnoses in UI state.
                        // Fetch only for render and drop immediately after.
                        let recent_summary: RecentSummary = match self.inference_service.lock() {
                            Ok(svc) => match svc.get_recent_diagnoses(10) {
                                Ok(diagnoses) => {
                                    let mut summary = RecentSummary::default();
                                    summary.total = diagnoses.len();
                                    for d in diagnoses.iter() {
                                        match d.risk_level {
                                            crate::domain::RiskLevel::Low => summary.low += 1,
                                            crate::domain::RiskLevel::Moderate => {
                                                summary.moderate += 1
                                            }
                                            crate::domain::RiskLevel::High => summary.high += 1,
                                        }
                                        if d.encrypted_computation {
                                            summary.encrypted += 1;
                                        }
                                    }
                                    summary
                                }
                                Err(_) => RecentSummary::default(),
                            },
                            Err(_) => RecentSummary::default(),
                        };

                        let fp = if self.dashboard_state.show_client_fingerprint {
                            match self.inference_service.lock() {
                                Ok(svc) => svc.key_fingerprints().map(|(client_fp, _)| client_fp),
                                Err(_) => None,
                            }
                        } else {
                            None
                        };

                        render_dashboard(
                            f,
                            content_area,
                            &self.dashboard_state,
                            recent_summary,
                            fp.as_deref(),
                        );
                    }
                    Screen::PatientForm => {
                        render_patient_form(f, content_area, &self.patient_form_state)
                    }
                    Screen::Inference => render_inference(f, content_area, &self.inference_state),
                    Screen::Analytics => render_analytics(f, content_area, &self.analytics_state),
                }

                render_disclaimer(f, disclaimer_area);
            })?;

            // Handle input (short poll to stay responsive)
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    self.handle_key(key.code, key.modifiers);
                }
            }

            if self.should_quit {
                break;
            }
        }

        Ok(())
    }

    /// Poll the background worker for progress updates.
    fn poll_worker(&mut self) {
        if self.pending_worker.is_none() {
            return;
        }

        // Process all available progress messages.
        // NOTE: We must not hold an immutable borrow of `pending_worker` while mutating `self`.
        loop {
            let progress = match self
                .pending_worker
                .as_ref()
                .and_then(|worker| worker.try_recv())
            {
                Some(p) => p,
                None => break,
            };

            match progress {
                InferenceProgress::Encrypting => {
                    self.set_inference_phase(InferencePhase::Encrypting);
                }
                InferenceProgress::Computing => {
                    self.set_inference_phase(InferencePhase::Computing);
                }
                InferenceProgress::Decrypting => {
                    self.set_inference_phase(InferencePhase::Decrypting);
                }
                InferenceProgress::Complete(diagnosis) => {
                    self.inference_state = InferenceState::Complete { diagnosis };
                    self.pending_worker = None;
                    self.inference_phase = None;
                    self.inference_phase_started_at = None;
                    break;
                }
                InferenceProgress::Error(message) => {
                    self.inference_state = InferenceState::Error { message };
                    self.pending_worker = None;
                    self.inference_phase = None;
                    self.inference_phase_started_at = None;
                    break;
                }
            }
        }
    }

    fn set_inference_phase(&mut self, phase: InferencePhase) {
        let now = Instant::now();
        let current_progress = match &self.inference_state {
            InferenceState::Encrypting { progress }
            | InferenceState::Computing { progress }
            | InferenceState::Decrypting { progress } => *progress,
            _ => 0.0,
        };

        let min_start = match phase {
            InferencePhase::Encrypting => 0.0,
            InferencePhase::Computing => 0.35,
            InferencePhase::Decrypting => 0.90,
        };
        let progress = current_progress.max(min_start);

        self.inference_phase = Some(phase);
        self.inference_phase_started_at = Some(now);

        self.inference_state = match phase {
            InferencePhase::Encrypting => InferenceState::Encrypting { progress },
            InferencePhase::Computing => InferenceState::Computing { progress },
            InferencePhase::Decrypting => InferenceState::Decrypting { progress },
        };
    }

    fn tick_inference_progress(&mut self) {
        // Only animate while a worker is running and we're in a progress state.
        if self.pending_worker.is_none() {
            return;
        }

        let Some(phase) = self.inference_phase else {
            return;
        };
        let Some(started_at) = self.inference_phase_started_at else {
            return;
        };

        let now = Instant::now();
        let elapsed = now.saturating_duration_since(started_at).as_secs_f64();

        let (start_floor, target, tau) = match phase {
            InferencePhase::Encrypting => (0.02, 0.35, 1.2),
            InferencePhase::Computing => (0.35, 0.90, 6.0),
            InferencePhase::Decrypting => (0.90, 0.98, 1.8),
        };

        let current_progress = match &self.inference_state {
            InferenceState::Encrypting { progress }
            | InferenceState::Computing { progress }
            | InferenceState::Decrypting { progress } => *progress,
            _ => return,
        };

        // Smooth, monotonic fake progress: asymptotically approaches the phase target.
        let k = if tau <= 0.0 { 1.0 } else { 1.0 - (-elapsed / tau).exp() };
        let desired = (start_floor + (target - start_floor) * k).clamp(0.0, target);
        let new_progress = desired.max(current_progress).min(target);

        self.inference_state = match phase {
            InferencePhase::Encrypting => InferenceState::Encrypting {
                progress: new_progress,
            },
            InferencePhase::Computing => InferenceState::Computing {
                progress: new_progress,
            },
            InferencePhase::Decrypting => InferenceState::Decrypting {
                progress: new_progress,
            },
        };
    }

    fn handle_key(&mut self, key: KeyCode, modifiers: KeyModifiers) {
        // Global quit handling
        if key == KeyCode::Char('q') && modifiers.contains(KeyModifiers::CONTROL) {
            self.should_quit = true;
            return;
        }

        match self.screen {
            Screen::Dashboard => self.handle_dashboard_key(key),
            Screen::PatientForm => self.handle_patient_form_key(key),
            Screen::Inference => self.handle_inference_key(key),
            Screen::Analytics => self.handle_analytics_key(key),
        }
    }

    fn handle_dashboard_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('n') | KeyCode::Char('N') => {
                self.patient_form_state = PatientFormState::default();
                self.screen = Screen::PatientForm;
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                self.load_analytics();
                self.screen = Screen::Analytics;
            }
            KeyCode::Char('k') | KeyCode::Char('K') => {
                match self.inference_service.lock() {
                    Ok(mut svc) => {
                        if let Err(e) = svc.regenerate_keys() {
                            tracing::error!("Failed to regenerate keys: {}", e);
                        }
                    }
                    Err(_) => {
                        tracing::error!("Inference service lock poisoned; cannot regenerate keys");
                    }
                }
                self.update_dashboard_state();
            }
            KeyCode::Char('f') | KeyCode::Char('F') => {
                self.dashboard_state.show_client_fingerprint =
                    !self.dashboard_state.show_client_fingerprint;
            }
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    fn handle_patient_form_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc => {
                self.screen = Screen::Dashboard;
            }
            KeyCode::Up => {
                self.patient_form_state.prev_field();
            }
            KeyCode::Down | KeyCode::Tab => {
                self.patient_form_state.next_field();
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                self.patient_form_state.load_sample_data();
            }
            KeyCode::Char(c) => {
                self.patient_form_state.input_char(c);
            }
            KeyCode::Backspace => {
                self.patient_form_state.delete_char();
            }
            KeyCode::Delete => {
                self.patient_form_state.clear_field();
            }
            KeyCode::Enter => {
                self.submit_patient_form();
            }
            _ => {}
        }
    }

    fn handle_inference_key(&mut self, key: KeyCode) {
        match &self.inference_state {
            InferenceState::Complete { .. } => match key {
                KeyCode::Enter | KeyCode::Esc => {
                    self.update_dashboard_state();
                    self.screen = Screen::Dashboard;
                }
                KeyCode::Char('n') | KeyCode::Char('N') => {
                    self.patient_form_state = PatientFormState::default();
                    self.screen = Screen::PatientForm;
                }
                _ => {}
            },
            InferenceState::Error { .. } => match key {
                KeyCode::Enter => {
                    // Could implement retry here
                    self.screen = Screen::PatientForm;
                }
                KeyCode::Esc => {
                    self.screen = Screen::Dashboard;
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn handle_analytics_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc => {
                self.screen = Screen::Dashboard;
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                self.load_analytics();
            }
            _ => {}
        }
    }

    fn submit_patient_form(&mut self) {
        match self.patient_form_state.to_patient_features() {
            Ok(features) => {
                // Validate
                if let Err(errors) = features.validate() {
                    self.patient_form_state.error_message = Some(errors.join(", "));
                    return;
                }

                // Create patient data
                let patient = PatientData::new(features);

                // Switch to inference screen with initial state
                self.screen = Screen::Inference;
                self.inference_state = InferenceState::Encrypting { progress: 0.0 };
                self.inference_phase = Some(InferencePhase::Encrypting);
                self.inference_phase_started_at = Some(Instant::now());

                // Spawn background worker for non-blocking FHE computation
                let worker = InferenceWorker::spawn(self.inference_service.clone(), patient);
                self.pending_worker = Some(worker);

                // Clear plaintext buffers from the UI immediately.
                self.patient_form_state.clear_sensitive();
            }
            Err(e) => {
                self.patient_form_state.error_message = Some(e);
            }
        }
    }

    fn update_dashboard_state(&mut self) {
        let service = match self.inference_service.lock() {
            Ok(svc) => svc,
            Err(_) => {
                // Fail-closed in UI: show no keys, but don't crash the app.
                self.dashboard_state.keys_generated = false;
                self.dashboard_state.privacy_budget = self.analytics_service.budget_remaining();
                return;
            }
        };

        // Update key status
        self.dashboard_state.keys_generated = service.is_initialized();

        // Model is always "loaded" in simulation mode
        self.dashboard_state.model_loaded = true;

        if let Ok(count) = service.get_diagnosis_count() {
            self.dashboard_state.diagnosis_count = count;
        }

        // Drop the lock before accessing analytics (avoid potential deadlock)
        drop(service);

        // Update budgets
        self.dashboard_state.privacy_budget = self.analytics_service.budget_remaining();
    }

    fn load_analytics(&mut self) {
        self.analytics_state.loading = true;
        self.analytics_state.error = None;

        // Check budget
        let epsilon = 0.1;
        if !self.analytics_service.can_query(epsilon) {
            self.analytics_state.loading = false;
            self.analytics_state.error = Some("Privacy budget exhausted".to_string());
            return;
        }

        // Get statistics
        match self.analytics_service.get_statistics(epsilon) {
            Ok(stats) => {
                self.analytics_state.statistics = Some(stats);
                self.analytics_state.loading = false;
            }
            Err(e) => {
                self.analytics_state.loading = false;
                self.analytics_state.error = Some(e.to_string());
            }
        }

        self.analytics_state.epsilon_used = self.analytics_service.epsilon_spent();
        self.analytics_state.budget_remaining = self.analytics_service.budget_remaining();
    }
}
