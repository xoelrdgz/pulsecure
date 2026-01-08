//! Pulsecure: Privacy-Preserving Medical ML Pipeline
//!
//! Main entry point for the terminal application.

#![allow(non_snake_case)]

use anyhow::Result;
use std::io::IsTerminal;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use Pulsecure::tui::App;
use Pulsecure::adapters::sanitize::SanitizingMakeWriter;

fn main() -> Result<()> {
    // Initialize logging.
    //
    // IMPORTANT: writing logs to the terminal will corrupt the TUI (alternate screen).
    // Default behavior:
    // - interactive TTY: log to a file (persisted via /app/data volume in Docker)
    // - non-interactive: log to stdout (so `docker logs` works)
    let log_mode = std::env::var("PULSECURE_LOG_MODE")
        .or_else(|_| std::env::var("Pulsecure_LOG_MODE"))
        .unwrap_or_else(|_| "auto".to_string());

    let interactive = std::io::stdout().is_terminal();
    let use_file = match log_mode.as_str() {
        "file" => true,
        "stdout" => false,
        // auto
        _ => interactive,
    };

    let (writer, _guard) = if use_file {
        let log_file = std::env::var("PULSECURE_LOG_FILE")
            .or_else(|_| std::env::var("Pulsecure_LOG_FILE"))
            .unwrap_or_else(|_| "/app/data/pulsecure.log".to_string());

        if let Some(parent) = std::path::Path::new(&log_file).parent() {
            // Best-effort: don't fail startup just because the directory is missing.
            let _ = std::fs::create_dir_all(parent);
        }

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file)?;
        tracing_appender::non_blocking(file)
    } else {
        tracing_appender::non_blocking(std::io::stdout())
    };

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer().with_writer(SanitizingMakeWriter::new(writer)))
        .init();

    tracing::info!("Starting Pulsecure...");

    // Run the TUI application
    let mut app = App::new()?;
    app.run()?;

    tracing::info!("Pulsecure shutdown complete.");
    Ok(())
}
