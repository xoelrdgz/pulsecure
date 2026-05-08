#![allow(non_snake_case)]

use anyhow::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use Pulsecure::adapters::sanitize::SanitizingMakeWriter;

#[tokio::main]
async fn main() -> Result<()> {
    let log_mode = std::env::var("PULSECURE_LOG_MODE")
        .or_else(|_| std::env::var("Pulsecure_LOG_MODE"))
        .unwrap_or_else(|_| "stdout".to_string());

    let use_file = log_mode == "file";

    let (writer, _guard) = if use_file {
        let log_file = std::env::var("PULSECURE_LOG_FILE")
            .or_else(|_| std::env::var("Pulsecure_LOG_FILE"))
            .unwrap_or_else(|_| "/app/data/pulsecure.log".to_string());

        if let Some(parent) = std::path::Path::new(&log_file).parent() {
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

    tracing::info!("Starting Pulsecure web server...");
    Pulsecure::web::serve().await?;
    tracing::info!("Pulsecure shutdown complete.");
    Ok(())
}
