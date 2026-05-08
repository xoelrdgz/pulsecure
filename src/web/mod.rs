use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::{
    extract::Request,
    extract::{OriginalUri, State},
    http::{header, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::adapters::sqlite::{AuditEvent, SqliteStorage};
use crate::adapters::tfhe::{ModelMetadata, TfheAdapter};
use crate::application::InferenceService;
use crate::domain::{Diagnosis, PatientData, PatientFeatures};

type SharedInference = Arc<Mutex<InferenceService<TfheAdapter, SqliteStorage>>>;

#[derive(Clone)]
struct WebState {
    inference: SharedInference,
    storage: Arc<SqliteStorage>,
    fhe_slots: Arc<Semaphore>,
    model_metadata: ModelMetadata,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    intended_use: &'static str,
}

#[derive(Debug, Serialize)]
struct ModelResponse {
    intended_use: String,
    schema_version: u32,
    feature_count: usize,
    features: Vec<String>,
    precision_bits: u32,
    scale_factor: i64,
    clinical_threshold: Option<f64>,
    recall_at_threshold: Option<f64>,
    precision_at_threshold: Option<f64>,
    auc_calibrated: Option<f64>,
    auc_quantized: Option<f64>,
    brier_score: Option<f64>,
    dataset_sha256: Option<String>,
    dataset_n_samples: Option<usize>,
    trained_at: Option<String>,
    metadata_complete: bool,
    signed_models_required_in_production: bool,
}

#[derive(Debug, Deserialize)]
struct ScreeningRequest {
    #[serde(default)]
    patient_id: Option<String>,
    features: PatientFeatures,
}

#[derive(Debug, Serialize)]
struct ScreeningResponse {
    id: String,
    patient_id: Option<String>,
    screening_positive: bool,
    calibrated_probability: f64,
    threshold_used: f64,
    confidence: f64,
    risk_level: String,
    encrypted_computation: bool,
    created_at: String,
    limitations: &'static str,
    recommended_follow_up: &'static str,
}

#[derive(Debug, Serialize)]
struct HistoryResponse {
    items: Vec<ScreeningResponse>,
}

#[derive(Debug, Serialize)]
struct AuditResponse {
    items: Vec<AuditEvent>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

const MAX_PATIENT_REFERENCE_LEN: usize = 96;

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: self.message,
            }),
        )
            .into_response()
    }
}

pub async fn serve() -> anyhow::Result<()> {
    let state = build_state()?;
    let app = router(state);

    let addr: SocketAddr = std::env::var("PULSECURE_WEB_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:8080".to_string())
        .parse()?;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Pulsecure local web API listening on http://{addr}");

    axum::serve(listener, app).await?;
    Ok(())
}

fn router(state: WebState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/model", get(model_info))
        .route("/api/screenings", get(history).post(create_screening))
        .route("/api/audit", get(audit))
        .fallback(static_fallback)
        .layer(middleware::from_fn(add_security_headers))
        .with_state(state)
}

fn build_state() -> anyhow::Result<WebState> {
    let db_path = std::env::var("PULSECURE_DB_PATH")
        .or_else(|_| std::env::var("Pulsecure_DB_PATH"))
        .unwrap_or_else(|_| "Pulsecure.db".to_string());
    let storage = Arc::new(SqliteStorage::new(&db_path)?);

    let mut fhe = TfheAdapter::new();
    let model_path = std::env::var("PULSECURE_MODEL_PATH")
        .or_else(|_| std::env::var("Pulsecure_MODEL_PATH"))
        .unwrap_or_else(|_| "models".to_string());
    let model_dir = Path::new(&model_path);
    fhe.load_model(model_dir)
        .map_err(|e| anyhow::anyhow!("Failed to load model from {:?}: {}", model_dir, e))?;
    let model_metadata = fhe
        .model_metadata()
        .ok_or_else(|| anyhow::anyhow!("Model metadata unavailable after load"))?;

    let inference = Arc::new(Mutex::new(InferenceService::new(
        Arc::new(fhe),
        storage.clone(),
    )));
    {
        let mut svc = inference
            .lock()
            .map_err(|_| anyhow::anyhow!("Inference service lock poisoned"))?;
        svc.initialize()?;
    }

    let max_concurrent = std::env::var("PULSECURE_FHE_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1);

    Ok(WebState {
        inference,
        storage,
        fhe_slots: Arc::new(Semaphore::new(max_concurrent)),
        model_metadata,
    })
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        intended_use:
            "Proof-of-concept encrypted cardiovascular screening; not for medical decisions.",
    })
}

async fn model_info(State(state): State<WebState>) -> Json<ModelResponse> {
    let metadata = &state.model_metadata;
    let threshold = metadata.clinical_threshold.as_ref();
    let validation = metadata.validation.as_ref();
    let dataset = metadata.dataset.as_ref();
    let training = metadata.training_metadata.as_ref();

    Json(ModelResponse {
        intended_use: metadata.intended_use.clone().unwrap_or_else(|| {
            "Proof-of-concept encrypted cardiovascular screening; not for medical decisions."
                .to_string()
        }),
        schema_version: metadata.schema_version,
        feature_count: metadata.feature_names.len(),
        features: metadata.feature_names.clone(),
        precision_bits: metadata.precision_bits,
        scale_factor: metadata.scale_factor,
        clinical_threshold: threshold.map(|t| t.value),
        recall_at_threshold: threshold.and_then(|t| t.recall),
        precision_at_threshold: threshold.and_then(|t| t.precision),
        auc_calibrated: validation.and_then(|v| v.auc_calibrated),
        auc_quantized: validation.and_then(|v| v.auc_quantized),
        brier_score: validation.and_then(|v| v.brier_after),
        dataset_sha256: dataset.and_then(|d| d.sha256.clone()),
        dataset_n_samples: dataset.and_then(|d| d.n_samples),
        trained_at: training.and_then(|t| t.trained_at.clone()),
        metadata_complete: metadata.metadata_complete,
        signed_models_required_in_production: true,
    })
}

async fn create_screening(
    State(state): State<WebState>,
    Json(request): Json<ScreeningRequest>,
) -> Result<Json<ScreeningResponse>, ApiError> {
    request
        .features
        .validate_for_model(&state.model_metadata.feature_names)
        .map_err(|errors| ApiError::bad_request(errors.join(", ")))?;

    if let Some(patient_id) = request.patient_id.as_deref() {
        validate_patient_reference(patient_id)?;
    }

    let _permit = state
        .fhe_slots
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::unavailable("FHE inference queue is unavailable"))?;

    let inference = state.inference.clone();
    let patient = match request.patient_id {
        Some(patient_id) => PatientData::with_id(patient_id, request.features),
        None => PatientData::new(request.features),
    };

    let diagnosis = tokio::task::spawn_blocking(move || {
        let svc = inference.lock().map_err(|_| {
            crate::PulsecureError::Validation("Inference service lock poisoned".into())
        })?;
        svc.run_inference(patient)
    })
    .await
    .map_err(|e| ApiError::internal(format!("Inference worker failed: {e}")))?
    .map_err(|e| ApiError::internal(e.to_string()))?;

    Ok(Json(ScreeningResponse::from(diagnosis)))
}

fn validate_patient_reference(patient_id: &str) -> Result<(), ApiError> {
    if patient_id.len() > MAX_PATIENT_REFERENCE_LEN {
        return Err(ApiError::bad_request(format!(
            "Local reference must be {MAX_PATIENT_REFERENCE_LEN} characters or fewer"
        )));
    }
    if patient_id.chars().any(|c| c.is_control()) {
        return Err(ApiError::bad_request(
            "Local reference cannot contain control characters",
        ));
    }
    Ok(())
}

async fn history(State(state): State<WebState>) -> Result<Json<HistoryResponse>, ApiError> {
    let inference = state.inference.clone();
    let diagnoses = tokio::task::spawn_blocking(move || {
        let svc = inference.lock().map_err(|_| {
            crate::PulsecureError::Validation("Inference service lock poisoned".into())
        })?;
        svc.get_recent_diagnoses(50)
    })
    .await
    .map_err(|e| ApiError::internal(format!("History worker failed: {e}")))?
    .map_err(|e| ApiError::internal(e.to_string()))?;

    Ok(Json(HistoryResponse {
        items: diagnoses.into_iter().map(ScreeningResponse::from).collect(),
    }))
}

async fn audit(State(state): State<WebState>) -> Result<Json<AuditResponse>, ApiError> {
    let storage = state.storage.clone();
    let events = tokio::task::spawn_blocking(move || storage.load_recent_audit_events(50))
        .await
        .map_err(|e| ApiError::internal(format!("Audit worker failed: {e}")))?
        .map_err(|e| ApiError::internal(e.to_string()))?;

    Ok(Json(AuditResponse { items: events }))
}

async fn static_fallback(OriginalUri(uri): OriginalUri) -> Response {
    let path = uri.path();
    if path.starts_with("/api/") {
        return (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "API route not found".to_string(),
            }),
        )
            .into_response();
    }

    let relative = if path == "/" {
        "index.html".to_string()
    } else {
        path.trim_start_matches('/').to_string()
    };

    if relative.contains("..") || relative.starts_with('/') {
        return StatusCode::BAD_REQUEST.into_response();
    }

    let dist_dir = std::env::var("PULSECURE_WEB_DIST")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("web/dist"));
    let candidate = dist_dir.join(&relative);
    let file_path = if candidate.is_file() {
        candidate
    } else {
        dist_dir.join("index.html")
    };

    match tokio::fs::read(&file_path).await {
        Ok(bytes) => {
            let content_type = content_type_for(file_path.extension().and_then(|ext| ext.to_str()));
            let mut response = bytes.into_response();
            response
                .headers_mut()
                .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
            response
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            "Pulsecure web UI is not built. Run `npm install && npm run build` in web/.",
        )
            .into_response(),
    }
}

async fn add_security_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    apply_security_headers(&mut response);
    response
}

fn apply_security_headers(response: &mut Response) {
    let headers = response.headers_mut();
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        ),
    );
}

fn content_type_for(extension: Option<&str>) -> &'static str {
    match extension {
        Some("css") => "text/css; charset=utf-8",
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("json") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("wasm") => "application/wasm",
        _ => "application/octet-stream",
    }
}

impl From<Diagnosis> for ScreeningResponse {
    fn from(diagnosis: Diagnosis) -> Self {
        let recommended_follow_up = if diagnosis.result.screening_positive {
            "PoC result only. In a real workflow this would require review by a responsible healthcare professional."
        } else {
            "PoC result only. A negative screening output does not rule out disease."
        };

        Self {
            id: diagnosis.id,
            patient_id: diagnosis.patient_id,
            screening_positive: diagnosis.result.screening_positive,
            calibrated_probability: diagnosis.result.probability,
            threshold_used: diagnosis.result.threshold_used,
            confidence: diagnosis.result.confidence,
            risk_level: diagnosis.risk_level.to_string(),
            encrypted_computation: diagnosis.encrypted_computation,
            created_at: diagnosis.created_at.to_rfc3339(),
            limitations: "Proof-of-concept screening support only; not autonomous diagnosis.",
            recommended_follow_up,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patient_reference_rejects_control_characters() {
        let err = validate_patient_reference("abc\n123").expect_err("control char rejected");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn patient_reference_rejects_excessive_length() {
        let long = "x".repeat(MAX_PATIENT_REFERENCE_LEN + 1);
        let err = validate_patient_reference(&long).expect_err("long reference rejected");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn security_headers_are_applied() {
        let mut response = StatusCode::OK.into_response();
        apply_security_headers(&mut response);
        let headers = response.headers();

        assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
        assert_eq!(headers[header::REFERRER_POLICY], "no-referrer");
        assert_eq!(headers[header::X_FRAME_OPTIONS], "DENY");
        assert!(headers[header::CONTENT_SECURITY_POLICY]
            .to_str()
            .expect("valid csp")
            .contains("frame-ancestors 'none'"));
    }
}
