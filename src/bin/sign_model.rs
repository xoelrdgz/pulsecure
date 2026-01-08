//! Model signing utility for Pulsecure ML models.
//!
//! Creates a signed manifest (`manifest.json`) and Ed25519 signature (`model.sig`)
//! for model files, enabling cryptographic verification at runtime.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin sign_model -- <model_dir> [--serial <n>] [--nonce-b64 <b64>]
//! ```
//!
//! # Security
//!
//! - Signing key sourced from secure locations (FD, file, Docker secret)
//! - Manifest includes SHA-256 hashes of all bound files
//! - Anti-rollback fields: serial number, creation timestamp, random nonce
//! - Private key material zeroized after use

use std::collections::BTreeMap;
use std::env;
use std::fs;
#[cfg(unix)]
use std::os::unix::io::FromRawFd;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose;
use base64::Engine;
use ed25519_dalek::{Signature, Signer, SigningKey};
use rand::RngCore;
use serde::Serialize;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(Debug, Serialize)]
struct SignedModelManifest {
    version: u32,
    /// Monotonic serial number for anti-rollback.
    ///
    /// Recommended: a CI build number or other strictly increasing value.
    /// If not provided, defaults to `created_at`.
    serial: u64,
    /// Unix timestamp (seconds) when this manifest was created.
    /// Used for anti-rollback checks during verification.
    created_at: i64,
    /// Random nonce (base64, 16 bytes) for uniqueness.
    nonce_b64: String,
    files: BTreeMap<String, String>,
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed to read {path:?}: {e}"))?;
    Ok(to_hex(&Sha256::digest(&bytes)))
}

#[derive(Zeroize, ZeroizeOnDrop)]
struct Seed([u8; 32]);

fn read_signing_seed_b64() -> Result<Zeroizing<String>, String> {
    const KEY_FD_ENV: &str = "PULSECURE_MODEL_SIGNING_KEY_B64_FD";
    const KEY_FILE_ENV: &str = "PULSECURE_MODEL_SIGNING_KEY_B64_FILE";
    const DOCKER_SECRET_PATH: &str = "/run/secrets/pulsecure_model_signing_key_b64";

    #[cfg(unix)]
    if let Ok(fd_str) = env::var(KEY_FD_ENV) {
        let fd: i32 = fd_str
            .trim()
            .parse()
            .map_err(|_| "Invalid key FD".to_string())?;
        if fd <= 2 {
            return Err("Refusing to read signing key from stdio FD".to_string());
        }
        // SAFETY: take ownership of FD for one-time secret read.
        let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
        let mut buf = String::new();
        use std::io::Read;
        file.read_to_string(&mut buf)
            .map_err(|e| format!("Failed reading signing key from FD: {e}"))?;
        let secret = buf.trim_end_matches(['\n', '\r']).to_string();
        if secret.is_empty() {
            return Err("Empty signing key".to_string());
        }
        return Ok(Zeroizing::new(secret));
    }

    if let Ok(path) = env::var(KEY_FILE_ENV) {
        let content = fs::read_to_string(path.trim())
            .map_err(|e| format!("Failed reading signing key file: {e}"))?;
        let secret = content.trim_end_matches(['\n', '\r']).to_string();
        if secret.is_empty() {
            return Err("Empty signing key".to_string());
        }
        return Ok(Zeroizing::new(secret));
    }

    if Path::new(DOCKER_SECRET_PATH).exists() {
        let content = fs::read_to_string(DOCKER_SECRET_PATH)
            .map_err(|e| format!("Failed reading docker secret: {e}"))?;
        let secret = content.trim_end_matches(['\n', '\r']).to_string();
        if secret.is_empty() {
            return Err("Empty signing key".to_string());
        }
        return Ok(Zeroizing::new(secret));
    }

    // Dev-only fallback for convenience.
    if cfg!(debug_assertions) {
        if let Ok(v) = env::var("PULSECURE_MODEL_SIGNING_KEY_B64")
            .or_else(|_| env::var("Pulsecure_MODEL_SIGNING_KEY_B64"))
        {
            let secret = v.trim_end_matches(['\n', '\r']).to_string();
            if secret.is_empty() {
                return Err("Empty signing key".to_string());
            }
            return Ok(Zeroizing::new(secret));
        }
    }

    Err(
        "Missing signing key. Provide one of: PULSECURE_MODEL_SIGNING_KEY_B64_FD, PULSECURE_MODEL_SIGNING_KEY_B64_FILE, or /run/secrets/pulsecure_model_signing_key_b64 (env var fallback only in debug builds)."
            .to_string(),
    )
}

fn read_signing_seed() -> Result<Seed, String> {
    let v = read_signing_seed_b64()?;

    let raw = general_purpose::STANDARD
        .decode(v.trim())
        .map_err(|e| format!("Invalid base64 in signing key: {e}"))?;

    if raw.len() != 32 {
        return Err(format!(
            "Signing key seed must be 32 bytes after base64 decode (got {})",
            raw.len()
        ));
    }

    let mut seed = [0u8; 32];
    seed.copy_from_slice(&raw);
    Ok(Seed(seed))
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn usage() -> String {
    "Usage: sign_model <model_dir> [--serial <u64>] [--nonce-b64 <b64_16_bytes>]".to_string()
}

fn parse_args() -> Result<(PathBuf, Option<u64>, Option<String>), String> {
    let mut args = env::args().skip(1);
    let mut model_dir: Option<PathBuf> = None;
    let mut serial: Option<u64> = None;
    let mut nonce_b64: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--serial" => {
                let v = args.next().ok_or_else(|| usage())?;
                let parsed = v
                    .trim()
                    .parse::<u64>()
                    .map_err(|_| "--serial must be a u64".to_string())?;
                serial = Some(parsed);
            }
            "--nonce-b64" => {
                let v = args.next().ok_or_else(|| usage())?;
                nonce_b64 = Some(v);
            }
            "-h" | "--help" => return Err(usage()),
            _ => {
                if model_dir.is_none() {
                    model_dir = Some(PathBuf::from(arg));
                } else {
                    return Err(usage());
                }
            }
        }
    }

    let model_dir = model_dir.ok_or_else(|| usage())?;
    Ok((model_dir, serial, nonce_b64))
}

fn make_nonce_b64() -> String {
    let mut nonce = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    general_purpose::STANDARD.encode(nonce)
}

fn validate_nonce_b64(nonce_b64: &str) -> Result<(), String> {
    let raw = general_purpose::STANDARD
        .decode(nonce_b64.trim())
        .map_err(|e| format!("Invalid base64 nonce: {e}"))?;
    if raw.len() != 16 {
        return Err("nonce must decode to exactly 16 bytes".to_string());
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let (model_dir, serial_arg, nonce_arg) = parse_args()?;

    let model_dir = if model_dir.is_file() {
        model_dir
            .parent()
            .ok_or_else(|| "Model path has no parent directory".to_string())?
            .to_path_buf()
    } else {
        model_dir
    };

    let mut seed = read_signing_seed()?;
    let signing_key = SigningKey::from_bytes(&seed.0);
    let verifying_key = signing_key.verifying_key();

    let candidates = ["model.json", "calibrated_model.json"];
    let mut files: BTreeMap<String, String> = BTreeMap::new();

    for rel in candidates {
        let p = model_dir.join(rel);
        if p.exists() {
            files.insert(rel.to_string(), sha256_hex(&p)?);
        }
    }

    if files.is_empty() {
        return Err(format!(
            "No model JSON found in {model_dir:?} (expected model.json or calibrated_model.json)"
        ));
    }

    let created_at = unix_now();
    let serial = serial_arg.unwrap_or_else(|| if created_at > 0 { created_at as u64 } else { 1 });

    let nonce_b64 = match nonce_arg {
        Some(v) => {
            validate_nonce_b64(&v)?;
            v
        }
        None => make_nonce_b64(),
    };

    let manifest = SignedModelManifest {
        version: 1,
        serial,
        created_at,
        nonce_b64,
        files,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| format!("Failed to serialize manifest.json: {e}"))?;

    let manifest_path = model_dir.join("manifest.json");
    fs::write(&manifest_path, &manifest_bytes)
        .map_err(|e| format!("Failed to write {manifest_path:?}: {e}"))?;

    let sig: Signature = signing_key.sign(&manifest_bytes);
    let sig_path = model_dir.join("model.sig");
    fs::write(&sig_path, sig.to_bytes())
        .map_err(|e| format!("Failed to write {sig_path:?}: {e}"))?;

    println!("Signed manifest: {manifest_path:?}");
    println!("Wrote signature: {sig_path:?}");
    println!("DEV_PUBKEY (hex)={}", to_hex(verifying_key.as_bytes()));

    // Best-effort: wipe seed from memory.
    seed.zeroize();

    Ok(())
}
