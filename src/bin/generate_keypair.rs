//! Ed25519 keypair generation utility for model signing.
//!
//! Generates a cryptographically secure Ed25519 signing keypair:
//! - Private seed (32 bytes) written to file with 0600 permissions
//! - Public key optionally written separately
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin generate_keypair -- --out-seed <path> [--out-pub <path>] [--force]
//! ```
//!
//! # Security
//!
//! - Uses OS entropy (OsRng) for key generation
//! - Private seed is zeroized from memory after use
//! - Output file has restricted permissions (Unix only)

use base64::engine::general_purpose;
use base64::Engine;
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use rand::RngCore;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use zeroize::Zeroize;
use zeroize::Zeroizing;

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut out_seed_path: Option<std::path::PathBuf> = None;
    let mut out_pub_path: Option<std::path::PathBuf> = None;
    let mut force = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            // Back-compat: --out writes the SEED.
            "--out" | "--out-seed" => {
                let p = args.next().unwrap_or_default();
                if p.is_empty() {
                    eprintln!(
                        "Usage: generate_keypair --out-seed <path> [--out-pub <path>] [--force]"
                    );
                    std::process::exit(2);
                }
                out_seed_path = Some(std::path::PathBuf::from(p));
            }
            "--out-pub" => {
                let p = args.next().unwrap_or_default();
                if p.is_empty() {
                    eprintln!(
                        "Usage: generate_keypair --out-seed <path> [--out-pub <path>] [--force]"
                    );
                    std::process::exit(2);
                }
                out_pub_path = Some(std::path::PathBuf::from(p));
            }
            "--force" => force = true,
            "-h" | "--help" => {
                println!(
                    "Usage: generate_keypair --out-seed <path> [--out-pub <path>] [--force]\n\nWrites the base64 Ed25519 seed to <path> with 0600 permissions. Optionally writes the base64 public key to --out-pub. Prints only non-secret material."
                );
                return;
            }
            _ => {
                eprintln!("Unknown arg: {arg}\nUsage: generate_keypair --out-seed <path> [--out-pub <path>] [--force]");
                std::process::exit(2);
            }
        }
    }

    let out_seed_path = out_seed_path.unwrap_or_else(|| {
        eprintln!("Usage: generate_keypair --out-seed <path> [--out-pub <path>] [--force]");
        std::process::exit(2);
    });

    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);

    let signing_key = SigningKey::from_bytes(&seed);
    let verifying_key = signing_key.verifying_key();

    let seed_b64 = Zeroizing::new(general_purpose::STANDARD.encode(seed));
    let pub_b64 = general_purpose::STANDARD.encode(verifying_key.as_bytes());
    let pub_hex = to_hex(verifying_key.as_bytes());

    if out_seed_path.exists() && !force {
        eprintln!(
            "Refusing to overwrite existing file {:?}. Use --force.",
            out_seed_path
        );
        std::process::exit(3);
    }

    if let Some(pub_path) = &out_pub_path {
        if pub_path.exists() && !force {
            eprintln!(
                "Refusing to overwrite existing file {:?}. Use --force.",
                pub_path
            );
            std::process::exit(3);
        }
    }

    if let Some(parent) = out_seed_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        opts.mode(0o600);
    }

    let mut file = opts.open(&out_seed_path).unwrap_or_else(|e| {
        eprintln!("Failed to open {:?}: {e}", out_seed_path);
        std::process::exit(4);
    });

    use std::io::Write;
    file.write_all(seed_b64.as_bytes()).unwrap();
    file.write_all(b"\n").unwrap();

    if let Some(pub_path) = &out_pub_path {
        if let Some(parent) = pub_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let mut pub_opts = std::fs::OpenOptions::new();
        pub_opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            // Public key is non-secret; allow read access.
            pub_opts.mode(0o644);
        }

        let mut pub_file = pub_opts.open(pub_path).unwrap_or_else(|e| {
            eprintln!("Failed to open {:?}: {e}", pub_path);
            std::process::exit(4);
        });
        pub_file.write_all(pub_b64.as_bytes()).unwrap();
        pub_file.write_all(b"\n").unwrap();
    }

    // Print only non-secret material.
    println!("Wrote signing seed (base64) to {:?}", out_seed_path);
    if let Some(pub_path) = &out_pub_path {
        println!("Wrote public key (base64) to {:?}", pub_path);
    }
    println!("DEV_PUBKEY (hex)={pub_hex}");

    // Best-effort: wipe seed from memory.
    seed.zeroize();
}
