# Pulsecure Threat Model

Version: 1.0
Methodology: STRIDE

## System Overview

Pulsecure is a privacy-preserving medical ML system using Fully Homomorphic Encryption (FHE) for cardiovascular disease risk prediction. The system processes Protected Health Information (PHI).

### Trust Boundaries

1. Client Machine: TUI runs here, holds ClientKey
2. Model Files: Pre-trained, signed by developer
3. SQLite Database: Encrypted keys, diagnosis history

## Assets

| Asset | Sensitivity | Location |
|-------|-------------|----------|
| Patient Data (PHI) | Critical | Memory during processing |
| ClientKey | Critical | Encrypted in SQLite |
| ServerKey | High | Encrypted in SQLite |
| Diagnosis Results | High | SQLite database |
| ML Model Weights | Medium | Disk (signed) |

## Threat Analysis (STRIDE)

### Spoofing

| ID | Threat | Mitigation |
|----|--------|------------|
| S1 | Malicious model injection | Ed25519 signature verification |
| S2 | Fake diagnostic results | Key fingerprint verification |

### Tampering

| ID | Threat | Mitigation |
|----|--------|------------|
| T1 | Model file modification | Mandatory signature in release builds |
| T2 | Database tampering | AES-256-GCM encryption |
| T3 | Cache poisoning (datasets) | SHA-256 hash verification |

### Repudiation

| ID | Threat | Mitigation |
|----|--------|------------|
| R1 | Deny diagnosis was performed | Diagnosis stored with timestamp |

### Information Disclosure

| ID | Threat | Mitigation |
|----|--------|------------|
| I1 | Key leakage via logs | Debug trait hides key bytes |
| I2 | Memory disclosure | Zeroize on key drop |
| I3 | Side-channel via epsilon | Fixed-point epsilon tracking |
| I4 | PII in logs | Sanitization layer |

### Denial of Service

| ID | Threat | Mitigation |
|----|--------|------------|
| D1 | Resource exhaustion | Docker memory limits (8GB) |
| D2 | Privacy budget exhaustion | Budget tracking with reset |

### Elevation of Privilege

| ID | Threat | Mitigation |
|----|--------|------------|
| E1 | Container escape | no-new-privileges, read-only filesystem |
| E2 | Network exfiltration | network_mode: none |

## FHE Implementation

This system uses genuine Fully Homomorphic Encryption via tfhe-rs. Patient data is encrypted and processed homomorphically. The server never sees plaintext patient features.

### Quantization Security

- 12-bit fixed-point precision
- 64-bit accumulators (no overflow)
- Sigmoid via 256-entry lookup table
- Calibration via 64-entry lookup table

## Compliance Considerations

- HIPAA: Air-gapped deployment, encryption at rest
- GDPR: Local processing only, no data transmission

## Review Schedule

This threat model should be reviewed:

- Before each major release
- After any security incident
- Annually at minimum
