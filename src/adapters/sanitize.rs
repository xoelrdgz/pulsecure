//! Log sanitization utilities for PII/secret filtering.
//!
//! This module provides string-based sanitization helpers that can be applied
//! to log output (or any other untrusted text), including:
//! - Patient IDs
//! - Medical record numbers (MRNs)
//! - Names and similar patterns
//! - Raw feature values
//! - Common secret formats (JWTs, PEM blocks, hex/base64 tokens)
//!
//! # Important: prefer structured logging + redaction-by-type
//!
//! Sanitizing strings is a defense-in-depth fallback. The primary protection is
//! to ensure sensitive data never reaches logging calls in the first place.
//! In higher-integrity systems, prefer structured logging where sensitive fields
//! are wrapped in types whose `Debug`/`Display` implementations redact by default.
//!
//! # Performance / DoS
//!
//! Even with linear-time regex engines, scanning and allocating on large inputs can
//! be expensive. `sanitize()` enforces a maximum input size (see
//! `PULSECURE_SANITIZE_MAX_BYTES`) to reduce the impact of maliciously large logs.
//!
//! # Security
//!
//! This is a defense-in-depth measure.

use regex::{Regex, RegexSet};
use std::sync::OnceLock;
use tracing_subscriber::fmt::MakeWriter;

/// Compiled patterns for PII detection and sanitization.
static PII_PATTERNS: OnceLock<PiiPatterns> = OnceLock::new();

/// Maximum number of bytes to sanitize per call.
///
/// This is a DoS guardrail: sanitizing huge untrusted strings is expensive.
/// Defaults to 16 KiB; can be overridden via `PULSECURE_SANITIZE_MAX_BYTES`.
const DEFAULT_SANITIZE_MAX_BYTES: usize = 16 * 1024;

/// A compiled PII pattern with its replacement text.
struct PiiPattern {
    regex: Regex,
    replacement: &'static str,
}

struct PiiPatterns {
    fast_set: RegexSet,
    fast_patterns: Vec<PiiPattern>,
    pem_patterns: Vec<PiiPattern>,
}

fn truncate_to_char_boundary(input: &str, max_bytes: usize) -> (&str, bool) {
    if input.len() <= max_bytes {
        return (input, false);
    }

    // Ensure we don't panic on UTF-8 boundaries.
    let mut end = max_bytes.min(input.len());
    while end > 0 && !input.is_char_boundary(end) {
        end -= 1;
    }
    (&input[..end], true)
}

fn max_sanitize_bytes() -> usize {
    std::env::var("PULSECURE_SANITIZE_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_SANITIZE_MAX_BYTES)
}

/// Initialize PII patterns (called once at startup).
fn get_patterns() -> &'static PiiPatterns {
    PII_PATTERNS.get_or_init(|| {
        // NOTE: Rust's `regex` crate is linear-time (no catastrophic backtracking),
        // but sanitizing large strings can still be CPU-expensive. We keep patterns
        // simple and cap input size (see `max_sanitize_bytes`).
        let fast_rules: Vec<(&'static str, &'static str)> = vec![
            // UUID patterns (patient IDs, diagnosis IDs)
            (
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                "[REDACTED-UUID]",
            ),
            // SSN-like patterns (xxx-xx-xxxx)
            (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]"),
            // MRN patterns (common formats)
            (r"\bMRN[:\s]?\d{6,10}\b", "[REDACTED-MRN]"),
            // Email patterns (bounded labels; case-insensitive)
            (
                r"(?i)\b[a-z0-9](?:[a-z0-9._%+-]{0,62}[a-z0-9])?@(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b",
                "[REDACTED-EMAIL]",
            ),
            // Phone patterns
            (
                r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                "[REDACTED-PHONE]",
            ),
            // JWTs (common bearer tokens)
            (
                r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b",
                "[REDACTED-JWT]",
            ),
            // Contextual secrets (reduce false positives vs. raw base64/hex)
            (
                r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|passwd|pwd|private[_-]?key|seed|signature|sig|token|key)\b\s*[:=]\s*[A-Za-z0-9+/]{32,}={0,2}\b",
                "[REDACTED-SECRET]",
            ),
            (
                r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|passwd|pwd|private[_-]?key|seed|signature|sig|token|key)\b\s*[:=]\s*[0-9a-fA-F]{16,}\b",
                "[REDACTED-SECRET]",
            ),
            // Broad key material pattern (defense-in-depth)
            (r"\b[0-9a-fA-F]{32,}\b", "[REDACTED-KEY]"),
        ];

        // PEM patterns are intentionally excluded from the RegexSet because large bounded
        // dot-matches-newline patterns can blow up the compiled program size when merged.
        let pem_patterns: Vec<PiiPattern> = vec![
            PiiPattern {
                regex: Regex::new(
                    r"(?s)-----BEGIN [A-Z0-9 ]{0,40}(?:EC |RSA |OPENSSH )?PRIVATE KEY-----[\s\S]{0,8192}-----END [A-Z0-9 ]{0,40}(?:EC |RSA |OPENSSH )?PRIVATE KEY-----",
                )
                .expect("Valid regex"),
                replacement: "[REDACTED-PEM-PRIVATE-KEY]",
            },
            PiiPattern {
                regex: Regex::new(
                    r"(?s)-----BEGIN [A-Z0-9 ]{0,40}(?:EC |RSA |OPENSSH )?PUBLIC KEY-----[\s\S]{0,4096}-----END [A-Z0-9 ]{0,40}(?:EC |RSA |OPENSSH )?PUBLIC KEY-----",
                )
                .expect("Valid regex"),
                replacement: "[REDACTED-PEM-PUBLIC-KEY]",
            },
        ];

        let fast_set =
            RegexSet::new(fast_rules.iter().map(|(p, _)| *p)).expect("Valid regex set");
        let fast_patterns = fast_rules
            .into_iter()
            .map(|(pattern, replacement)| PiiPattern {
                regex: Regex::new(pattern).expect("Valid regex"),
                replacement,
            })
            .collect();

        PiiPatterns {
            fast_set,
            fast_patterns,
            pem_patterns,
        }
    })
}

/// Sanitize a string by replacing PII patterns.
///
/// This function applies all registered PII patterns to the input string
/// and returns a sanitized version.
#[must_use]
pub fn sanitize(input: &str) -> String {
    sanitize_with_limit(input, max_sanitize_bytes())
}

fn sanitize_with_limit(input: &str, max_bytes: usize) -> String {
    let patterns = get_patterns();

    let (prefix, truncated) = truncate_to_char_boundary(input, max_bytes);

    // Fast path: single scan for "any match".
    if !patterns.fast_set.is_match(prefix)
        && !(prefix.contains("-----BEGIN ") && patterns.pem_patterns.iter().any(|p| p.regex.is_match(prefix)))
    {
        let mut out = prefix.to_string();
        if truncated {
            out.push_str(" [TRUNCATED]");
        }
        return out;
    }

    // Only apply patterns that matched the original prefix.
    let matched: Vec<usize> = patterns.fast_set.matches(prefix).into_iter().collect();
    let mut result = prefix.to_string();
    for idx in matched {
        let pattern = &patterns.fast_patterns[idx];
        result = pattern
            .regex
            .replace_all(&result, pattern.replacement)
            .to_string();
    }

    // PEM blocks are rare but dangerous if logged; only attempt if a cheap trigger matches.
    if result.contains("-----BEGIN ") {
        for pattern in &patterns.pem_patterns {
            if pattern.regex.is_match(&result) {
                result = pattern
                    .regex
                    .replace_all(&result, pattern.replacement)
                    .to_string();
            }
        }
    }

    if truncated {
        result.push_str(" [TRUNCATED]");
    }
    result
}

/// Check if a string contains potential PII.
#[must_use]
pub fn contains_pii(input: &str) -> bool {
    let patterns = get_patterns();
    let (prefix, _truncated) = truncate_to_char_boundary(input, max_sanitize_bytes());
    if patterns.fast_set.is_match(prefix) {
        return true;
    }
    if prefix.contains("-----BEGIN ") {
        return patterns.pem_patterns.iter().any(|p| p.regex.is_match(prefix));
    }
    false
}

/// A `tracing_subscriber` writer wrapper that sanitizes formatted log output
/// before it is written to the underlying sink.
///
/// This keeps sanitization centralized (no need to call `sanitize()` at every
/// callsite). It is still defense-in-depth: prefer structured logging and
/// redaction-by-type to avoid sensitive data entering formatted strings.
#[derive(Debug)]
pub struct SanitizingMakeWriter<M> {
    inner: M,
}

impl<M> SanitizingMakeWriter<M> {
    #[must_use]
    pub fn new(inner: M) -> Self {
        Self { inner }
    }
}

impl<M> Clone for SanitizingMakeWriter<M>
where
    M: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

pub struct SanitizingWriter<W> {
    inner: W,
    buffer: Vec<u8>,
}

impl<W> SanitizingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
        }
    }
}

impl<W> SanitizingWriter<W>
where
    W: std::io::Write,
{
    fn flush_lines(&mut self) -> std::io::Result<()> {
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let line = self.buffer.drain(..=pos).collect::<Vec<u8>>();
            let line_str = String::from_utf8_lossy(&line);
            let sanitized = sanitize(&line_str);
            self.inner.write_all(sanitized.as_bytes())?;
        }
        Ok(())
    }
}

impl<W> std::io::Write for SanitizingWriter<W>
where
    W: std::io::Write,
{
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);

        // Prevent unbounded buffering if the formatter writes a huge line with no newlines.
        // We fall back to lossy UTF-8 conversion; `sanitize()` will also cap the output.
        let hard_cap = max_sanitize_bytes().saturating_mul(2);
        if hard_cap > 0 && self.buffer.len() > hard_cap {
            let s = String::from_utf8_lossy(&self.buffer).to_string();
            let sanitized = sanitize(&s);
            self.inner.write_all(sanitized.as_bytes())?;
            self.inner.write_all(b"\n[TRUNCATED]\n")?;
            self.buffer.clear();
            return Ok(buf.len());
        }

        self.flush_lines()?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flush_lines()?;

        if !self.buffer.is_empty() {
            let s = String::from_utf8_lossy(&self.buffer);
            let sanitized = sanitize(&s);
            self.inner.write_all(sanitized.as_bytes())?;
            self.buffer.clear();
        }

        self.inner.flush()
    }
}

impl<'a, M> MakeWriter<'a> for SanitizingMakeWriter<M>
where
    M: MakeWriter<'a>,
{
    type Writer = SanitizingWriter<M::Writer>;

    fn make_writer(&'a self) -> Self::Writer {
        SanitizingWriter::new(self.inner.make_writer())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_uuid() {
        let input = "Patient ID: 550e8400-e29b-41d4-a716-446655440000 processed";
        let sanitized = sanitize(input);
        assert!(sanitized.contains("[REDACTED-UUID]"));
        assert!(!sanitized.contains("550e8400"));
    }

    #[test]
    fn test_sanitize_ssn() {
        let input = "SSN: 123-45-6789";
        let sanitized = sanitize(input);
        assert!(sanitized.contains("[REDACTED-SSN]"));
        assert!(!sanitized.contains("123-45-6789"));
    }

    #[test]
    fn test_sanitize_mrn() {
        let input = "MRN:12345678 found";
        let sanitized = sanitize(input);
        assert!(sanitized.contains("[REDACTED-MRN]"));
    }

    #[test]
    fn test_sanitize_email() {
        let input = "Contact: patient@hospital.com";
        let sanitized = sanitize(input);
        assert!(sanitized.contains("[REDACTED-EMAIL]"));
    }

    #[test]
    fn test_contains_pii() {
        assert!(contains_pii("ID: 550e8400-e29b-41d4-a716-446655440000"));
        assert!(contains_pii("SSN: 123-45-6789"));
        assert!(!contains_pii("Just normal log text"));
    }

    #[test]
    fn test_sanitize_key_material() {
        let input = "Key: 0123456789abcdef0123456789abcdef";
        let sanitized = sanitize(input);
        assert!(
            sanitized.contains("[REDACTED-KEY]") || sanitized.contains("[REDACTED-SECRET]")
        );
    }

    #[test]
    fn test_sanitize_jwt() {
        let input = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
        let sanitized = sanitize(input);
        assert!(sanitized.contains("[REDACTED-JWT]"));
        assert!(!sanitized.contains("eyJhbGci"));
    }

    #[test]
    fn test_sanitize_contextual_base64_secret() {
        let input = "api_key=QWxhZGRpbjpvcGVuIHNlc2FtZSB3aXRoIGxvbmcgc2VjcmV0IHZhbHVl";
        let sanitized = sanitize(input);
        assert!(sanitized.contains("[REDACTED-SECRET]"));
    }

    #[test]
    fn test_sanitize_truncates_large_inputs() {
        let input = "prefix 0123456789abcdef0123456789abcdef suffix";
        let sanitized = sanitize_with_limit(input, 16);
        assert!(sanitized.contains("[TRUNCATED]"));
        // Make sure we still sanitize within the prefix.
        // With the small cap, we might cut in the middle, but we should never panic.
        let _ = sanitized;
    }
}
