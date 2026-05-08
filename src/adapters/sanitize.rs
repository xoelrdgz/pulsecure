use regex::{Regex, RegexSet};
use std::sync::OnceLock;
use tracing_subscriber::fmt::MakeWriter;

static PII_PATTERNS: OnceLock<PiiPatterns> = OnceLock::new();

const DEFAULT_SANITIZE_MAX_BYTES: usize = 16 * 1024;

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

fn get_patterns() -> &'static PiiPatterns {
    PII_PATTERNS.get_or_init(|| {


        let fast_rules: Vec<(&'static str, &'static str)> = vec![

            (
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                "[REDACTED-UUID]",
            ),

            (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]"),

            (r"\bMRN[:\s]?\d{6,10}\b", "[REDACTED-MRN]"),

            (
                r"(?i)\b[a-z0-9](?:[a-z0-9._%+-]{0,62}[a-z0-9])?@(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b",
                "[REDACTED-EMAIL]",
            ),

            (
                r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                "[REDACTED-PHONE]",
            ),

            (
                r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b",
                "[REDACTED-JWT]",
            ),

            (
                r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|passwd|pwd|private[_-]?key|seed|signature|sig|token|key)\b\s*[:=]\s*[A-Za-z0-9+/]{32,}={0,2}\b",
                "[REDACTED-SECRET]",
            ),
            (
                r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|passwd|pwd|private[_-]?key|seed|signature|sig|token|key)\b\s*[:=]\s*[0-9a-fA-F]{16,}\b",
                "[REDACTED-SECRET]",
            ),

            (r"\b[0-9a-fA-F]{32,}\b", "[REDACTED-KEY]"),
        ];



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

#[must_use]
pub fn sanitize(input: &str) -> String {
    sanitize_with_limit(input, max_sanitize_bytes())
}

fn sanitize_with_limit(input: &str, max_bytes: usize) -> String {
    let patterns = get_patterns();

    let (prefix, truncated) = truncate_to_char_boundary(input, max_bytes);

    if !patterns.fast_set.is_match(prefix)
        && !(prefix.contains("-----BEGIN ")
            && patterns
                .pem_patterns
                .iter()
                .any(|p| p.regex.is_match(prefix)))
    {
        let mut out = prefix.to_string();
        if truncated {
            out.push_str(" [TRUNCATED]");
        }
        return out;
    }

    let matched: Vec<usize> = patterns.fast_set.matches(prefix).into_iter().collect();
    let mut result = prefix.to_string();
    for idx in matched {
        let pattern = &patterns.fast_patterns[idx];
        result = pattern
            .regex
            .replace_all(&result, pattern.replacement)
            .to_string();
    }

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

#[must_use]
pub fn contains_pii(input: &str) -> bool {
    let patterns = get_patterns();
    let (prefix, _truncated) = truncate_to_char_boundary(input, max_sanitize_bytes());
    if patterns.fast_set.is_match(prefix) {
        return true;
    }
    if prefix.contains("-----BEGIN ") {
        return patterns
            .pem_patterns
            .iter()
            .any(|p| p.regex.is_match(prefix));
    }
    false
}

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
        assert!(sanitized.contains("[REDACTED-KEY]") || sanitized.contains("[REDACTED-SECRET]"));
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

        let _ = sanitized;
    }
}
